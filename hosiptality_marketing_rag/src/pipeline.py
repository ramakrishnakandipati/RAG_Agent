"""End-to-end orchestrator: ingest -> index -> rewrite -> retrieve -> rerank -> answer -> verify."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from src.config import load_config, resolve
from src.generation.generator import REFUSAL, Generator, out_of_scope
from src.generation.verifier import Verifier
from src.indexing.vector_sparse_store import HybridIndex
from src.ingest.chunker import build_chunks, load_chunks
from src.ingest.data_generator import generate
from src.llm_client import LLMClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import Reranker
from src.schemas import Claim, QueryResponse, VerificationResult
from src.generation.generator import cite_tag
from src.generation.verifier import CITE_RE

_log_ready = False


def setup_logging(cfg: dict, name: str = "pipeline_execution.log") -> None:
    global _log_ready
    if _log_ready:
        return
    d = resolve(cfg, "outputs") / "logs"
    d.mkdir(parents=True, exist_ok=True)
    logger.add(d / name, level="INFO", encoding="utf-8", mode="w",
               format="{time:HH:mm:ss.SSS} | {level:<7} | {message}")
    _log_ready = True


class RAGPipeline:
    def __init__(self, cfg: dict | None = None, llm: LLMClient | None = None):
        self.cfg = cfg or load_config()
        setup_logging(self.cfg)
        self.llm = llm or LLMClient(self.cfg)
        self.index = HybridIndex(self.cfg)
        self._reranker: Optional[Reranker] = None
        self.retriever: Optional[HybridRetriever] = None
        self.rewriter = QueryRewriter(self.llm, self.cfg["retrieval"]["max_queries"])
        self.generator = Generator(self.llm)
        self.verifier = Verifier(self.llm)
        self.art_dir = resolve(self.cfg, "outputs") / "stage_artifacts"
        self.art_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ ingest / index
    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.cfg)
        return self._reranker

    def ingest(self, regenerate: bool = False, raw_dir: Path | None = None) -> dict:
        t = {}
        raw = Path(raw_dir) if raw_dir else resolve(self.cfg, "raw")
        if regenerate or not raw.exists() or not any(raw.iterdir()):
            t0 = time.perf_counter(); generate(raw, self.cfg["seed"]); t["generate"] = time.perf_counter() - t0
        t0 = time.perf_counter(); chunks = build_chunks(self.cfg, raw); t["ingest_chunk"] = time.perf_counter() - t0
        self._dump("01_chunks", [{"id": c.chunk_id, "kind": c.kind, **c.metadata.model_dump(),
                                  "tokens_text": len(c.text.split())} for c in chunks])
        t0 = time.perf_counter(); self.index.build(chunks); t["embed_index"] = time.perf_counter() - t0
        self.retriever = HybridRetriever(self.index, self.cfg)
        logger.info(f"ingest complete: {len(chunks)} chunks, timings={ {k: round(v, 2) for k, v in t.items()} }")
        return {"chunks": len(chunks), "tables": sum(c.kind == "table" for c in chunks),
                "seconds": {k: round(v, 2) for k, v in t.items()}}

    def load_index(self) -> "RAGPipeline":
        self.index.load(load_chunks(self.cfg))
        self.retriever = HybridRetriever(self.index, self.cfg)
        return self

    @property
    def ready(self) -> bool:
        return self.retriever is not None

    def _dump(self, name: str, obj) -> None:
        with open(self.art_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.model_dump())

    # ------------------------------------------------------------------ query
    def query(self, question: str) -> QueryResponse:
        assert self.ready, "index not loaded - call ingest() or load_index()"
        lat: dict[str, float] = {}
        r = self.cfg["retrieval"]

        t = time.perf_counter()
        rq = self.rewriter.rewrite(question, self.index.vocab)
        lat["rewrite"] = (time.perf_counter() - t) * 1000
        self._dump("02_rewritten_query", rq)

        t = time.perf_counter()
        cands = self.retriever.retrieve(rq, r["rerank_in"])
        lat["retrieve"] = (time.perf_counter() - t) * 1000
        self._dump("03_rrf_candidates", [{"id": c.chunk_id, "rrf": c.score, **c.metadata.model_dump()} for c in cands])

        t = time.perf_counter()
        top = self.reranker.rerank(rq.search_queries[0], cands, r["final_k"])
        lat["rerank"] = (time.perf_counter() - t) * 1000
        self._dump("04_reranked_top5", [{"id": c.chunk_id, "score": c.score, **c.metadata.model_dump()} for c in top])

        t = time.perf_counter()
        unknown = out_of_scope(question, self.index.vocab)
        weak = (not top) or top[0].score < r["min_rerank_score"]
        if unknown or weak:
            logger.info(f"short-circuit refusal (unknown_entity={unknown}, weak_evidence={weak})")
            answer = REFUSAL
        else:
            answer = self.generator.answer(question, top)
        lat["generate"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        ver = self.verifier.verify(answer, top)
        lat["verify"] = (time.perf_counter() - t) * 1000
        self._dump("05_answer_verification", {"answer": answer, "verification": ver})

        citations = sorted({f"[{a.strip()} | {p} | {per.strip()}]" for a, p, per in CITE_RE.findall(answer)})
        lat["total"] = sum(lat.values())
        logger.info(f"Q={question!r} filters={rq.filters} faith={ver.faithfulness_score} lat={ {k: round(v) for k, v in lat.items()} }ms")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return QueryResponse(question=question, rewritten=rq, answer=answer, citations=citations, contexts=top,
                             verification=ver, latency_ms={k: round(v, 1) for k, v in lat.items()})
