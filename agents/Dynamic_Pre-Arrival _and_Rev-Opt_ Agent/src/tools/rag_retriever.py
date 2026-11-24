"""GPU-accelerated hybrid retriever: ChromaDB dense (bge, fp16 on CUDA) + BM25 sparse -> RRF -> cross-encoder rerank."""
from __future__ import annotations
import re
import threading
from typing import Optional

from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.utils import device as dev
from src.utils.logger import get_logger

log = get_logger("rag")
_tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())  # noqa: E731


def load_chunks() -> list[dict]:
    """Paragraph-level chunks from knowledge_base/*.md with doc_type metadata."""
    s = get_settings()
    chunks = []
    for p in sorted(s.kb_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"doc_type:\s*(\w+)", text)
        dtype = m.group(1) if m else p.stem
        title = text.splitlines()[0].lstrip("# ").strip()
        lines = [l.strip() for l in text.splitlines()[1:] if l.strip() and not l.startswith("doc_type:")]
        for i, line in enumerate(lines):
            chunks.append({"id": f"{p.stem}-{i}", "text": line, "doc_type": dtype, "source": p.name, "title": title})
    return chunks


class HybridRetriever:
    def __init__(self, rebuild: bool = False):
        self.s = get_settings()
        self.chunks = load_chunks()
        self.bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])
        self.device = dev.get_device()
        self.embed_name = self.s.embed_model
        if self.device == "cuda" and dev.free_vram_gb() < self.s.min_free_vram_gb:
            log.warning("low_vram_fallback", free_gb=dev.free_vram_gb(), embed=self.s.embed_model_small)
            self.embed_name = self.s.embed_model_small
        self._embedder = None
        self._reranker = None
        self._lock = threading.Lock()
        self._build_index(rebuild)

    # --- models (lazy singletons)
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.embed_name, device=self.device)
            if self.device == "cuda":
                m.half()
            self._embedder = m
            log.info("embedder_loaded", model=self.embed_name, device=self.device, fp16=self.device == "cuda")
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            kw = {"model_kwargs": {"torch_dtype": dev.torch_dtype()}}
            self._reranker = CrossEncoder(self.s.rerank_model, device=self.device, **kw)
            log.info("reranker_loaded", model=self.s.rerank_model, device=self.device)
        return self._reranker

    def _embed(self, texts: list[str]):
        return self.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _build_index(self, rebuild: bool):
        import chromadb
        self.s.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.s.chroma_dir))
        name = "hotel_kb_" + re.sub(r"\W", "_", self.embed_name.split("/")[-1])
        if rebuild:
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        if self.col.count() != len(self.chunks):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass
            self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
            self.col.add(ids=[c["id"] for c in self.chunks], documents=[c["text"] for c in self.chunks],
                         embeddings=self._embed([c["text"] for c in self.chunks]),
                         metadatas=[{"doc_type": c["doc_type"], "source": c["source"], "title": c["title"]} for c in self.chunks])
            log.info("chroma_indexed", n=len(self.chunks), collection=name)

    def search(self, query: str, k: int = 4, doc_types: Optional[list[str]] = None, rerank: bool = True) -> list[dict]:
        pool = max(k * 3, 10)
        where = {"doc_type": {"$in": doc_types}} if doc_types else None
        with self._lock:
            dense = self.col.query(query_embeddings=self._embed([query]), n_results=min(pool, self.col.count()), where=where)
        dense_ids = dense["ids"][0]
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        sparse_ids = [self.chunks[i]["id"] for i in order if (not doc_types or self.chunks[i]["doc_type"] in doc_types)][:pool]
        fused: dict[str, float] = {}
        for ranking in (dense_ids, sparse_ids):
            for r, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + r + 1)
        by_id = {c["id"]: c for c in self.chunks}
        cand = [dict(by_id[i], rrf=round(v, 5)) for i, v in sorted(fused.items(), key=lambda x: -x[1])][:pool]
        if rerank and cand:
            with self._lock:
                sc = self.reranker.predict([(query, c["text"]) for c in cand], show_progress_bar=False)
            for c, v in zip(cand, sc):
                c["rerank"] = round(float(v), 4)
            cand.sort(key=lambda c: -c["rerank"])
        return cand[:k]


_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
