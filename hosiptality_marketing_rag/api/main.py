"""FastAPI server: /health, /ingest, /query, /verify.
Run:  uvicorn api.main:app --port 8000      (set RAG_LLM_MODE=mock to run without the Qwen server)"""
from __future__ import annotations
import csv
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config, resolve
from src.pipeline import RAGPipeline
from api.examples import EXAMPLES
from src.schemas import ChunkDetail, FilterOptions, MetricsSummary, QueryResponse, VerificationResult

state: dict = {"pipeline": None}


def get_pipeline() -> RAGPipeline:
    if state["pipeline"] is None:
        cfg = load_config()
        if os.getenv("RAG_LLM_MODE"):
            cfg["llm"]["mode"] = os.environ["RAG_LLM_MODE"]
        p = RAGPipeline(cfg)
        if resolve(cfg, "bm25").exists() and resolve(cfg, "chunks").exists():
            p.load_index()
        state["pipeline"] = p
    return state["pipeline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield


app = FastAPI(title="Hospitality Marketing RAG", version="1.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    regenerate: bool = False


FILTER_KEYS = {"property", "period", "channel", "segment", "doc_type"}


class QueryRequest(BaseModel):
    question: str
    filters: dict[str, str] | None = None   # explicit overrides (win over filters extracted from the question)
    top_k: int | None = Field(default=None, ge=1, le=10)


class VerifyRequest(BaseModel):
    answer: str
    chunk_ids: list[str]


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    import torch
    return {"status": "ok" if p.ready else "index_not_built", "index_ready": p.ready,
            "chunks": len(p.index.chunks), "llm_mode": p.llm.cfg["mode"], "llm_reachable": p.llm.is_up(),
            "cuda": torch.cuda.is_available(),
            "models": {"embedder": p.cfg["models"]["embedder"], "reranker": p.cfg["models"]["reranker"],
                       "llm": p.llm.cfg["model"]}}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    return get_pipeline().ingest(regenerate=req.regenerate)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    p = get_pipeline()
    if not p.ready:
        raise HTTPException(409, "Index not built - call POST /ingest first")
    if not req.question.strip():
        raise HTTPException(422, "question must not be empty")
    bad = set(req.filters or {}) - FILTER_KEYS
    if bad:
        raise HTTPException(422, f"unknown filter keys: {sorted(bad)}; allowed: {sorted(FILTER_KEYS)}")
    return p.query(req.question, overrides=req.filters, top_k=req.top_k)


@app.post("/verify", response_model=VerificationResult)
def verify(req: VerifyRequest) -> VerificationResult:
    p = get_pipeline()
    if not p.ready:
        raise HTTPException(409, "Index not built - call POST /ingest first")
    missing = [i for i in req.chunk_ids if i not in p.index.by_id]
    if missing:
        raise HTTPException(404, f"unknown chunk ids: {missing}")
    from src.schemas import RetrievedChunk
    ctx = [RetrievedChunk(chunk_id=i, text=p.index.by_id[i].text, metadata=p.index.by_id[i].metadata)
           for i in req.chunk_ids]
    return p.verifier.verify(req.answer, ctx)


# ---------------------------------------------------------------- metadata endpoints used by the UI
@app.get("/meta/filters", response_model=FilterOptions)
def meta_filters() -> FilterOptions:
    p = get_pipeline()
    if not p.ready:
        raise HTTPException(409, "Index not built - call POST /ingest first")
    return FilterOptions(**p.index.vocab)


@app.get("/meta/examples")
def meta_examples() -> list[dict]:
    return EXAMPLES


@app.get("/meta/metrics", response_model=MetricsSummary)
def meta_metrics() -> MetricsSummary:
    out = resolve(get_pipeline().cfg, "outputs") / "metrics"
    res: dict = {}
    rm = out / "retrieval_metrics.json"
    if rm.exists():
        j = json.loads(rm.read_text(encoding="utf-8"))
        res.update(llm_mode=j.get("llm_mode"), retrieval=j.get("retrieval", {}),
                   mean_faithfulness=j.get("mean_faithfulness"), ingest=j.get("ingest", {}))
    lc = out / "latency_breakdown.csv"
    if lc.exists():
        with open(lc, encoding="utf-8", newline="") as f:
            res["latency"] = [{k: (float(v) if k != "question" else v) for k, v in r.items()}
                              for r in csv.DictReader(f)]
    tr = out / "test_report.json"
    if tr.exists():
        t = json.loads(tr.read_text(encoding="utf-8"))
        res["tests"] = {**t.get("summary", {}), "duration_s": round(t.get("duration", 0), 1)}
    return MetricsSummary(**res)


@app.get("/chunks/{chunk_id:path}", response_model=ChunkDetail)
def get_chunk(chunk_id: str) -> ChunkDetail:
    p = get_pipeline()
    c = p.index.by_id.get(chunk_id)
    if c is None:
        raise HTTPException(404, f"unknown chunk id: {chunk_id}")
    return ChunkDetail(chunk_id=c.chunk_id, kind=c.kind, text=c.text, table_md=c.table_md, metadata=c.metadata)
