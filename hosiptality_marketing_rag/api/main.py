"""FastAPI server: /health, /ingest, /query, /verify.
Run:  uvicorn api.main:app --port 8000      (set RAG_LLM_MODE=mock to run without the Qwen server)"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import load_config, resolve
from src.pipeline import RAGPipeline
from src.schemas import QueryResponse, VerificationResult

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


class QueryRequest(BaseModel):
    question: str


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
    return p.query(req.question)


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
