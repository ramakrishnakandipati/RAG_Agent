"""GPU cross-encoder reranker (BAAI/bge-reranker-large, FP16)."""
from __future__ import annotations
import torch

from src.config import load_config
from src.indexing.embedder import log_vram
from src.schemas import RetrievedChunk


class Reranker:
    def __init__(self, cfg: dict | None = None):
        from sentence_transformers import CrossEncoder
        cfg = cfg or load_config()
        m = cfg["models"]
        dev = m["device"] if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(m["reranker"], device=dev, max_length=512)
        if dev == "cuda" and m.get("fp16", True):
            self.model.model.half()
        self.batch_size = m["rerank_batch_size"]
        log_vram("reranker loaded")

    def rerank(self, query: str, cands: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not cands:
            return []
        scores = self.model.predict([(query, c.text) for c in cands], batch_size=self.batch_size,
                                    show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda x: -float(x[1]))[:top_k]
        return [c.model_copy(update={"score": float(s)}) for c, s in ranked]
