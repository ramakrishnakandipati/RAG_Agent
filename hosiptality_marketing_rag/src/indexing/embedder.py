"""GPU BGE embedding wrapper (FP16, batching, VRAM logging)."""
from __future__ import annotations
import numpy as np
import torch
from loguru import logger

from src.config import load_config

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def log_vram(tag: str) -> float:
    if not torch.cuda.is_available():
        return 0.0
    used = torch.cuda.memory_allocated() / 2**20
    logger.info(f"[VRAM] {tag}: allocated={used:.0f} MiB reserved={torch.cuda.memory_reserved() / 2**20:.0f} MiB")
    return used


class Embedder:
    def __init__(self, cfg: dict | None = None):
        from sentence_transformers import SentenceTransformer
        cfg = cfg or load_config()
        m = cfg["models"]
        dev = m["device"] if torch.cuda.is_available() else "cpu"
        self.batch_size = m["embed_batch_size"]
        self.model = SentenceTransformer(m["embedder"], device=dev)
        if dev == "cuda" and m.get("fp16", True):
            self.model.half()
        self.model.max_seq_length = 512
        self.device = dev
        log_vram("embedder loaded")

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        v = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=show_progress, convert_to_numpy=True)
        log_vram(f"embedded {len(texts)} docs")
        return v.astype(np.float32)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.embed_documents([QUERY_INSTRUCTION + q for q in queries])

    def unload(self) -> None:
        del self.model
        torch.cuda.empty_cache()
