"""Unified ChromaDB (dense) + BM25Okapi (sparse) index with synchronized ids and metadata filters."""
from __future__ import annotations
import pickle
import re
from typing import Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import load_config, resolve
from src.indexing.embedder import Embedder
from src.ingest.chunker import embedding_text
from src.schemas import Chunk, ChunkMetadata, RetrievedChunk

# Fields where a chunk tagged "all" (e.g. multi-channel table) must still match a specific value.
SOFT_FIELDS = {"channel", "segment"}
_TOK = re.compile(r"[a-z0-9$%.]+")


def tokenize(text: str) -> list[str]:
    return [t.strip(".") for t in _TOK.findall(text.lower()) if t.strip(".")]


def matches(md: dict, filters: Optional[dict]) -> bool:
    for k, v in (filters or {}).items():
        have = md.get(k)
        if k in SOFT_FIELDS:
            if have not in (v, "all"):
                return False
        elif have != v:
            return False
    return True


def to_chroma_where(filters: Optional[dict]) -> Optional[dict]:
    conds = []
    for k, v in (filters or {}).items():
        conds.append({k: {"$in": [v, "all"]}} if k in SOFT_FIELDS else {k: {"$eq": v}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class HybridIndex:
    def __init__(self, cfg: dict | None = None, embedder: Embedder | None = None):
        self.cfg = cfg or load_config()
        self._embedder = embedder
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.bm25: BM25Okapi | None = None
        self._col = None
        self.vocab: dict[str, list[str]] = {}

    # -- lazy components
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.cfg)
        return self._embedder

    def _client(self):
        import chromadb
        return chromadb.PersistentClient(path=str(resolve(self.cfg, "chroma")))

    def _set_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.vocab = {f: sorted({getattr(c.metadata, f) for c in chunks}) for f in
                      ("property", "period", "channel", "segment", "doc_type")}

    # -- build / load
    def build(self, chunks: list[Chunk]) -> None:
        self._set_chunks(chunks)
        resolve(self.cfg, "chroma").parent.mkdir(parents=True, exist_ok=True)
        client = self._client()
        try:  # delete_collection (not rmtree): Windows keeps sqlite files locked while a client is open
            client.delete_collection(self.cfg["chroma"]["collection"])
        except Exception:
            pass
        col = client.create_collection(self.cfg["chroma"]["collection"], metadata={"hnsw:space": "cosine"})
        vecs = self.embedder.embed_documents([embedding_text(c) for c in chunks], show_progress=True)
        B = 256
        for i in range(0, len(chunks), B):
            part = chunks[i:i + B]
            col.add(ids=[c.chunk_id for c in part], embeddings=vecs[i:i + B].tolist(),
                    documents=[c.text for c in part],
                    metadatas=[{**c.metadata.model_dump(), "kind": c.kind} for c in part])
        self._col = col
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        with open(resolve(self.cfg, "bm25"), "wb") as f:
            pickle.dump({"ids": [c.chunk_id for c in chunks], "bm25": self.bm25}, f)
        logger.info(f"Index built: {len(chunks)} chunks (chroma={col.count()}, bm25 synced)")

    def load(self, chunks: list[Chunk]) -> "HybridIndex":
        self._set_chunks(chunks)
        with open(resolve(self.cfg, "bm25"), "rb") as f:
            blob = pickle.load(f)
        if blob["ids"] != [c.chunk_id for c in chunks]:
            raise RuntimeError("BM25 index out of sync with chunks.jsonl - rebuild the index")
        self.bm25 = blob["bm25"]
        self._col = self._client().get_collection(self.cfg["chroma"]["collection"])
        if self._col.count() != len(chunks):
            raise RuntimeError("Chroma index out of sync with chunks.jsonl - rebuild the index")
        return self

    # -- search
    def _rc(self, cid: str, score: float) -> RetrievedChunk:
        c = self.by_id[cid]
        return RetrievedChunk(chunk_id=cid, text=c.text, metadata=c.metadata, score=score, kind=c.kind)

    def dense_search(self, query: str, k: int, filters: Optional[dict] = None) -> list[RetrievedChunk]:
        q = self.embedder.embed_queries([query])
        kw = {"where": to_chroma_where(filters)} if filters else {}
        n = min(k, self._col.count())
        r = self._col.query(query_embeddings=q.tolist(), n_results=n, **kw)
        return [self._rc(i, 1.0 - d) for i, d in zip(r["ids"][0], r["distances"][0])]

    def sparse_search(self, query: str, k: int, filters: Optional[dict] = None) -> list[RetrievedChunk]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        mask = np.array([matches(c.metadata.model_dump(), filters) for c in self.chunks])
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [self._rc(self.chunks[i].chunk_id, float(scores[i])) for i in order
                if np.isfinite(scores[i]) and scores[i] > 0]
