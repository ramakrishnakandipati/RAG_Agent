"""Filtered dense + BM25 retrieval over all rewritten queries, fused with Reciprocal Rank Fusion."""
from __future__ import annotations
from collections import defaultdict

from src.indexing.vector_sparse_store import HybridIndex, matches
from src.schemas import RetrievedChunk, RewrittenQuery


def rrf_fuse(rank_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(rank_lists)
    s: dict[str, float] = defaultdict(float)
    for w, lst in zip(weights, rank_lists):
        for r, cid in enumerate(lst, start=1):
            s[cid] += w / (k + r)
    return sorted(s.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, index: HybridIndex, cfg: dict):
        self.index, self.r = index, cfg["retrieval"]

    def retrieve(self, rq: RewrittenQuery, top_n: int | None = None, mode: str = "hybrid") -> list[RetrievedChunk]:
        top_n = top_n or self.r["rerank_in"]
        lists, weights = [], []
        for q in rq.search_queries:
            if mode in ("hybrid", "dense"):
                lists.append([c.chunk_id for c in self.index.dense_search(q, self.r["dense_k"], rq.filters)])
                weights.append(self.r["rrf_weights"]["dense"])
            if mode in ("hybrid", "sparse"):
                lists.append([c.chunk_id for c in self.index.sparse_search(q, self.r["sparse_k"], rq.filters)])
                weights.append(self.r["rrf_weights"]["sparse"])
        fused = rrf_fuse(lists, self.r["rrf_k"], weights)[:top_n]
        out = []
        for cid, sc in fused:
            ch = self.index.by_id[cid]
            if not matches(ch.metadata.model_dump(), rq.filters):  # post-hoc guard: zero filter leakage
                continue
            out.append(RetrievedChunk(chunk_id=cid, text=ch.text, metadata=ch.metadata, score=sc, kind=ch.kind))
        return out
