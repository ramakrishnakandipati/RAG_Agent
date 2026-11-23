"""Retrieval metrics (HitRate@K, MRR@K, NDCG@K), golden QA builder, latency + faithfulness aggregation."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

from src.schemas import Chunk

METRICS = ("DAILY", "ADR", "RevPAR", "Occupancy")


def build_golden(chunks: list[Chunk]) -> list[dict]:
    """Deterministic golden QA from the indexed corpus. Questions are unfiltered (no 'only ...'), so the
    system must discriminate near-identical KPI tables by property + period + channel semantics.
    Gold = every table chunk (md/pdf/csv copies) for that property+period."""
    tables = [c for c in chunks if c.kind == "table"]
    props = sorted({c.metadata.property for c in tables})
    periods = sorted({c.metadata.period for c in tables})
    channels = ["Direct", "Booking.com OTA", "Expedia OTA", "Corporate", "Wholesale"]
    metric_words = [("ADR", "average daily rate"), ("RevPAR", "revenue per available room"),
                    ("Occupancy", "occupancy")]
    qa = []
    for i, p in enumerate(props):
        for j, per in enumerate(periods):
            ch = channels[(i + j) % len(channels)]
            m_abbr, m_full = metric_words[(i + 2 * j) % 3]
            gold = [c.chunk_id for c in tables if c.metadata.property == p and c.metadata.period == per]
            qa.append({"question": f"What was the {m_full} ({m_abbr}) for the {ch} channel at {p} in {per}?",
                       "gold_ids": gold, "property": p, "period": per, "channel": ch, "metric": m_abbr})
    return qa


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return float(any(r in gold for r in ranked[:k]))


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for i, r in enumerate(ranked[:k], 1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(runs: list[list[str]], golds: list[set[str]], k: int = 5) -> dict:
    n = max(len(runs), 1)
    return {f"hit@{k}": sum(hit_at_k(r, g, k) for r, g in zip(runs, golds)) / n,
            f"mrr@{k}": sum(mrr_at_k(r, g, k) for r, g in zip(runs, golds)) / n,
            f"ndcg@{k}": sum(ndcg_at_k(r, g, k) for r, g in zip(runs, golds)) / n}


def compare_systems(pipe, qa: list[dict], k: int = 5) -> dict:
    """BM25-only vs Dense-only vs RRF vs RRF+cross-encoder rerank (all unfiltered, per-question)."""
    from src.retrieval.query_rewriter import expand_acronyms
    from src.schemas import RewrittenQuery
    sys_runs: dict[str, list[list[str]]] = {"bm25": [], "dense": [], "rrf": [], "rrf+rerank": []}
    golds = [set(q["gold_ids"]) for q in qa]
    for q in qa:
        text, _ = expand_acronyms(q["question"])
        rq = RewrittenQuery(original=q["question"], search_queries=[text])
        for name, mode in (("bm25", "sparse"), ("dense", "dense"), ("rrf", "hybrid")):
            sys_runs[name].append([c.chunk_id for c in pipe.retriever.retrieve(rq, 25, mode=mode)])
        cands = pipe.retriever.retrieve(rq, 25)
        sys_runs["rrf+rerank"].append([c.chunk_id for c in pipe.reranker.rerank(text, cands, k)])
    return {name: aggregate(runs, golds, k) for name, runs in sys_runs.items()}


def write_latency_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["question", "rewrite", "retrieve", "rerank", "generate", "verify", "total"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
