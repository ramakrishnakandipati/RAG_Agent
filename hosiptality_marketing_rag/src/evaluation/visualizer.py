"""High-res PNG charts: latency waterfall, retrieval comparison, embedding projection."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAL = {"teal": "#0F766E", "sand": "#D6A24A", "slate": "#475569", "rose": "#BE123C", "blue": "#2563EB"}
STAGES = ["rewrite", "retrieve", "rerank", "generate", "verify"]


def latency_waterfall(rows: list[dict], path: Path) -> Path:
    """rows: list of latency dicts (ms) per query. Plots mean per-stage cumulative waterfall."""
    mean = {s: float(np.mean([r.get(s, 0) for r in rows])) for s in STAGES}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    left = 0.0
    colors = [PAL["teal"], PAL["blue"], PAL["sand"], PAL["slate"], PAL["rose"]]
    for s, c in zip(STAGES, colors):
        ax.barh(0, mean[s], left=left, color=c, label=f"{s} ({mean[s]:.0f} ms)", height=0.5)
        left += mean[s]
    ax.set_yticks([]); ax.set_xlabel("milliseconds (mean over queries)")
    ax.set_title(f"Per-stage latency waterfall - total {left:.0f} ms")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def retrieval_comparison(results: dict, path: Path, k: int = 5) -> Path:
    systems = list(results)
    metrics = [f"hit@{k}", f"mrr@{k}", f"ndcg@{k}"]
    x = np.arange(len(metrics)); w = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    for i, (s, c) in enumerate(zip(systems, PAL.values())):
        vals = [results[s][m] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=s, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x + w * (len(systems) - 1) / 2); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.12); ax.set_title("Retrieval quality: BM25 vs Dense vs RRF vs RRF + Cross-Encoder")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def embedding_projection(index, path: Path) -> Path:
    """2-D UMAP (PCA fallback) of the stored dense embeddings, coloured by period."""
    got = index._col.get(include=["embeddings", "metadatas"])
    X = np.asarray(got["embeddings"]); metas = got["metadatas"]
    try:
        import umap
        Z = umap.UMAP(n_neighbors=min(10, len(X) - 1), min_dist=0.2, random_state=42).fit_transform(X)
        name = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        Z = PCA(2, random_state=42).fit_transform(X); name = "PCA"
    periods = sorted({m["period"] for m in metas})
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(periods):
        idx = [j for j, m in enumerate(metas) if m["period"] == p]
        mk = ["o" if metas[j]["kind"] == "table" else "^" for j in idx]
        for j, m in zip(idx, mk):
            ax.scatter(Z[j, 0], Z[j, 1], color=cmap(i), marker=m, s=45, alpha=0.85)
        ax.scatter([], [], color=cmap(i), label=p)
    ax.scatter([], [], color="k", marker="o", label="table chunk"); ax.scatter([], [], color="k", marker="^", label="text chunk")
    ax.set_title(f"{name} of bge-large-en-v1.5 chunk embeddings (colour = period)")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.set_xticks([]); ax.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
