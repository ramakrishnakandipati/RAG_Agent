"""Shared utilities: config loading, seeding, device selection, GPU telemetry, JSONL logging.

Math: latency = (t1 - t0) in ms after cuda.synchronize(); throughput = N / latency;
VRAM = max_memory_allocated / 2^20 MiB.
"""
from __future__ import annotations
import json, time, random, os
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_policies() -> dict:
    return yaml.safe_load((ROOT / "configs" / "brand_policies.yaml").read_text(encoding="utf-8"))


def path(key: str) -> Path:
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_all(seed: int | None = None) -> None:
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    cfg = load_config()["cuda"]
    if torch.cuda.is_available():
        return torch.device(cfg["device"])
    if cfg.get("require"):
        raise RuntimeError("CUDA required by config but torch.cuda.is_available() is False")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def jsonl_append(name: str, record: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(path("logs") / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def gpu_snapshot(device: torch.device | None = None) -> dict:
    device = device or get_device()
    if device.type != "cuda":
        return {"device": "cpu"}
    return {"device": torch.cuda.get_device_name(0), "vram_alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "vram_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 0)}


@contextmanager
def timed(label: str, n_items: int = 1, log: bool = True):
    """Context manager yielding a dict filled with latency_ms / throughput / VRAM, logged to gpu_telemetry.jsonl."""
    dev = get_device(); out: dict = {}
    sync(dev); t0 = time.perf_counter()
    try:
        yield out
    finally:
        sync(dev); ms = (time.perf_counter() - t0) * 1000
        out.update(label=label, latency_ms=round(ms, 3), throughput_per_s=round(n_items / max(ms / 1000, 1e-9), 1),
                   **gpu_snapshot(dev))
        if log:
            jsonl_append("gpu_telemetry.jsonl", out)


def write_json(name: str, obj, key: str = "metrics") -> Path:
    p = path(key) / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def load_logs():
    """Load simulated reservation/bandit logs (generates them on first use)."""
    import pandas as pd
    p = path("data") / "reservations_logs.parquet"
    if not p.exists():
        import importlib, sys
        sys.path.insert(0, str(ROOT))
        importlib.import_module("data.synthetic_generator").main()
    return pd.read_parquet(p)


def split_idx(n: int, val_frac: float = 0.2, seed: int | None = None):
    rng = np.random.default_rng(load_config()["seed"] if seed is None else seed)
    perm = rng.permutation(n); k = int(n * (1 - val_frac))
    return perm[:k], perm[k:]
