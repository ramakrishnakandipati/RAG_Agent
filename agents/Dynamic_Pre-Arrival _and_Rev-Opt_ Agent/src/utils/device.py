"""Centralised GPU detection with automatic CPU fallback (torch, sentence-transformers, XGBoost)."""
from __future__ import annotations
from functools import lru_cache
from typing import Any
from src.utils.logger import get_logger

log = get_logger("device")


@lru_cache
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def gpu_info() -> dict[str, Any]:
    if not cuda_available():
        return {"cuda": False, "device": "cpu"}
    import torch
    free, total = torch.cuda.mem_get_info()
    p = torch.cuda.get_device_properties(0)
    return {"cuda": True, "device": "cuda", "name": p.name,
            "total_vram_gb": round(total / 1e9, 2), "free_vram_gb": round(free / 1e9, 2),
            "torch": torch.__version__}


def log_device() -> dict[str, Any]:
    info = gpu_info()
    log.info("device_detected", **info)
    return info


def free_vram_gb() -> float:
    return gpu_info().get("free_vram_gb", 0.0) if cuda_available() else 0.0


def torch_dtype():
    import torch
    return torch.float16 if cuda_available() else torch.float32


def xgb_device_params() -> dict[str, str]:
    """XGBoost: hist tree method; device=cuda when available, else cpu."""
    return {"tree_method": "hist", "device": get_device()}
