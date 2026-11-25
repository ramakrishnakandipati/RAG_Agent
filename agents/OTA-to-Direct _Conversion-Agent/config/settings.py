"""Central settings: paths, local-LLM endpoint discovery, CUDA detection."""
from __future__ import annotations
import os, shutil, subprocess
from functools import lru_cache
from pathlib import Path
import requests, yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_URLS = ["http://localhost:1234/v1", "http://localhost:8000/v1", "http://localhost:11434/v1"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    LLM_BASE_URL: str = ""
    LLM_MODEL_NAME: str = "qwen3-vl-30b-a3b-instruct"
    LLM_API_KEY: str = "not-needed"
    LLM_TIMEOUT_S: float = 60.0
    FORCE_OFFLINE_LLM: bool = False
    FORCE_CPU: bool = False
    SEED: int = 42
    HOLDOUT_FRAC: float = 0.15
    N_BOOKINGS: int = 20000
    FLASK_PORT: int = 5055
    STREAMLIT_PORT: int = 8511
    OUTPUT_DIR: str = ""             # override for tests / alternate deployments

    @property
    def out(self) -> Path: return Path(self.OUTPUT_DIR) if self.OUTPUT_DIR else ROOT / "outputs"
    def path(self, *parts: str) -> Path:
        p = self.out.joinpath(*parts); p.parent.mkdir(parents=True, exist_ok=True); return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_catalog() -> dict:
    return yaml.safe_load((ROOT / "config" / "offer_catalog.yaml").read_text(encoding="utf-8"))


def discover_llm(timeout: float = 2.0) -> dict:
    """Probe candidate ports via /v1/models. Returns {base_url, model, available}."""
    s = get_settings()
    if s.FORCE_OFFLINE_LLM or os.getenv("PYTEST_CURRENT_TEST") and os.getenv("LLM_ALLOW_LIVE_IN_TESTS") != "1":
        return {"base_url": None, "model": s.LLM_MODEL_NAME, "available": False}
    urls = ([s.LLM_BASE_URL] if s.LLM_BASE_URL else []) + CANDIDATE_URLS
    for u in urls:
        try:
            r = requests.get(u.rstrip("/") + "/models", timeout=timeout)
            if r.ok:
                ids = [m.get("id", "") for m in r.json().get("data", [])]
                model = next((i for i in ids if s.LLM_MODEL_NAME in i), None) or (ids[0] if ids else s.LLM_MODEL_NAME)
                return {"base_url": u, "model": model, "available": True}
        except Exception:
            continue
    return {"base_url": None, "model": s.LLM_MODEL_NAME, "available": False}


def detect_hardware() -> dict:
    info = {"cuda": False, "device_name": None, "cuda_version": None, "vram_total_mb": None, "vram_used_mb": None}
    try:
        import torch
        if torch.cuda.is_available() and not get_settings().FORCE_CPU:
            info.update(cuda=True, device_name=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda,
                        vram_total_mb=int(torch.cuda.get_device_properties(0).total_memory / 2**20),
                        vram_used_mb=int(torch.cuda.memory_allocated(0) / 2**20))
    except Exception:
        pass
    if not info["cuda"] and shutil.which("nvidia-smi") and not get_settings().FORCE_CPU:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            info.update(device_name=out[0].strip(), vram_total_mb=int(out[1]), vram_used_mb=int(out[2]))
        except Exception:
            pass
    return info


def xgb_device() -> str:
    return "cuda" if detect_hardware()["cuda"] else "cpu"
