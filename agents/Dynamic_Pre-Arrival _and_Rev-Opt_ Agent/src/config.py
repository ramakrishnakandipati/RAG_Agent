"""Central Pydantic Settings: paths, thresholds, LLM + GPU configuration (env / .env driven)."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    # LLM
    llm_mode: str = "live"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen3-vl-30b-a3b-instruct"
    llm_api_key: str = "lm-studio"
    llm_timeout: int = 300
    llm_max_tokens: int = 700

    # GPU / models
    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_model_small: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    min_free_vram_gb: float = 2.5

    # Guardrails
    max_discount_pct: float = 20.0
    min_margin_pct: float = 35.0
    propensity_threshold: float = 0.25
    max_retries: int = 2
    max_msgs_per_7d: int = 2
    quiet_hours_start: int = 21
    quiet_hours_end: int = 8

    # Services
    flask_host: str = "127.0.0.1"
    flask_port: int = 5055
    streamlit_port: int = 8511

    seed: int = 42

    @property
    def root(self) -> Path: return ROOT
    @property
    def raw_dir(self) -> Path: return ROOT / "data" / "raw"
    @property
    def processed_dir(self) -> Path: return ROOT / "data" / "processed"
    @property
    def kb_dir(self) -> Path: return ROOT / "data" / "knowledge_base"
    @property
    def chroma_dir(self) -> Path: return ROOT / "data" / "chroma"
    @property
    def models_dir(self) -> Path: return ROOT / "models"
    @property
    def logs_dir(self) -> Path: return ROOT / "logs"
    @property
    def metrics_dir(self) -> Path: return ROOT / "metrics"
    @property
    def eval_dir(self) -> Path: return ROOT / "evaluation"
    @property
    def validation_dir(self) -> Path: return ROOT / "validation"
    @property
    def charts_dir(self) -> Path: return ROOT / "outputs" / "charts"
    @property
    def screens_dir(self) -> Path: return ROOT / "outputs" / "ui_screens"
    @property
    def db_path(self) -> Path: return ROOT / "data" / "processed" / "pms.sqlite"
    @property
    def checkpoint_path(self) -> Path: return ROOT / "data" / "processed" / "checkpoints.sqlite"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.kb_dir, self.models_dir, self.logs_dir,
                  self.metrics_dir, self.eval_dir, self.validation_dir, self.charts_dir, self.screens_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
