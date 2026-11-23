"""YAML config loader. Paths resolve relative to the project root."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> dict:
    """Return a fresh deep copy so callers may mutate (e.g. tests switching to mock)."""
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    return copy.deepcopy(_load(str(p)))


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p
