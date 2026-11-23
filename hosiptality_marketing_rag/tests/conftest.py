"""Shared fixtures: mock/live Qwen toggle (--live-llm), session pipeline, golden hospitality QA set."""
from __future__ import annotations
import pytest
import torch

from src.config import load_config, resolve
from src.pipeline import RAGPipeline
from src.ingest.chunker import load_chunks
from src.evaluation.metrics import build_golden


def pytest_addoption(parser):
    parser.addoption("--live-llm", action="store_true", default=False,
                     help="Use the real local Qwen3-Coder endpoint instead of the deterministic mock")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA (embedder / cross-encoder)")
    config.addinivalue_line("markers", "live: requires the live Qwen3-Coder endpoint")


def pytest_collection_modifyitems(config, items):
    live = config.getoption("--live-llm")
    for it in items:
        if "live" in it.keywords and not live:
            it.add_marker(pytest.mark.skip(reason="needs --live-llm"))
        if "gpu" in it.keywords and not torch.cuda.is_available():
            it.add_marker(pytest.mark.skip(reason="CUDA not available"))


class FakeLLM:
    """Stand-in for LLMClient returning canned JSON (used to test guardrails against a sloppy model)."""
    mock = False

    def __init__(self, payload: dict):
        self.payload = payload
        self.cfg = {"mode": "fake", "model": "fake"}

    def json(self, *a, **k):
        return self.payload

    def chat(self, *a, **k):
        return ""

    def is_up(self):
        return True


@pytest.fixture(scope="session")
def cfg(request):
    c = load_config()
    c["llm"]["mode"] = "live" if request.config.getoption("--live-llm") else "mock"
    return c


@pytest.fixture(scope="session")
def pipeline(cfg):
    p = RAGPipeline(cfg)
    if resolve(cfg, "bm25").exists() and resolve(cfg, "chunks").exists():
        p.load_index()
    else:
        p.ingest(regenerate=True)
    return p


@pytest.fixture(scope="session")
def chunks(pipeline):
    return load_chunks(pipeline.cfg)


@pytest.fixture(scope="session")
def golden(chunks):
    return build_golden(chunks)
