"""Start the FastAPI server and the Streamlit UI together.

    python scripts/run_ui.py            # live Qwen if reachable, else mock
    python scripts/run_ui.py --mock     # force the deterministic mock LLM
Open http://localhost:8501  (API docs: http://localhost:8000/docs).  Ctrl+C stops both.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API_PORT, UI_PORT = 8000, 8501
API_URL, UI_URL = f"http://127.0.0.1:{API_PORT}", f"http://127.0.0.1:{UI_PORT}"


def pick_mode(force_mock: bool) -> str:
    if force_mock:
        return "mock"
    from src.config import load_config
    from src.llm_client import LLMClient
    return "live" if LLMClient(load_config()).is_up() else "mock"


def _wait(url: str, timeout: float, what: str) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"{what} did not become ready within {timeout:.0f}s")


class Stack:
    """Context manager that runs uvicorn + streamlit as child processes and cleans them up."""

    def __init__(self, mode: str, warmup: bool = True):
        self.mode, self.warmup, self.procs = mode, warmup, []
        self.logs = ROOT / "outputs" / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def _spawn(self, args: list[str], log: str, env: dict) -> None:
        f = open(self.logs / log, "w", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.procs.append(subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=flags))

    def __enter__(self) -> "Stack":
        env = {**os.environ, "RAG_LLM_MODE": self.mode, "RAG_API_URL": API_URL, "PYTHONUTF8": "1"}
        self._spawn([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(API_PORT)], "api_server.log", env)
        _wait(f"{API_URL}/health", 120, "API")
        if self.warmup:  # loads embedder + reranker so the first UI query is not a 40 s stall
            requests.post(f"{API_URL}/query", json={"question": "warm up ADR Q3 2025"}, timeout=600)
        self._spawn([sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", str(UI_PORT),
                     "--server.headless", "true"], "ui_server.log", env)
        _wait(f"{UI_URL}/_stcore/health", 60, "Streamlit")
        return self

    def __exit__(self, *exc) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    mode = pick_mode(a.mock)
    print(f"LLM mode: {mode}. Starting API and UI (models warm up first)...")
    with Stack(mode, warmup=not a.no_warmup):
        print(f"UI:  {UI_URL}\nAPI: {API_URL}/docs\nCtrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")


if __name__ == "__main__":
    main()
