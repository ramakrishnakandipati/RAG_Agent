"""Thin HTTP client for the RAG FastAPI server (the UI never touches GPU models directly)."""
from __future__ import annotations
import os
from typing import Optional

import requests

BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    pass


def _call(method: str, path: str, timeout: float = 300, **kw):
    try:
        r = requests.request(method, BASE_URL + path, timeout=timeout, **kw)
    except requests.ConnectionError as e:
        raise ApiError(f"Cannot reach the RAG API at {BASE_URL}. Start it with: uvicorn api.main:app --port 8000") from e
    except requests.Timeout as e:
        raise ApiError("The API timed out (first query loads the GPU models; try again).") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ApiError(f"{r.status_code}: {detail}")
    return r.json()


def health() -> dict:
    return _call("GET", "/health", timeout=15)


def filters() -> dict:
    return _call("GET", "/meta/filters", timeout=15)


def examples() -> list[dict]:
    return _call("GET", "/meta/examples", timeout=15)


def metrics() -> dict:
    return _call("GET", "/meta/metrics", timeout=15)


def chunk(chunk_id: str) -> dict:
    return _call("GET", f"/chunks/{chunk_id}", timeout=15)


def query(question: str, flt: Optional[dict] = None, top_k: Optional[int] = None) -> dict:
    body: dict = {"question": question}
    if flt:
        body["filters"] = flt
    if top_k:
        body["top_k"] = top_k
    return _call("POST", "/query", json=body)


def verify(answer: str, chunk_ids: list[str]) -> dict:
    return _call("POST", "/verify", json={"answer": answer, "chunk_ids": chunk_ids}, timeout=120)
