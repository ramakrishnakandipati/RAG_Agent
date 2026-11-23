"""Streamlit AppTest smoke tests: the UI is exercised against a faked API client (no GPU, no server)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui import api_client as api
from ui.components import render_answer_html

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")

META = {"property": "Miami Beach Resort", "period": "Q3 2025", "channel": "all", "segment": "all",
        "doc_type": "case_study", "source": "case_study_miami_beach_resort.md", "section": "s", "page": 1}
OK = {"question": "q", "answer": "**Direct ADR**\n• ADR was $182 [case_study_miami_beach_resort.md | 1 | Q3 2025].",
      "citations": ["[case_study_miami_beach_resort.md | 1 | Q3 2025]"],
      "rewritten": {"original": "q", "search_queries": ["average daily rate"], "filters": {"period": "Q3 2025"},
                    "expansions": {"ADR": "average daily rate"}},
      "contexts": [{"chunk_id": "a::p1::t1", "text": "Row summaries:\nMiami: ADR $182", "kind": "table", "score": 0.99, "metadata": META}],
      "verification": {"claims": [{"text": "t", "value": "$182", "supported": True, "evidence_chunk": "a::p1::t1"}],
                       "unsupported_claims": [], "citation_errors": [], "judge_unsupported": [],
                       "faithfulness_score": 1.0, "is_refusal": False},
      "latency_ms": {"rewrite": 100, "retrieve": 50, "rerank": 80, "generate": 900, "verify": 20, "total": 1150}}
REFUSE = {**OK, "answer": "Insufficient information: the available documents do not contain the data.",
          "contexts": [], "citations": [],
          "verification": {**OK["verification"], "claims": [], "is_refusal": True}}


@pytest.fixture
def fake_api(monkeypatch):
    calls = {}
    monkeypatch.setattr(api, "health", lambda: {"status": "ok", "index_ready": True, "chunks": 58, "llm_mode": "mock", "cuda": True})
    monkeypatch.setattr(api, "filters", lambda: {"property": ["Miami Beach Resort"], "period": ["Q3 2025"], "channel": ["Direct"],
                                                  "segment": ["all"], "doc_type": ["case_study"]})
    monkeypatch.setattr(api, "examples", lambda: [{"id": "e1", "label": "Miami ADR", "category": "answerable", "question": "Miami ADR?"},
                                                   {"id": "e2", "label": "Tokyo", "category": "out_of_scope", "question": "Tokyo ADR?"}])
    monkeypatch.setattr(api, "metrics", lambda: {"retrieval": {"bm25": {"hit@5": 1.0, "mrr@5": 0.62}}, "tests": {"passed": 41, "total": 41},
                                                 "latency": [], "mean_faithfulness": 0.96, "ingest": {"chunks": 58}})
    monkeypatch.setattr(api, "chunk", lambda cid: {"table_md": "| a |\n|---|\n| $182 |"})

    def fake_query(q, flt=None, top_k=None):
        calls["query"] = (q, flt, top_k)
        return REFUSE if "Tokyo" in q else OK
    monkeypatch.setattr(api, "query", fake_query)
    return calls


def test_render_answer_html_escapes_and_chips():
    h = render_answer_html("**T**\n• <b>x</b> $5 [a.md | 2 | Q1 2025].")
    assert "&lt;b&gt;" in h and 'class="chip"' in h and "a.md" in h and "<li>" in h


def test_ui_example_then_ask_shows_answer(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    next(b for b in at.sidebar.button if b.label == "Miami ADR").click().run()
    assert at.text_area[0].value == "Miami ADR?"
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    assert fake_api["query"][0] == "Miami ADR?"
    html = " ".join(m.value for m in at.markdown)
    assert "answer-card" in html and "$182" in html and "1.00" in html


def test_ui_filter_overrides_are_sent(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.selectbox[1].select("Q3 2025")   # Period
    at.text_area[0].set_value("ADR?").run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert fake_api["query"][1] == {"period": "Q3 2025"}


def test_ui_out_of_scope_shows_refusal(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].set_value("Tokyo ADR?").run()
    next(b for b in at.button if b.label == "Ask").click().run()
    html = " ".join(m.value for m in at.markdown)
    assert "refusal" in html and "Out of scope" in html


def test_ui_api_down_shows_error(monkeypatch):
    def boom():
        raise api.ApiError("Cannot reach the RAG API")
    monkeypatch.setattr(api, "health", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("Cannot reach" in e.value for e in at.error)
