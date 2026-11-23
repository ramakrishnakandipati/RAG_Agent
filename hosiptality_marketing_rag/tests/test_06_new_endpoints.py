import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture(scope="module")
def client(pipeline):
    api_main.state["pipeline"] = pipeline
    with TestClient(api_main.app) as c:
        yield c


def test_meta_filters_lists_vocabulary(client):
    j = client.get("/meta/filters").json()
    assert "Q3 2025" in j["period"] and "Miami Beach Resort" in j["property"]
    assert "Direct" in j["channel"] and set(j["doc_type"]) == {"case_study", "campaign_report", "brand_guidelines"}


def test_meta_examples(client):
    ex = client.get("/meta/examples").json()
    assert len(ex) >= 4 and {"question", "label", "category"} <= set(ex[0])
    assert any(e["category"] == "out_of_scope" for e in ex)


def test_meta_metrics_shape(client):
    j = client.get("/meta/metrics").json()
    assert set(j) >= {"retrieval", "latency", "tests", "mean_faithfulness"}


@pytest.mark.gpu
def test_query_filter_override_isolation(client):
    r = client.post("/query", json={"question": "Show RevPAR by channel",
                                    "filters": {"period": "Q3 2025", "property": "Orlando Grand Hotel"}})
    assert r.status_code == 200
    j = r.json()
    assert j["rewritten"]["filters"]["period"] == "Q3 2025"
    assert j["contexts"] and all(c["metadata"]["period"] == "Q3 2025"
                                 and c["metadata"]["property"] == "Orlando Grand Hotel" for c in j["contexts"])


@pytest.mark.gpu
def test_query_override_beats_extracted_filter(client):
    j = client.post("/query", json={"question": "ADR only Q3 2025", "filters": {"period": "Q1 2025"}}).json()
    assert j["rewritten"]["filters"]["period"] == "Q1 2025"
    assert all(c["metadata"]["period"] == "Q1 2025" for c in j["contexts"])


@pytest.mark.gpu
def test_query_top_k_and_validation(client):
    j = client.post("/query", json={"question": "What was the ADR for Direct at Miami in Q3 2025?", "top_k": 2}).json()
    assert len(j["contexts"]) <= 2
    assert client.post("/query", json={"question": "x", "top_k": 99}).status_code == 422
    assert client.post("/query", json={"question": "x", "filters": {"bogus": "1"}}).status_code == 422


def test_get_chunk(client, pipeline):
    c = next(c for c in pipeline.index.chunks if c.kind == "table")
    j = client.get(f"/chunks/{c.chunk_id}").json()
    assert j["chunk_id"] == c.chunk_id and j["table_md"] == c.table_md and j["metadata"]["period"]
    assert client.get("/chunks/nope::p1::t0").status_code == 404
