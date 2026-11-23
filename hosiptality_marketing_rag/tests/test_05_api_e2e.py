import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture(scope="module")
def client(pipeline):
    api_main.state["pipeline"] = pipeline
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["index_ready"] and j["chunks"] > 0 and "embedder" in j["models"]


def test_ingest_rebuilds_index(client):
    r = client.post("/ingest", json={"regenerate": False})
    assert r.status_code == 200 and r.json()["chunks"] > 0 and r.json()["tables"] > 0
    assert client.get("/health").json()["index_ready"]


def test_query_returns_grounded_cited_response(client, pipeline):
    r = client.post("/query", json={"question": "What was the ADR for Direct bookings at Miami Beach Resort only Q3 2025?"})
    assert r.status_code == 200
    j = r.json()
    assert j["rewritten"]["filters"]["period"] == "Q3 2025"
    assert len(j["rewritten"]["search_queries"]) <= 3
    assert "$182" in j["answer"] and j["citations"]
    # mock: deterministic extractive answer must be fully faithful. live: LLM-judge may flag valid inferences
    # ("highest RevPAR"), so only numeric support is strict; the overall score gets a lower bound.
    assert not j["verification"]["unsupported_claims"] and not j["verification"]["citation_errors"]
    assert j["verification"]["faithfulness_score"] >= (0.95 if pipeline.llm.mock else 0.75)
    assert all(c["metadata"]["period"] == "Q3 2025" for c in j["contexts"])
    assert set(j["latency_ms"]) >= {"rewrite", "retrieve", "rerank", "generate", "verify", "total"}


def test_query_out_of_scope_refuses(client):
    j = client.post("/query", json={"question": "What was Q3 2025 ADR in Tokyo?"}).json()
    assert j["verification"]["is_refusal"] and "$" not in j["answer"]


def test_query_validation(client):
    assert client.post("/query", json={}).status_code == 422
    assert client.post("/query", json={"question": "  "}).status_code == 422


def test_verify_endpoint_flags_fabrication(client, pipeline):
    ids = [c.chunk_id for c in pipeline.index.chunks if c.metadata.source == "case_study_miami_beach_resort.md"
           and c.metadata.period == "Q3 2025" and c.kind == "table"]
    r = client.post("/verify", json={"answer": "ADR was $299 [case_study_miami_beach_resort.md | 1 | Q3 2025].",
                                     "chunk_ids": ids})
    assert r.status_code == 200 and len(r.json()["unsupported_claims"]) == 1
    assert client.post("/verify", json={"answer": "x", "chunk_ids": ["nope"]}).status_code == 404
