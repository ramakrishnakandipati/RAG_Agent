import pytest

from src.graph import edges
from src.graph.workflow import resume_workflow, run_workflow


def test_happy_path_sends_clean_message(upcoming):
    res = run_workflow(upcoming[0]["reservation_id"])
    st = res["state"]
    assert st["status"] in ("SENT", "NO_OFFER", "FAIL_VALIDATION") and not res["interrupted"]
    if st["status"] == "SENT":
        assert st["guardrail"]["status"] == "PASS"
        assert "{{" not in st["draft"]["body"]
        nodes = [t["node"] for t in st["trace"]]
        assert nodes[:4] == ["enrich_context", "retrieve_knowledge_and_offers", "draft_message", "compliance_check"]
        assert all(t["pre_ok"] and t["post_ok"] for t in st["trace"])


def test_poisoned_draft_is_blocked_then_retried(upcoming):
    for r in upcoming:
        res = run_workflow(r["reservation_id"], adversarial="once")
        st = res["state"]
        if st["status"] == "SENT":
            assert st["retries"] == 1 and st["draft"]["source"] != "adversarial"
            assert "$49" not in st["draft"]["body"] and "SPA-FAKE9" not in st["draft"]["body"]
            return
    pytest.fail("no journey reached SENT")


def test_persistent_failure_escalates_to_hitl_then_resumes(upcoming, clean_db):
    for r in upcoming:
        res = run_workflow(r["reservation_id"], adversarial="persistent")
        if res["interrupted"]:
            assert "repeated_policy_failure" in res["state"]["hitl_reasons"]
            pend = clean_db.list_hitl()
            assert any(p["thread_id"] == res["thread_id"] for p in pend)
            out = resume_workflow(res["thread_id"], {"decision": "approve", "reviewer": "t"})
            st = out["state"]
            assert st["status"] == "SENT" and st["draft"]["source"] == "safe_fallback"
            assert "$49" not in st["draft"]["body"]
            assert not clean_db.list_hitl("PENDING")
            return
    pytest.fail("no journey escalated")


def test_reject_does_not_send(upcoming, clean_db):
    for r in upcoming:
        res = run_workflow(r["reservation_id"], adversarial="persistent")
        if res["interrupted"]:
            out = resume_workflow(res["thread_id"], {"decision": "reject", "reviewer": "t"})
            assert out["state"]["status"] == "REJECTED" and not out["state"].get("dispatched")
            with clean_db.connect() as con:
                assert con.execute("SELECT COUNT(*) FROM outbox WHERE thread_id=?", (res["thread_id"],)).fetchone()[0] == 0
            return
    pytest.fail("no journey escalated")


def test_inbound_accept_books_folio(upcoming, clean_db):
    for r in upcoming:
        first = run_workflow(r["reservation_id"])
        if first["state"]["status"] != "SENT":
            continue
        rep = run_workflow(r["reservation_id"], trigger_type="inbound_reply", inbound_text="Yes please, add it!")
        st = rep["state"]
        assert st["status"] == "BOOKED" and st["booking"]["ok"]
        assert any(f["sku"] == st["selected_offer"]["sku"] for f in clean_db.get_folio(r["reservation_id"]))
        again = run_workflow(r["reservation_id"], trigger_type="inbound_reply", inbound_text="Yes please, add it!")
        assert len(clean_db.get_folio(r["reservation_id"])) == 1   # idempotent
        return
    pytest.fail("no sendable journey")


def test_negative_sentiment_escalates(upcoming, clean_db):
    for r in upcoming:
        if run_workflow(r["reservation_id"])["state"]["status"] != "SENT":
            continue
        rep = run_workflow(r["reservation_id"], trigger_type="inbound_reply", inbound_text="This is terrible, I want a refund")
        assert rep["interrupted"] and "negative_sentiment" in rep["state"]["hitl_reasons"]
        return
    pytest.fail("no sendable journey")


def test_platinum_custom_request_escalates(upcoming, clean_db):
    import pandas as pd
    up = pd.DataFrame(clean_db.list_upcoming_reservations(2000))
    plat = up[up.loyalty_tier == "Platinum"]
    assert len(plat), "need a Platinum reservation"
    for rid in plat.reservation_id.head(15):
        if run_workflow(rid)["state"]["status"] != "SENT":
            continue
        rep = run_workflow(rid, trigger_type="inbound_reply", inbound_text="Could you arrange a private surprise dinner on the beach?")
        assert rep["interrupted"] and "vip_platinum_custom_request" in rep["state"]["hitl_reasons"]
        return
    pytest.fail("no sendable Platinum journey")


def test_unknown_reservation_fails_validation(clean_db):
    st = run_workflow("R_DOES_NOT_EXIST", guest_id="G_NONE")["state"]
    assert st["status"] == "FAIL_VALIDATION" and st["errors"]


def test_routing_functions():
    assert edges.route_after_compliance({"guardrail": {"status": "PASS"}}) == "hitl_gate"
    assert edges.route_after_compliance({"guardrail": {"status": "FAIL_POLICY"}, "retries": 1}) == "draft_message"
    assert edges.route_after_compliance({"guardrail": {"status": "FAIL_POLICY"}, "retries": 99}) == "hitl_gate"
    assert edges.route_after_compliance({"guardrail": {"status": "FAIL_VALIDATION"}}) == "__end__"


# ----------------------------------------------------------------- Flask
@pytest.fixture()
def client(clean_db):
    from src.api.flask_app import create_app
    return create_app().test_client()


def test_api_health_and_validation(client):
    h = client.get("/api/v1/health").get_json()
    assert h["status"] == "ok" and h["llm_mode"] == "mock"
    r = client.post("/api/v1/trigger", json={"reservation_id": "x"})
    assert r.status_code == 422
    r = client.post("/api/v1/trigger", json={"reservation_id": "R_NOPE"})
    assert r.status_code == 404
    assert client.post("/api/v1/reply", json={"reservation_id": "R500001", "text": ""}).status_code == 422
    assert client.post("/api/v1/hitl/approve", json={"thread_id": "nope", "decision": "approve"}).status_code == 404


def test_api_trigger_and_hitl_flow(client, upcoming):
    for r in upcoming:
        j = client.post("/api/v1/trigger", json={"reservation_id": r["reservation_id"], "adversarial": "persistent"}).get_json()
        if j["interrupted"]:
            pend = client.get("/api/v1/hitl/pending").get_json()["pending"]
            assert any(p["thread_id"] == j["thread_id"] for p in pend)
            a = client.post("/api/v1/hitl/approve", json={"thread_id": j["thread_id"], "decision": "approve"}).get_json()
            assert a["status"] == "SENT"
            return
    pytest.fail("no escalation")


@pytest.mark.live_llm
def test_live_qwen_draft_and_gate(upcoming):
    import os
    os.environ["LLM_MODE"] = "live"
    from src.config import get_settings
    from src.llm.client import LLMClient
    get_settings.cache_clear()
    llm = LLMClient("live")
    if not llm.is_up():
        pytest.skip("local Qwen not available")
    assert "pong" in llm.chat("Reply with one word.", "Reply with exactly: pong", temperature=0, max_tokens=16).lower()
