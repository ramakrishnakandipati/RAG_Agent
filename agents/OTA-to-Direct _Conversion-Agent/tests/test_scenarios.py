"""The five curated end-to-end flows must each produce their documented outcome (LLM forced offline under pytest)."""
import pytest
from agent.graph import run_agent
from agent.scenarios import build_scenarios
from agent import tools as T

RUNS = {}


@pytest.fixture(scope="module")
def runs():
    if not RUNS:
        for s in build_scenarios():
            RUNS[s["id"]] = run_agent(s["webhook"])
    return RUNS


def test_five_distinct_flows():
    sc = build_scenarios()
    assert len(sc) == 5 and len({s["id"] for s in sc}) == 5 and len({s["webhook"]["booking_id"] for s in sc}) == 5


def test_every_flow_passes_all_twelve_gates_and_has_no_error(runs):
    for k, r in runs.items():
        assert not r["error"], k
        assert len(r["gates"]) >= 12 and all(g["passed"] for g in r["gates"]), k
        assert [t["node"] for t in r["trace"]][0] == "ingest_and_resolve" and r["trace"][-1]["node"] == "log_and_attribute"


def test_happy_path_sent(runs):
    r = runs["happy_path"]
    assert r["dispatch"]["status"] == "sent" and r["final_offer_id"] != "NO_OFFER" and r["guardrail_retries"] == 0 and T.verify_link(r["dispatch"]["link"])
    assert not r["resolved_profile"].get("shadow") and r["attribution"]["treatment"] is True


def test_parity_breach_blocks_and_sends_nothing(runs):
    r = runs["parity_breach"]
    assert r["guardrail_retries"] == 2 and r["final_offer_id"] == "NO_OFFER" and r["dispatch"]["status"] == "no_action" and r["attribution"]["treatment"] is False
    assert r["webhook"]["direct_public_rate"] < r["webhook"]["ota_public_rate"]


def test_ghost_holdout_is_approved_but_suppressed(runs):
    r = runs["ghost_holdout"]
    assert r["is_holdout"] and r["guardrail"]["approved"] and r["final_offer_id"] != "NO_OFFER"
    assert r["dispatch"]["status"] == "suppressed_ghost_holdout" and r["attribution"]["treatment"] is False and r["attribution"]["ghost_holdout"] is True


def test_new_guest_uses_shadow_profile_and_wifi_window(runs):
    r = runs["new_guest_wifi"]
    assert r["resolved_profile"]["profile_id"].startswith("SHADOW-") and r["webhook"]["window"] == "on_property"
    assert r["dispatch"]["status"] == "sent" and r["final_offer_id"].startswith("WIFI_")


def test_post_stay_winback_sent(runs):
    r = runs["post_stay_winback"]
    assert r["webhook"]["window"] == "post_stay" and r["dispatch"]["status"] == "sent" and r["final_offer_id"].startswith("WINBACK_")
