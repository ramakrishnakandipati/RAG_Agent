"""Unit tests: data gates D1-D3 and the pre/post validators of all six LangGraph nodes."""
import json
import numpy as np, pandas as pd
import pytest
from agent import validators as V
from data.schema_validator import gate_d1
from config.settings import get_settings
from models import identity_resolution as ir


def _failed(r): return {c["check"] for c in r["checks"] if not c["passed"]}


# ---------------------------------------------------------------- Gate D1
def test_d1_passes_on_generated(bookings):
    assert gate_d1(bookings)["passed"]


@pytest.mark.parametrize("col,val,check", [("ota_commission", 0.5, "commission_range_0.10_0.30"), ("adr", 0.0, "adr_positive"), ("los", 0, "los_ge_1")])
def test_d1_catches_bad_values(bookings, col, val, check):
    b = bookings.copy(); b.loc[b.index[0], col] = val
    assert check in _failed(gate_d1(b))


def test_d1_catches_duplicate_pk_and_bad_holdout(bookings):
    b = bookings.copy(); b.loc[b.index[1], "booking_id"] = b.booking_id.iloc[0]
    assert "pk_non_null_unique" in _failed(gate_d1(b))
    b = bookings.copy(); b["is_ghost_holdout"] = np.arange(len(b)) % 3 == 0     # ~33%
    assert "ghost_holdout_15pct_pm1" in _failed(gate_d1(b))


def test_d1_missing_column(bookings):
    assert "required_columns" in _failed(gate_d1(bookings.drop(columns=["adr"])))


# ---------------------------------------------------------------- Gates D2 / D3
def test_d2_metric_definitions():
    df = pd.DataFrame({"true_profile_id": ["a", "b", None, "c"], "resolved_profile_id": ["a", "x", "y", None], "resolution_method": ["m"] * 4})
    m = ir.evaluate(df)
    assert m["recall"] == pytest.approx(1 / 3, abs=1e-3) and m["false_positive_merge_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert not ir.gate_d2(df)["passed"]


def test_d2_and_d3_recorded_and_passing():
    d = json.loads((get_settings().out / "validation" / "data_validation.json").read_text())
    assert d["D2"]["metrics"]["recall"] > 0.90 and d["D2"]["metrics"]["false_positive_merge_rate"] < 0.02
    assert d["D3"]["qini"] > 0 and d["D3"]["cancel_auc"] >= 0.72


def test_models_promoted_only_when_gate_passes():
    pr = json.loads((get_settings().out / "models" / "promotion.json").read_text())
    assert pr["promoted"] == pr["gate"]["passed"]


# ---------------------------------------------------------------- node validators
GOOD_FEATS = {k: 1.0 for k in __import__("models.uplift_pipeline", fromlist=["FEATURES"]).FEATURES}


def test_node1_validators(bookings):
    from tests.conftest import make_webhook
    wh = make_webhook(bookings.iloc[0], window="pre_arrival")
    assert V.pre_ingest({"webhook": wh})["passed"]
    assert not V.pre_ingest({"webhook": {**wh, "ota_commission": 0.9}})["passed"]
    assert not V.pre_ingest({"webhook": {**wh, "ota": "Hotels.com"}})["passed"]
    ok = {"resolution": {"method": "x"}, "resolved_profile": {"profile_id": "P1"}, "saved_commission_usd": 10.0, "is_holdout": False}
    assert V.post_ingest(ok)["passed"]
    assert not V.post_ingest({**ok, "saved_commission_usd": 0})["passed"]
    assert not V.post_ingest({**ok, "resolved_profile": None})["passed"]


def test_node2_validators():
    assert V.pre_score({"features": GOOD_FEATS})["passed"]
    assert not V.pre_score({"features": {"adr": 1.0}})["passed"]
    s = {"features": GOOD_FEATS, "scores": {"tau": 0.1, "p_cancel": 0.2, "latency_ms": 5.0}}
    assert V.post_score(s)["passed"]
    assert not V.post_score({**s, "scores": {"tau": 1.5, "p_cancel": 0.2, "latency_ms": 5}})["passed"]
    assert not V.post_score({**s, "scores": {"tau": 0.1, "p_cancel": 1.2, "latency_ms": 5}})["passed"]
    assert not V.post_score({**s, "scores": {"tau": 0.1, "p_cancel": 0.2, "latency_ms": 5000}})["passed"]
    assert not V.post_score({**s, "features": {**GOOD_FEATS, "adr": float("nan")}})["passed"]


def test_node3_validators():
    base = {"webhook": {"window": "pre_arrival"}, "llm_attempts": 1,
            "strategy": {"window": "pre_arrival", "offer_id": "CUG_BREAKFAST", "channel": "email", "rationale": "Good fit for leisure guest", "personalized_copy": "Hi Sam, enjoy breakfast on us."}}
    assert V.post_strategy(base)["passed"]
    assert not V.post_strategy({**base, "strategy": {**base["strategy"], "offer_id": "FREE_CAR"}})["passed"]          # not in catalog
    assert not V.post_strategy({**base, "strategy": {**base["strategy"], "personalized_copy": ""}})["passed"]          # empty copy
    assert not V.post_strategy({**base, "strategy": {**base["strategy"], "window": "post_stay"}})["passed"]            # wrong window
    assert not V.post_strategy({**base, "strategy": {}})["passed"]


def test_node4_5_6_validators():
    g = {"guardrail": {"approved": True}, "final_offer_id": "CUG_BREAKFAST", "guardrail_retries": 1}
    assert V.post_guardrails(g)["passed"]
    assert not V.post_guardrails({**g, "guardrail_retries": 3})["passed"]
    assert not V.post_guardrails({"guardrail": {"approved": False, "retry": False}, "final_offer_id": "X"})["passed"]
    d = {"is_holdout": True, "dispatch": {"status": "sent", "link_valid": True}}
    assert not V.post_dispatch(d)["passed"]                                                          # holdout must never be sent
    assert V.post_dispatch({"is_holdout": True, "dispatch": {"status": "suppressed_ghost_holdout", "link_valid": True}})["passed"]
    assert V.post_dispatch({"is_holdout": True, "dispatch": {"status": "no_action", "link_valid": True}})["passed"]
    assert not V.post_dispatch({"is_holdout": False, "dispatch": {"status": "sent", "link_valid": False}})["passed"]
    assert V.post_log({"attribution": {"treatment": False}})["passed"] and not V.post_log({})["passed"]
    assert not V.pre_dispatch({"final_offer_id": "X", "guardrail": {"approved": False}})["passed"]
