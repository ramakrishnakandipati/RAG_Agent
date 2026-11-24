import copy

import pandas as pd
import pytest

from src.tools import compliance_tools as comp
from src.tools import nba_rms_engine as nba
from src.validation import step_validators as sv
from src.validation.data_validator import psi, validate_all


# ----------------------------------------------------------------- device / llm
def test_device_fallback_params(monkeypatch):
    from src.utils import device
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    assert device.get_device() == "cpu"
    assert device.xgb_device_params() == {"tree_method": "hist", "device": "cpu"}
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    assert device.xgb_device_params()["device"] == "cuda"


def test_llm_json_extraction():
    from src.llm.client import extract_json
    assert extract_json('<think>x</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise {"b": 2} tail') == {"b": 2}


# ----------------------------------------------------------------- data validation
def test_data_validator_passes_on_generated_data():
    rep = validate_all()
    assert rep["all_passed"], [c for c in rep["checks"] if not c["passed"]]


def test_data_validator_catches_corruption():
    from src.config import get_settings
    s = get_settings()
    data = {k: pd.read_csv(s.raw_dir / f"{k}.csv", keep_default_na=False, na_values=[""]) for k in
            ["guests", "reservations", "weather", "occupancy", "inventory", "offer_history", "sku_catalog", "events"]}
    data["weather"].loc[0, "precip_prob"] = 1.7
    data["reservations"].loc[0, "guest_id"] = "G_MISSING"
    rep = validate_all(data)
    failed = {c["check"] for c in rep["checks"] if not c["passed"]}
    assert "range_precip_prob[0,1]" in failed and "ref_guest_exists" in failed


def test_psi_zero_for_identical():
    s = pd.Series(range(1000))
    assert psi(s, s) < 1e-3


# ----------------------------------------------------------------- step validators
def _ctx(upcoming):
    from src.tools import environment_tools as envt
    from src.tools import pms_crm_tools as pms
    r = upcoming[0]
    g, st = pms.get_guest_360(r["guest_id"]), pms.get_stay(r["reservation_id"])
    return {"guest_id": g["guest_id"], "reservation_id": st["reservation_id"], "guest": g, "stay": st,
            "env": envt.get_environmental_signals(str(st["check_in"]), str(st["check_out"]))}


def test_enrich_validators(upcoming):
    s = _ctx(upcoming)
    sv.pre_enrich_context(s); sv.post_enrich_context(s)
    bad = copy.deepcopy(s); bad["env"]["precip_prob"] = 1.4
    with pytest.raises(sv.ValidationFailure):
        sv.post_enrich_context(bad)
    bad = copy.deepcopy(s); bad["stay"]["party_size"] = 0
    with pytest.raises(sv.ValidationFailure):
        sv.post_enrich_context(bad)
    bad = copy.deepcopy(s); bad["stay"]["check_out"] = bad["stay"]["check_in"]
    with pytest.raises(sv.ValidationFailure):
        sv.post_enrich_context(bad)
    with pytest.raises(sv.ValidationFailure):
        sv.pre_enrich_context({"guest_id": "", "reservation_id": "R1"})


# ----------------------------------------------------------------- RMS rules + hallucination gate
def test_offers_respect_rms_rules(upcoming):
    from src.config import get_settings
    cfg = get_settings()
    n = 0
    for r in upcoming[:25]:
        s = _ctx([r])
        for o in nba.generate_offers(s["guest"], s["stay"], s["env"]):
            n += 1
            assert o["available_units"] > 0
            assert abs(o["final_price"] - round(o["list_price"] * (1 - o["discount_pct"] / 100), 2)) < 0.011
            if o["category"] != "loyalty":
                assert o["discount_pct"] <= cfg.max_discount_pct and o["margin_pct"] >= cfg.min_margin_pct
            assert 0 <= o["propensity"] <= 1
    assert n > 0


OFFER = {"sku": "SPA-RELAX", "list_price": 140.0, "final_price": 126.0, "discount_pct": 10.0}


@pytest.mark.parametrize("text,expected", [
    ("Enjoy it for $126.00, that is 10% off the usual $140.00.", []),
    ("Only $99 today!", ["price_hallucination:$99"]),
    ("Get 40% off", ["discount_hallucination:40%"]),
    ("Try SPA-FAKE9 now", ["sku_hallucination:SPA-FAKE9"]),
    ("Your SPA-RELAX is ready", []),
])
def test_hallucination_gate(text, expected):
    assert comp.check_text_tokens(text, OFFER) == expected


def test_no_discount_means_no_percent_allowed():
    o = dict(OFFER, discount_pct=0.0, final_price=140.0)
    assert comp.check_text_tokens("Save 10% today", o) == ["discount_hallucination:10%"]


# ----------------------------------------------------------------- inventory / folio
def test_booking_is_atomic_idempotent_and_sold_out(clean_db):
    pms = clean_db
    with pms.connect() as con:
        con.execute("UPDATE inventory SET booked=capacity-1 WHERE sku='SPA-RELAX' AND date=(SELECT MIN(date) FROM inventory)")
        d = con.execute("SELECT MIN(date) d FROM inventory").fetchone()["d"]
    a = pms.book_ancillary_item("R_T1", "SPA-RELAX", 140.0, d)
    assert a["ok"] and not a["already_booked"]
    assert pms.book_ancillary_item("R_T1", "SPA-RELAX", 140.0, d)["already_booked"]
    b = pms.book_ancillary_item("R_T2", "SPA-RELAX", 140.0, d)
    assert not b["ok"] and b["reason"] == "sold_out"
    assert pms.get_availability("SPA-RELAX", d) == 0


# ----------------------------------------------------------------- retriever
def test_hybrid_retriever_grounds_facts():
    from src.tools.rag_retriever import HybridRetriever
    r = HybridRetriever()
    hits = r.search("what time does late checkout end", k=3, doc_types=["policy"])
    assert hits and any("4 PM" in h["text"] for h in hits)
    hits = r.search("couples massage hydrotherapy", k=2, doc_types=["spa"])
    assert "Couples" in hits[0]["text"]
