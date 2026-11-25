"""Policy engine (parity, margin floor, fatigue) and end-to-end graph behaviour: holdout, retries, downgrade, fallback, LLM retry."""
import json
import pytest
from agent import policy_engine as P, tools as T
from agent.graph import run_agent
from agent.state import OfferStrategySchema
from config.settings import get_settings
from tests.conftest import make_webhook

KW = dict(tau=0.25, p_cancel=0.1, saved_commission=60.0, stay_revenue=500.0, ota_rate=200.0, direct_rate=200.0, msgs_30d=0, opt_in=True, resolved=True)


def rules(v): return {r["rule"] for r in v.rule_results if not r["passed"]}


# ---------------------------------------------------------------- policy engine
def test_compliant_offer_approved():
    v = P.evaluate("CUG_LATE_CHECKOUT", "pre_arrival", **KW)
    assert v.approved and v.enrd > 0


def test_parity_blocks_public_rate_cut_and_undercut_direct_rate():
    assert "R1_ota_rate_parity" in rules(P.evaluate("PUBLIC_RATE_DISCOUNT_10", "pre_arrival", **KW))
    assert "R1_ota_rate_parity" in rules(P.evaluate("CUG_LATE_CHECKOUT", "pre_arrival", **{**KW, "direct_rate": 180.0}))   # direct below OTA


def test_margin_floor_and_ceiling():
    cheap_room = {**KW, "stay_revenue": 60.0}                       # 0.6*60-35 = 1 -> margin 1.7% < 35% floor
    assert "R2_margin_floor" in rules(P.evaluate("CUG_ROOM_UPGRADE", "pre_arrival", **cheap_room))
    low_commission = {**KW, "saved_commission": 20.0}               # cost 35 > 60% of 20
    assert "R3_incentive_to_commission_ceiling" in rules(P.evaluate("CUG_ROOM_UPGRADE", "pre_arrival", **low_commission))


def test_negative_uplift_and_high_cancel_risk_blocked():
    assert "R4_positive_expected_net_revenue_delta" in rules(P.evaluate("CUG_LATE_CHECKOUT", "pre_arrival", **{**KW, "tau": -0.05}))
    assert "R4_positive_expected_net_revenue_delta" in rules(P.evaluate("CUG_LATE_CHECKOUT", "pre_arrival", **{**KW, "p_cancel": 1.0}))


def test_fatigue_consent_and_window():
    assert "R5_fatigue_and_consent" in rules(P.evaluate("CUG_LATE_CHECKOUT", "pre_arrival", **{**KW, "msgs_30d": 3}))
    assert "R5_fatigue_and_consent" in rules(P.evaluate("CUG_LATE_CHECKOUT", "pre_arrival", **{**KW, "opt_in": False}))
    assert "R6_window_fit" in rules(P.evaluate("WIFI_PREMIUM", "pre_arrival", **KW))
    assert not P.evaluate("NOT_AN_OFFER", "pre_arrival", **KW).approved


def test_no_offer_always_allowed_and_ladder_never_returns_violation():
    assert P.evaluate("NO_OFFER", "pre_arrival", **{**KW, "msgs_30d": 99}).approved
    oid, v = P.best_compliant_offer("pre_arrival", [], **{**KW, "direct_rate": 150.0})
    assert oid == "NO_OFFER" and v.approved                         # parity breach -> nothing compliant
    oid, v = P.best_compliant_offer("pre_arrival", [], **KW)
    assert oid != "NO_OFFER" and v.approved


def test_enrd_formula():
    # (1-0.1)*(0.25*60 - (0.12+0.25)*12) = 0.9*(15-4.44)
    assert P.expected_net_revenue_delta(0.25, 0.1, 60.0, 12.0) == pytest.approx(0.9 * (15 - 4.44))


# ---------------------------------------------------------------- tools
def test_holdout_rate_and_gated_link():
    n = sum(T.assign_holdout(f"B{i}") for i in range(20000)) / 20000
    assert 0.14 < n < 0.16
    url = T.gated_link("B1", "CUG_BREAKFAST")
    assert T.verify_link(url) and not T.verify_link(url.replace("CUG_BREAKFAST", "CUG_ROOM_UPGRADE"))


# ---------------------------------------------------------------- graph
def _first(bookings, mask, k=0):
    return bookings[mask].iloc[k]


def test_holdout_guest_is_never_messaged(bookings):
    for i in range(12):
        s = run_agent(make_webhook(bookings[bookings.is_ghost_holdout].iloc[i]))
        assert s["dispatch"]["status"] != "sent"
        assert s["attribution"]["treatment"] is False and s["attribution"]["ghost_holdout"] is True


def test_treated_guest_can_be_dispatched_with_valid_gated_link(bookings):
    sent = None
    for i in range(60):
        s = run_agent(make_webhook(bookings[~bookings.is_ghost_holdout].iloc[i]))
        if s["dispatch"].get("status") == "sent": sent = s; break
    assert sent is not None, "no treated guest received an offer in 60 tries"
    assert T.verify_link(sent["dispatch"]["link"]) and sent["attribution"]["treatment"] is True and sent["llm_mode"] == "offline_fallback"
    assert all(g["passed"] for g in sent["gates"]) and len(sent["gates"]) == 12          # 6 nodes x (pre, post)


def test_parity_breach_offline_policy_retries_then_settles_on_no_offer(bookings):
    row = _first(bookings, ~bookings.is_ghost_holdout)
    s = run_agent(make_webhook(row, direct_public_rate=float(row.adr) * 0.9))
    assert s["guardrail_retries"] == 2 and s["final_offer_id"] == "NO_OFFER" and not s["error"]
    assert len(s["rejected_offers"]) == 2 and s["dispatch"]["status"] == "no_action"
    assert sum(1 for t in s["trace"] if t["node"] == "select_strategy") == 3      # initial + 2 retries, then stop


def test_stubborn_llm_hits_retry_cap_then_deterministic_downgrade_is_audited(monkeypatch, bookings):
    _patch(monkeypatch, [GOOD])                                      # LLM keeps proposing CUG_BREAKFAST
    row = _first(bookings, ~bookings.is_ghost_holdout)
    s = run_agent(make_webhook(row, direct_public_rate=float(row.adr) * 0.9))
    assert s["guardrail_retries"] == 2 and s["downgraded"] and s["final_offer_id"] == "NO_OFFER" and not s["error"]
    assert "R1_ota_rate_parity" in s["guardrail"]["initial_blocking"] and s["guardrail"]["downgrade_reason"]
    assert sum(1 for t in s["trace"] if t["node"] == "select_strategy") == 3
    audits = [json.loads(x) for x in (get_settings().out / "logs" / "guardrail_audits.jsonl").read_text().splitlines()[-6:]]
    assert any(a.get("downgraded_to") == "NO_OFFER" for a in audits)


def test_invalid_webhook_short_circuits_without_dispatch():
    s = run_agent({"booking_id": "BAD1", "ota": "Expedia"})
    assert s["error"] and "ingest_and_resolve" in s["error"] and not s.get("dispatch") and s["attribution"]["treatment"] is False


def test_out_of_range_commission_rejected(bookings):
    s = run_agent(make_webhook(bookings.iloc[0], ota_commission=0.45))
    assert s["error"] and not s.get("dispatch")


def test_unresolved_guest_gets_shadow_profile_and_no_preararrival_offer(bookings):
    row = _first(bookings, bookings.true_profile_id.isna() & ~bookings.is_ghost_holdout)
    s = run_agent(make_webhook(row, masked_email="zzz@m.expedia.com", first_name="Qxzv", last_name="Wkjh", phone_last4=None))
    assert s["resolved_profile"]["profile_id"].startswith("SHADOW-") and s["features"]["is_resolved"] == 0
    assert s["final_offer_id"] == "NO_OFFER"                         # consent unknown (R5) -> no pre-arrival incentive


def test_all_three_windows_run(bookings):
    for w in ("pre_arrival", "on_property", "post_stay"):
        s = run_agent(make_webhook(_first(bookings, ~bookings.is_ghost_holdout), window=w))
        assert not s["error"] and s["strategy"]["window"] == w


# ---------------------------------------------------------------- LLM retry + fallback behaviour
class _Msg:  # minimal fake of the OpenAI response
    def __init__(self, txt): self.choices = [type("C", (), {"message": type("M", (), {"content": txt})()})()]


def _fake_client(replies):
    calls = {"n": 0}
    class Comp:
        def create(self, **kw):
            r = replies[min(calls["n"], len(replies) - 1)]; calls["n"] += 1
            if isinstance(r, Exception): raise r
            return _Msg(r)
    return type("Cl", (), {"chat": type("Ch", (), {"completions": Comp()})()})(), calls


GOOD = json.dumps({"window": "pre_arrival", "offer_id": "CUG_BREAKFAST", "channel": "email", "rationale": "Leisure guest values breakfast.", "personalized_copy": "Hi there, enjoy a complimentary breakfast on us."})


def _patch(monkeypatch, replies):
    from agent import llm_client as L
    cl, calls = _fake_client(replies)
    monkeypatch.setattr(L, "endpoint", lambda refresh=False: {"available": True, "base_url": "x", "model": "m"}); monkeypatch.setattr(L, "_client", lambda ep: cl)
    return L, calls


def _fb(): return OfferStrategySchema(window="pre_arrival", offer_id="NO_OFFER", channel="email", rationale="fallback used here", personalized_copy="fallback copy text here")


def test_malformed_llm_output_retried_up_to_2_times(monkeypatch):
    L, calls = _patch(monkeypatch, ["not json", '{"offer_id": "x"}', GOOD])
    obj, meta = L.structured("s", "u", OfferStrategySchema, _fb, max_retries=2)
    assert meta["mode"] == "live" and meta["attempts"] == 3 and calls["n"] == 3 and obj.offer_id == "CUG_BREAKFAST"


def test_persistently_malformed_llm_falls_back(monkeypatch):
    L, calls = _patch(monkeypatch, ["garbage"])
    obj, meta = L.structured("s", "u", OfferStrategySchema, _fb, max_retries=2)
    assert meta["mode"] == "offline_fallback" and calls["n"] == 3 and obj.rationale == "fallback used here"


def test_transport_error_falls_back_immediately(monkeypatch):
    L, calls = _patch(monkeypatch, [ConnectionError("down")])
    _, meta = L.structured("s", "u", OfferStrategySchema, _fb)
    assert meta["mode"] == "offline_fallback" and calls["n"] == 1


def test_llm_proposing_parity_breaking_offer_is_blocked_by_policy(monkeypatch, bookings):
    bad = json.dumps({"window": "pre_arrival", "offer_id": "PUBLIC_RATE_DISCOUNT_10", "channel": "email", "rationale": "Cut the rate to win them.", "personalized_copy": "Hi, 10% off our public rate!"})
    _patch(monkeypatch, [bad])
    s = run_agent(make_webhook(_first(bookings, ~bookings.is_ghost_holdout)))
    assert s["final_offer_id"] != "PUBLIC_RATE_DISCOUNT_10" and "PUBLIC_RATE_DISCOUNT_10" in s["rejected_offers"]
