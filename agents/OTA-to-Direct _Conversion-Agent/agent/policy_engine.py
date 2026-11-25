"""Deterministic hard guardrails. The LLM proposes; this module disposes (never LLM-decided).

Rules:
  R1 OTA rate parity   - hotel public direct rate must be >= OTA rate (tolerance) and the offer must not be a public rate cut.
  R2 Margin floor      - direct flow-through margin after incentive cost >= min_margin_pct.
  R3 Margin ceiling    - incentive cost <= max_incentive_to_commission_ratio * saved commission.
  R4 Positive ENRD     - expected net revenue delta must be > 0 for any non-NO_OFFER action.
  R5 Fatigue / consent - msgs_30d < cap and email/SMS opt-in (guests with no CDP match have unknown consent and are only eligible in the on_property window, where the Wi-Fi portal captures opt-in).
  R6 Window fit        - offer must belong to the active window (or 'any').
"""
from __future__ import annotations
from dataclasses import dataclass, field
from config.settings import load_catalog

NO_OFFER = "NO_OFFER"


def catalog_offers() -> dict[str, dict]:
    return {o["id"]: o for o in load_catalog()["offers"]}


def expected_net_revenue_delta(tau: float, p_cancel: float, saved_commission: float, cost: float) -> float:
    """ENRD = (1 - Pc) * [ tau * SC  -  (p0 + tau) * cost ]   (cost paid only by guests who redeem/convert)."""
    p0 = load_catalog()["constraints"]["base_conversion_rate"]
    return (1 - p_cancel) * (tau * saved_commission - max(p0 + tau, 0.0) * cost)


@dataclass
class Verdict:
    approved: bool
    offer_id: str
    rule_results: list[dict] = field(default_factory=list)
    enrd: float = 0.0
    margin_pct: float = 0.0
    blocking_rules: list[str] = field(default_factory=list)

    def asdict(self) -> dict:
        return {"approved": self.approved, "offer_id": self.offer_id, "rule_results": self.rule_results, "enrd": round(self.enrd, 4),
                "margin_pct": round(self.margin_pct, 4), "blocking_rules": self.blocking_rules}


def evaluate(offer_id: str, window: str, *, tau: float, p_cancel: float, saved_commission: float, stay_revenue: float,
             ota_rate: float | None, direct_rate: float | None, msgs_30d: int, opt_in: bool, resolved: bool) -> Verdict:
    cat = load_catalog(); c = cat["constraints"]; offers = catalog_offers(); o = offers.get(offer_id)
    rr = []

    def rule(name, ok, detail):
        rr.append({"rule": name, "passed": bool(ok), "detail": detail})

    if o is None:
        rule("catalog_membership", False, f"unknown offer {offer_id}")
        return Verdict(False, offer_id, rr, blocking_rules=["catalog_membership"])
    cost = float(o["cost_usd"])
    if offer_id == NO_OFFER:
        return Verdict(True, offer_id, [{"rule": "no_action", "passed": True, "detail": "NO_OFFER always permitted"}], 0.0, c["flow_through_margin"])

    tol = c["parity_tolerance_pct"]
    parity_ok = o["kind"] != "public_rate" and (ota_rate is None or direct_rate is None or direct_rate >= ota_rate * (1 - tol))
    rule("R1_ota_rate_parity", parity_ok, {"offer_kind": o["kind"], "ota_rate": ota_rate, "direct_rate": direct_rate})
    margin = (c["flow_through_margin"] * stay_revenue - cost) / max(stay_revenue, 1e-9)
    rule("R2_margin_floor", margin >= c["min_margin_pct"], {"margin_pct": round(margin, 4), "floor": c["min_margin_pct"]})
    ratio = cost / max(saved_commission, 1e-9)
    rule("R3_incentive_to_commission_ceiling", ratio <= c["max_incentive_to_commission_ratio"], {"ratio": round(ratio, 4), "ceiling": c["max_incentive_to_commission_ratio"]})
    enrd = expected_net_revenue_delta(tau, p_cancel, saved_commission, cost)
    rule("R4_positive_expected_net_revenue_delta", enrd > 0, {"enrd_usd": round(enrd, 4)})
    consent_ok = opt_in if resolved else window == "on_property"      # unknown consent: only captive-portal (on-property) opt-in flow
    fatigue_ok = msgs_30d < c["fatigue_max_msgs_30d"] and consent_ok
    rule("R5_fatigue_and_consent", fatigue_ok, {"msgs_30d": msgs_30d, "cap": c["fatigue_max_msgs_30d"], "opt_in": opt_in, "resolved": resolved})
    rule("R6_window_fit", o["window"] in (window, "any"), {"offer_window": o["window"], "active_window": window})
    blocking = [r["rule"] for r in rr if not r["passed"]]
    return Verdict(not blocking, offer_id, rr, enrd, margin, blocking)


def best_compliant_offer(window: str, exclude: list[str], **kw) -> tuple[str, Verdict]:
    """Deterministic downgrade ladder: highest-ENRD compliant offer, else NO_OFFER."""
    best, bv = NO_OFFER, evaluate(NO_OFFER, window, **kw)
    for oid, o in catalog_offers().items():
        if oid in exclude or oid == NO_OFFER or o["window"] not in (window, "any"):
            continue
        v = evaluate(oid, window, **kw)
        if v.approved and (best == NO_OFFER or v.enrd > bv.enrd):
            best, bv = oid, v
    return best, bv
