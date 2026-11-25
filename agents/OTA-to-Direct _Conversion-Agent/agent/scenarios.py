"""Five curated end-to-end sample flows built from real rows of the generated dataset (deterministic selection).

Each scenario exercises a different path through the 6-node graph:
  1 happy_path_pre_arrival   - resolved, consented guest -> perk approved -> message SENT
  2 parity_breach            - hotel direct rate undercuts OTA -> R1 blocks -> retries -> NO_OFFER
  3 ghost_holdout            - offer approved but guest is in the 15% control arm -> NEVER sent
  4 new_guest_wifi_portal    - no CDP match (shadow profile), on-property window -> Wi-Fi opt-in offer SENT
  5 post_stay_winback        - resolved guest after checkout -> loyalty-points win-back SENT
"""
from __future__ import annotations
import pandas as pd
from data.synthetic_generator import GEN_DIR, load

_cache: list[dict] | None = None


def _webhook(r, window: str, **over) -> dict:
    wh = {"booking_id": r.booking_id, "ota": r.ota, "first_name": r.first_name, "last_name": r.last_name, "masked_email": r.masked_email,
          "phone_last4": r.phone_last4 if isinstance(r.phone_last4, str) else None, "country": r.country, "checkin_date": r.checkin_date,
          "los": int(r.los), "adr": float(r.adr), "ota_commission": float(r.ota_commission), "lead_days": int(r.lead_days),
          "trip_purpose": r.trip_purpose, "free_cancellation": bool(r.free_cancellation), "ota_public_rate": float(r.adr),
          "direct_public_rate": float(r.adr), "window": window, "is_ghost_holdout": bool(r.is_ghost_holdout)}
    wh.update(over)
    return wh


def build_scenarios() -> list[dict]:
    global _cache
    if _cache is not None:
        return _cache
    _, cdp = load()
    rb = pd.read_csv(GEN_DIR / "resolved_bookings.csv", dtype={"phone_last4": str})
    sf = pd.read_csv(GEN_DIR / "scored_features.csv")
    d = pd.concat([rb, sf[["tau_pred", "p_cancel"]]], axis=1)
    d = d.merge(cdp[["profile_id", "email_opt_in", "msgs_30d", "loyalty_tier"]], left_on="resolved_profile_id", right_on="profile_id", how="left")
    good = d[d.resolved_profile_id.notna() & (d.email_opt_in == True) & (d.msgs_30d < 3) & (d.tau_pred > 0.20) & (d.p_cancel < 0.25)]  # noqa: E712
    pre = good[(good.adr > 140) & (good.saved_commission_usd > 70) & ~good.is_ghost_holdout]
    post = good[(good.saved_commission_usd > 90) & ~good.is_ghost_holdout & (good.treated == 1)]
    new = d[d.resolved_profile_id.isna() & d.true_profile_id.isna() & (d.tau_pred > 0.15) & (d.p_cancel < 0.30) & (d.saved_commission_usd > 40) & ~d.is_ghost_holdout]
    ho = good[good.is_ghost_holdout & (good.adr > 140)]
    S = [
        dict(id="happy_path", title="1. Happy path - pre-arrival perk", window="pre_arrival", row=pre.iloc[0], over={"is_ghost_holdout": False},
             expect="Guest resolved from CDP, positive uplift, compliant perk approved, message SENT with signed link.",
             description="A consented loyalty-member guest booked via an OTA 3+ weeks ahead."),
        dict(id="parity_breach", title="2. OTA rate-parity breach", window="pre_arrival", row=pre.iloc[1], over={"is_ghost_holdout": False, "direct_factor": 0.9},
             expect="Hotel direct rate is 10% below the OTA rate -> R1 blocks every offer -> 2 retries -> NO_OFFER, nothing sent.",
             description="Same kind of guest, but the hotel's own website undercuts the OTA, which would violate parity."),
        dict(id="ghost_holdout", title="3. Ghost holdout (control arm)", window="pre_arrival", row=ho.iloc[0], over={"is_ghost_holdout": True},
             expect="Offer is approved by policy but the guest is in the 15% holdout -> status suppressed_ghost_holdout, never messaged.",
             description="Eligible guest randomly assigned to the control group so lift can be measured causally."),
        dict(id="new_guest_wifi", title="4. New guest - on-property Wi-Fi portal", window="on_property", row=new.iloc[0], over={"is_ghost_holdout": False},
             expect="No CDP match -> SHADOW profile (unknown consent). Only the on_property window is allowed -> Wi-Fi opt-in offer SENT.",
             description="Unknown OTA guest connects to hotel Wi-Fi; the portal captures opt-in."),
        dict(id="post_stay_winback", title="5. Post-stay win-back", window="post_stay", row=post[~post.resolved_profile_id.isin(pre.iloc[:2].resolved_profile_id)].iloc[0], over={"is_ghost_holdout": False},
             expect="Resolved guest after checkout -> loyalty-points win-back approved and SENT.",
             description="Guest has checked out; agent invites them to book direct next time."),
    ]
    out = []
    for s in S:
        over = dict(s["over"]); f = over.pop("direct_factor", None)
        wh = _webhook(s["row"], s["window"], **over)
        if f: wh["direct_public_rate"] = round(wh["ota_public_rate"] * f, 2)
        out.append({k: s[k] for k in ("id", "title", "description", "expect", "window")} | {"webhook": wh})
    _cache = out
    return out
