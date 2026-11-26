"""Agent 2 - Yield & Brand Compliance Auditor (critic). Deterministic contracts first, LLM judgement second.

Contracts (docs L7); each failure yields a coded Violation + machine-actionable remediation:
  1 INVENTORY_BUFFER           live_available(target) >= B = max(B_min, ceil(z*sigma_D*sqrt(L)))          (PMS)
  2 BELOW_RMS_FLOOR / MARGIN   p >= floor_k   and   (p - c_COGS - c_HK)/p >= theta_m                      (RMS)
  3 LOYALTY_PERK_DUPLICATION   bundled_amenities  INTERSECT  tier_perks = {}   (and SKU not already a free tier perk)
  4 TONE                       regex ban-list AND tone_score >= theta_tone (Qwen3-VL audit, heuristic fallback) AND copy length AND fatigue
  5 VISUAL (multimodal hook)   Qwen3-VL checks asset_url vs. offered room/SKU when an asset is attached
Decision: no violations -> APPROVED (or ESCALATE_HITL if tier_weight * price >= V_VIP);  violations -> REJECTED,
          or ESCALATE_HITL when revision >= R_max (unresolved) or contact fatigue blocks the send.
"""
from __future__ import annotations
import math
import re
import time

from .llm_client import get_llm
from .schemas import (ComplianceAuditVerdict, Directives, GraphState, GuestProfile, ToneAudit, UpsellOfferDraft, ViolationCode as V, Violation, VisualAudit)
from .tools import CATEGORY_PERK, PMS, contribution_margin, rms_floor, rms_margin_ratio, trace
from models.utils import load_config, load_policies

CFG, POL = load_config(), load_policies()
BANNED = [re.compile(p) for p in POL["tone"]["banned_regex"]]


def heuristic_tone(text: str) -> ToneAudit:
    hits = [p.pattern for p in BANNED if p.search(text)]
    letters = [c for c in text if c.isalpha()]; caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
    score = 1.0 - 0.3 * len(hits) - (0.2 if text.count("!") else 0) - (0.2 if caps > 0.18 else 0)
    issues = [f"banned pattern {h}" for h in hits] + (["exclamation marks"] if "!" in text else []) + (["excessive capitals"] if caps > 0.18 else [])
    s = max(0.0, min(1.0, score))
    return ToneAudit(score=s, issues=issues, verdict="LUXURY_OK" if s >= CFG["agents"]["tone_threshold"] else "OFF_BRAND")


def _merge(a: Directives, b: Directives) -> Directives:
    mp = {k: max(a.min_price.get(k, 0.0), b.min_price.get(k, 0.0)) for k in set(a.min_price) | set(b.min_price)}
    return Directives(min_price=mp, remove_perks=sorted(set(a.remove_perks) | set(b.remove_perks)),
                      exclude_skus=sorted(set(a.exclude_skus) | set(b.exclude_skus)), refined_tone=a.refined_tone or b.refined_tone,
                      shorten_copy=a.shorten_copy or b.shorten_copy, drop_asset=a.drop_asset or b.drop_asset)


def audit(guest: GuestProfile, draft: UpsellOfferDraft, revision: int, max_revisions: int, pms_override: dict | None = None) -> ComplianceAuditVerdict:
    llm = get_llm(); viol: list[Violation] = []; d = Directives(); checks: dict[str, bool] = {}
    # 1 inventory
    inv = PMS.check(draft.sku, guest.room_type, pms_override)
    checks["inventory_buffer"] = inv["ok"]
    if not inv["ok"]:
        why = inv["reason"] or f"{inv['key']} available={inv['available']} < dynamic safety buffer {inv['buffer']}"
        viol.append(Violation(code=V.INELIGIBLE_SKU if inv["key"] is None else V.INVENTORY_BUFFER, detail=why,
                              remediation=f"Do not offer {draft.sku}; protect last-minute BAR demand. Pivot to a non-constrained SKU."))
        d.exclude_skus.append(draft.sku)
    # 2 RMS floor + margin
    floor = rms_floor(draft.sku); cm = contribution_margin(draft.sku, draft.offer_price); ratio = cm / draft.offer_price
    checks["rms_floor"] = draft.offer_price >= floor; checks["margin_floor"] = ratio >= rms_margin_ratio()
    if not checks["rms_floor"]:
        viol.append(Violation(code=V.BELOW_RMS_FLOOR, detail=f"offer ${draft.offer_price:.2f} < RMS floor ${floor:.2f}", remediation=f"Re-price at or above ${math.ceil(floor):d}."))
        d.min_price[draft.sku] = float(math.ceil(floor))
    if not checks["margin_floor"]:
        need = (draft.offer_price - cm) / (1 - rms_margin_ratio())      # price s.t. (p - cost)/p = theta_m
        viol.append(Violation(code=V.MARGIN_BELOW_THRESHOLD, detail=f"contribution margin {ratio:.1%} < {rms_margin_ratio():.0%}", remediation=f"Raise price to >= ${math.ceil(need):d}."))
        d.min_price[draft.sku] = max(d.min_price.get(draft.sku, 0.0), float(math.ceil(need)))
    # 3 loyalty parity
    dup = sorted(set(draft.bundled_amenities) & set(guest.existing_tier_perks)); checks["loyalty_parity"] = not dup
    if dup:
        viol.append(Violation(code=V.LOYALTY_PERK_DUPLICATION, detail=f"{guest.loyalty_tier} members already receive {dup} complimentary",
                              remediation=f"Remove {dup} from the bundle; offer a non-duplicated amenity or none."))
        d.remove_perks += dup
    free_sku = CATEGORY_PERK.get(draft.sku) in guest.existing_tier_perks; checks["sku_not_entitled"] = not free_sku
    if free_sku:
        viol.append(Violation(code=V.SKU_ENTITLED_FREE, detail=f"{draft.sku} is already a complimentary {guest.loyalty_tier} perk (cannibalisation)",
                              remediation="Select a different SKU; selling an entitled perk is zero-revenue cannibalisation."))
        d.exclude_skus.append(draft.sku)
    # 4 brand tone
    text = f"{draft.subject}\n{draft.body_copy}"; banned = [p.pattern for p in BANNED if p.search(text)]
    checks["tone_regex"] = not banned
    if banned:
        viol.append(Violation(code=V.TONE_BANNED_PHRASE, detail=f"discount-spam patterns: {banned}", remediation="Rewrite in refined luxury register; no discount triggers, urgency or exclamation marks."))
        d.refined_tone = True
    tone = llm.structured(ToneAudit, "You are a luxury-hospitality brand auditor. Score 0-1 how well the message matches a refined, understated five-star voice "
                          "(0 = spammy/discount-driven). List concrete issues.", text)
    tone = tone or heuristic_tone(text); tone_mode = "qwen" if llm.mode == "qwen" and tone is not None else "fallback"
    checks["tone_score"] = tone.score >= CFG["agents"]["tone_threshold"]
    if not checks["tone_score"] and not banned:
        viol.append(Violation(code=V.TONE_SCORE_LOW, detail=f"tone score {tone.score:.2f} < {CFG['agents']['tone_threshold']}: {tone.issues}", remediation="Warmer, understated phrasing; remove pressure."))
        d.refined_tone = True
    mx = POL["tone"]["max_copy_chars"][draft.channel]; checks["copy_length"] = len(draft.body_copy) <= mx
    if not checks["copy_length"]:
        viol.append(Violation(code=V.COPY_TOO_LONG, detail=f"{len(draft.body_copy)} chars > {mx} for {draft.channel}", remediation=f"Shorten body to <= {mx} characters."))
        d.shorten_copy = True
    # 5 multimodal visual alignment (optional)
    if draft.asset_url:
        va = llm.vision(VisualAudit, f"Does this image match an offer for {draft.sku} ({draft.target_room_type or ''}) in a luxury resort? Answer aligned true/false.", draft.asset_url)
        checks["visual_alignment"] = True if va is None else va.aligned
        if va is not None and not va.aligned:
            viol.append(Violation(code=V.VISUAL_MISALIGNMENT, detail=va.notes, remediation="Remove or replace the visual asset.")); d.drop_asset = True
    fatigue = guest.contacts_7d >= POL["fatigue"]["max_contacts_7d"]; checks["contact_fatigue"] = not fatigue
    vip_value = POL["hitl"]["clv_weight_tiers"][guest.loyalty_tier] * draft.offer_price
    hitl_reason, decision = None, "APPROVED"
    if fatigue:
        decision, hitl_reason = "ESCALATE_HITL", f"contact fatigue: {guest.contacts_7d} touches in 7d (max {POL['fatigue']['max_contacts_7d']}) - human to defer/approve the send"
        viol.append(Violation(code=V.CONTACT_FATIGUE, detail=hitl_reason, remediation="Defer outreach or obtain human approval."))
    elif viol:
        if revision >= max_revisions:
            decision, hitl_reason = "ESCALATE_HITL", f"unresolved after {max_revisions} revisions: {[v.code.value for v in viol]}"
        else:
            decision = "REJECTED"
    elif vip_value >= POL["hitl"]["vip_offer_value_usd"]:
        decision, hitl_reason = "ESCALATE_HITL", f"high-value VIP threshold: tier weight x price = ${vip_value:,.0f} >= ${POL['hitl']['vip_offer_value_usd']}"
    return ComplianceAuditVerdict(decision=decision, reservation_id=guest.reservation_id, revision=revision, violations=viol,
                                  remediation_instructions=[v.remediation for v in viol], directives=d, checks=checks, tone_score=round(tone.score, 3),
                                  hitl_reason=hitl_reason, vip_value=round(vip_value, 2), llm_mode=tone_mode)


def auditor_node(state: GraphState) -> dict:
    t0 = time.perf_counter(); g = GuestProfile(**state["guest"]); draft = UpsellOfferDraft(**state["draft"]); rev = int(state.get("revision", 0))
    mx = int(state.get("max_revisions", load_config()["agents"]["max_revisions"]))
    v = audit(g, draft, rev, mx, state.get("pms_override"))
    merged = _merge(Directives(**state.get("directives", {})), v.directives)
    ev = {"agent": "auditor", "revision": rev, "decision": v.decision, "violations": [x.code.value for x in v.violations], "tone_score": v.tone_score,
          "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "llm_mode": v.llm_mode}
    trace({"thread_id": state.get("thread_id"), **ev})
    out = {"verdict": v.model_dump(mode="json"), "directives": merged.model_dump(), "history": [ev]}
    if v.decision == "REJECTED":
        out["revision"] = rev + 1
    if v.decision == "APPROVED":
        out["final_status"] = "APPROVED"
    return out
