"""Agent 1 - Revenue & Personalization Strategist (generator).

Pipeline per draft
  1. GPU quant tools rank candidate SKUs:  S_k = tau_k(x)(p_k-c_k) + UCB_k - h_k   (CATE, bandit bonus, shadow-price hurdle)
  2. Price p* = argmax_{p>=max(floor, directive.min_price)} (p-c) P_hat(x,k,p)  (monotone elasticity net)
  3. Deterministic amenity bundling (segment sweeteners), then Auditor directives (remove_perks / exclude_skus / min_price) applied
  4. Qwen3-VL writes the copy via a JSON-schema contract (CopyOut); numbers/SKUs are NEVER LLM-chosen. Fallback = refined templates.
`injections` are explicit stress-test hooks (documented in tools.SCENARIOS) used to exercise the critic loop.
"""
from __future__ import annotations
import time

from .llm_client import get_llm
from .schemas import CopyOut, Directives, GraphState, GuestProfile, UpsellOfferDraft
from .tools import (CATEGORY_PERK, UPGRADE_PATH, get_engine, trace)
from models.utils import load_policies

POL = load_policies()
SWEETENERS = {"Corporate": ["LATE_CHECKOUT_2PM"], "VIP_Loyalty": ["WELCOME_DRINK", "LATE_CHECKOUT_2PM"], "Honeymoon": ["CHAMPAGNE_BOTTLE", "TURNDOWN_PETALS"],
              "Family": ["BREAKFAST_FOR_TWO"], "Leisure": ["WELCOME_DRINK"]}
UNIT = {"ROOM_UPGRADE": "per_night", "SPA_PACKAGE": "per_person", "DINING_EXPERIENCE": "per_person", "LATE_CHECKOUT": "per_stay"}
LABEL = {"ROOM_UPGRADE": "a room upgrade", "SPA_PACKAGE": "a spa experience", "DINING_EXPERIENCE": "a private dining experience", "LATE_CHECKOUT": "a relaxed late checkout"}
AMENITY_TXT = {"WELCOME_DRINK": "a welcome drink", "LATE_CHECKOUT_2PM": "a 2 PM late checkout", "BREAKFAST_FOR_TWO": "breakfast for two", "SPA_CREDIT_50": "a spa credit",
               "TURNDOWN_PETALS": "a rose-petal turndown", "CHAMPAGNE_BOTTLE": "a chilled bottle of champagne"}


def _refined_copy(g: GuestProfile, sku: str, price: float, unit: str, amenities: list[str], room_to: str | None, channel: str):
    inc = ""
    if amenities:
        names = [AMENITY_TXT.get(a, a.lower().replace("_", " ")) for a in amenities]
        inc = " Your stay would also include " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + "."
    what = f"{LABEL[sku]} to our {room_to.replace('_', ' ').title()}" if (sku == "ROOM_UPGRADE" and room_to) else LABEL[sku]
    if channel == "SMS":
        return CopyOut(subject="Your stay", body=f"{g.first_name}, we would be delighted to arrange {what} for ${price:,.0f} {unit.replace('_', ' ')}. Reply to accept.", rationale="")
    return CopyOut(subject=f"A considered touch for your upcoming stay, {g.first_name}",
                   body=f"Dear {g.first_name},\n\nWe are looking forward to welcoming you in {g.lead_time} days. We would be delighted to arrange {what} "
                        f"for ${price:,.0f} {unit.replace('_', ' ')}, tailored to your {g.los}-night stay.{inc}\n\nShould you wish to proceed, simply reply to this message "
                        f"and our concierge team will take care of everything.\n\nWarm regards,\nGuest Experience Team", rationale="")


def _spam_copy(g: GuestProfile, sku: str, price: float):
    return CopyOut(subject=f"ACT NOW {g.first_name}!! 50% OFF!!", body=f"CHEAP {sku.replace('_', ' ')} for only ${price:,.0f} - 50% OFF, ACT NOW, last chance!!", rationale="")


def _channel(g: GuestProfile, shorten: bool) -> str:
    return "SMS" if (g.lead_time <= 2 or shorten) else "EMAIL"


def strategist_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    g = GuestProfile(**state["guest"]); d = Directives(**state.get("directives", {})); rev = int(state.get("revision", 0)); inj = state.get("injections", {})
    cands = get_engine().rank(g, exclude=set(d.exclude_skus), min_price=d.min_price)
    live = [c for c in cands if not c["excluded"]]
    viable = [c for c in live if c["unit_margin"] > c["hurdle"]] or live or cands
    pick = next((c for c in live if c["sku"] == inj.get("prefer_sku")), viable[0])        # prefer_sku = stress hook (targeted pick)
    sku, price = pick["sku"], pick["price"]
    if "price_mult_rev0" in inj and rev <= inj.get("price_mult_until", 0):                        # stress hook: over-discount
        price = round(pick["list_price"] * inj["price_mult_rev0"], 2)
    bundle = list(SWEETENERS.get(g.segment, []))
    if not inj.get("naive_bundle", False):                                                         # tier-aware generator
        bundle = [a for a in bundle if a not in g.existing_tier_perks]
    bundle = [a for a in bundle if a not in d.remove_perks]
    room_to = UPGRADE_PATH.get(g.room_type) if sku == "ROOM_UPGRADE" else None
    channel = _channel(g, d.shorten_copy); llm = get_llm(); copy = None
    aggressive = rev < inj.get("aggressive_rev_lt", 0)
    if aggressive:
        copy = _spam_copy(g, sku, price)
    else:
        sys_p = ("You are the Revenue & Personalization Strategist of a five-star resort group. Write refined, warm, understated luxury copy. "
                 f"Banned: {', '.join(['CHEAP', '% OFF', 'ACT NOW', 'LAST CHANCE', 'HURRY'])}, exclamation marks, urgency pressure. "
                 f"Max {POL['tone']['max_copy_chars'][channel]} characters for the body. Use the price EXACTLY as given.")
        usr = (f"Guest: {g.first_name} ({g.segment}, {g.loyalty_tier} tier, {g.los} nights, arriving in {g.lead_time} days). Offer: {LABEL[sku]}"
               f"{' to ' + room_to if room_to else ''} at ${price:,.0f} {UNIT[sku].replace('_', ' ')}. Included amenities: {bundle or 'none'}. Channel: {channel}. "
               f"Quant rationale (for the 'rationale' field): CATE={pick['cate_dragon']:.3f}, P(accept)={pick['p_accept']:.2f}, margin ${pick['unit_margin']:.0f}. "
               f"{'Auditor feedback to honour: refined tone, concise.' if d.refined_tone or d.shorten_copy else ''}")
        copy = llm.structured(CopyOut, sys_p, usr)
        if copy is not None and f"{price:,.0f}" not in copy.body:                                  # numbers must come from tools
            copy = None
        if copy is None:
            copy = _refined_copy(g, sku, price, UNIT[sku], bundle, room_to, channel)
    if d.shorten_copy:
        copy.body = copy.body[: POL["tone"]["max_copy_chars"][channel] - 3].rsplit(" ", 1)[0] + "..."
    rationale = (f"{sku}: DragonNet CATE={pick['cate_dragon']:+.3f} (X-learner {pick['cate_xlearner']:+.3f}); elasticity-optimal price ${pick['price']:.0f} with "
                 f"P(accept)={pick['p_accept']:.2f}; unit margin ${pick['unit_margin']:.0f} vs shadow-price hurdle ${pick['hurdle']:.1f} "
                 f"(lambda_inv={pick['lam_inv']:.1f}, lambda_B={pick['lam_budget']:.2f}); bandit exploration bonus ${pick['ucb_bonus']:.1f}; score {pick['score']:.1f}.")
    draft = UpsellOfferDraft(reservation_id=g.reservation_id, sku=sku, target_room_type=room_to, offer_price=price, list_price=pick["list_price"], pricing_unit=UNIT[sku],
                             bundled_amenities=bundle, channel=channel, subject=copy.subject, body_copy=copy.body, quant_rationale=rationale,
                             quant={k: pick[k] for k in ("cate_dragon", "cate_xlearner", "p_accept", "unit_margin", "hurdle", "lam_inv", "lam_budget", "ucb_bonus", "score", "gpu_latency_ms")},
                             revision=rev, asset_url=inj.get("asset_url") if not d.drop_asset else None, llm_mode=llm.mode)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    ev = {"agent": "strategist", "revision": rev, "sku": sku, "price": price, "bundle": bundle, "channel": channel, "llm_mode": draft.llm_mode, "latency_ms": ms,
          "ranked": [(c["sku"], c["score"]) for c in cands]}
    trace({"thread_id": state.get("thread_id"), **ev})
    return {"draft": draft.model_dump(), "history": [ev]}
