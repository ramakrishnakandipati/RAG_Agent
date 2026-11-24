"""Five runnable end-to-end use cases (shown in the Streamlit 'Use Cases' tab, captured by Playwright, listed in the README).

Each scenario picks a suitable *upcoming* reservation automatically (pre-screened with the real offer engine, so no LLM
calls are wasted), then executes its steps through the real LangGraph workflow.
"""
from __future__ import annotations
import time
from typing import Callable, Optional

from src.graph.workflow import resume_workflow, run_workflow
from src.schemas.domain import WorkflowResponse
from src.tools import environment_tools as envt
from src.tools import nba_rms_engine as nba
from src.tools import pms_crm_tools as pms

ACCEPT = "Yes please, add it to my booking!"
VIP_CUSTOM = "Could you arrange a private surprise anniversary dinner on the beach for us?"

SCENARIOS: list[dict] = [
    {"id": "uc1_upsell_accept", "title": "1 · Pre-arrival upsell, guest accepts, folio booking",
     "persona": "Any opted-in guest with a high-propensity ancillary (spa / dining / upgrade / late checkout)",
     "story": "Agent scores eligible SKUs on GPU, applies RMS price/margin rules, Qwen3-VL drafts warm copy with placeholders, the gate verifies "
              "every number, message is sent. The guest replies YES, live inventory is checked and the item is posted to the room folio.",
     "outcome": "SENT, then BOOKED on the folio",
     "steps": [{"kind": "trigger", "title": "Pre-arrival offer"}, {"kind": "reply", "title": "Guest replies YES", "text": ACCEPT}]},
    {"id": "uc2_ota_to_direct", "title": "2 · OTA guest converted to direct-booking loyalty member",
     "persona": "Booked via an OTA, no loyalty status, offer engine ranks the loyalty offer first",
     "story": "The highest expected-value action is the free Azure Circle membership (avoids 15-25% OTA commission on the next stay). "
              "The offer is a fixed 10% direct-booking benefit supplied by the RMS payload, never invented by the LLM.",
     "outcome": "SENT loyalty offer, guest accepts, membership posted",
     "steps": [{"kind": "trigger", "title": "Channel-shift offer"}, {"kind": "reply", "title": "Guest accepts membership", "text": ACCEPT}]},
    {"id": "uc3_guardrail_retry", "title": "3 · Hallucinated price is blocked, redraft passes",
     "persona": "Any opted-in guest; the first draft is deliberately poisoned (test hook)",
     "story": "The first draft claims a fake SKU, a $49 price and 40% off. The deterministic gate finds three token violations (FAIL_POLICY), "
              "routes back to draft_message with the violations as feedback, and the second draft passes. The poisoned text never reaches the guest.",
     "outcome": "FAIL_POLICY, retry, PASS, SENT (retries = 1)",
     "steps": [{"kind": "trigger", "title": "Offer with poisoned first draft", "adversarial": "once"}]},
    {"id": "uc4_policy_escalation", "title": "4 · Repeated policy failure escalates to a human concierge",
     "persona": "Any opted-in guest; every draft is poisoned (test hook)",
     "story": "Retries are exhausted, so the graph pauses at hitl_gate with interrupt() and a safe fallback draft is queued. "
              "The concierge approves and the paused graph resumes from its SQLite checkpoint.",
     "outcome": "ESCALATED, then SENT with the safe fallback draft",
     "steps": [{"kind": "trigger", "title": "Offer, every draft poisoned", "adversarial": "persistent"},
               {"kind": "approve", "title": "Concierge approves safe fallback", "decision": "approve"}]},
    {"id": "uc5_vip_custom_request", "title": "5 · Platinum VIP custom request routed to concierge",
     "persona": "Platinum loyalty guest",
     "story": "After the offer is sent the guest asks for a bespoke beach dinner. Qwen3-VL plus keyword safety net flags a custom request from a "
              "Platinum guest; the agent never improvises, it queues a holding reply for a human.",
     "outcome": "ESCALATED (vip_platinum_custom_request), concierge approves holding reply",
     "steps": [{"kind": "trigger", "title": "Pre-arrival offer"}, {"kind": "reply", "title": "VIP asks for a custom experience", "text": VIP_CUSTOM},
               {"kind": "approve", "title": "Concierge approves holding reply", "decision": "approve"}]},
]
BY_ID = {s["id"]: s for s in SCENARIOS}


def to_response(res: dict) -> dict:
    st = res["state"]
    intr = res.get("interrupt") or {}
    return WorkflowResponse(
        thread_id=res["thread_id"], status=st.get("status", "UNKNOWN"), interrupted=res["interrupted"],
        hitl_reasons=st.get("hitl_reasons") or intr.get("reasons", []),
        message=(st.get("draft") if not res["interrupted"] else intr.get("draft")) or None,
        offer=st.get("selected_offer"), guardrail=st.get("guardrail"), booking=st.get("booking"),
        trace=st.get("trace", []), errors=st.get("errors", [])).model_dump(mode="json")


# ------------------------------------------------------------------------------------------ reservation picking
def _has_messages(rid: str) -> bool:
    with pms.connect() as con:
        return con.execute("SELECT COUNT(*) FROM outbox WHERE reservation_id=?", (rid,)).fetchone()[0] > 0


def _offers(r: dict) -> list[dict]:
    g, st = pms.get_guest_360(r["guest_id"]), pms.get_stay(r["reservation_id"])
    env = envt.get_environmental_signals(str(st["check_in"]), str(st["check_out"]))
    return nba.generate_offers(g, st, env)


PICKERS: dict[str, tuple[Callable[[dict], bool], Callable[[list[dict]], bool]]] = {
    "uc1_upsell_accept": (lambda r: True, lambda o: bool(o) and o[0]["category"] != "loyalty"),
    "uc2_ota_to_direct": (lambda r: r["channel_group"] == "OTA" and r["loyalty_tier"] in ("Basic", "Silver"),
                          lambda o: bool(o) and o[0]["sku"] == "LOY-DIRECT"),
    "uc3_guardrail_retry": (lambda r: True, lambda o: bool(o) and o[0]["category"] != "loyalty"),
    "uc4_policy_escalation": (lambda r: True, lambda o: bool(o) and o[0]["category"] != "loyalty"),
    "uc5_vip_custom_request": (lambda r: r["loyalty_tier"] == "Platinum", lambda o: bool(o)),
}


def pick_reservation(sid: str) -> Optional[dict]:
    flt, pred = PICKERS[sid]
    for r in pms.list_upcoming_reservations(limit=4000, min_days=1, max_days=14):
        if not flt(r) or _has_messages(r["reservation_id"]):
            continue
        g = pms.get_guest_360(r["guest_id"])
        if not g or not g["opt_in_marketing"] or pms.recent_message_count(r["guest_id"]) >= 2:
            continue
        if pred(_offers(r)):
            return r
    return None


# ------------------------------------------------------------------------------------------ execution
def run_scenario(sid: str) -> dict:
    sc = BY_ID[sid]
    r = pick_reservation(sid)
    if not r:
        return {"scenario": {k: sc[k] for k in ("id", "title", "persona", "story", "outcome")},
                "error": "no suitable reservation left for this scenario (all candidates already messaged); reset runtime tables"}
    guest = pms.get_guest_360(r["guest_id"])
    out_steps, thread, t0 = [], None, time.perf_counter()
    for i, step in enumerate(sc["steps"], 1):
        ts = time.perf_counter()
        if step["kind"] == "trigger":
            res = run_workflow(r["reservation_id"], adversarial=step.get("adversarial", "none"))
            req = {"endpoint": "POST /api/v1/trigger", "body": {"reservation_id": r["reservation_id"], "adversarial": step.get("adversarial", "none")}}
        elif step["kind"] == "reply":
            res = run_workflow(r["reservation_id"], trigger_type="inbound_reply", inbound_text=step["text"])
            req = {"endpoint": "POST /api/v1/reply", "body": {"reservation_id": r["reservation_id"], "text": step["text"]}}
        else:
            res = resume_workflow(thread, {"decision": step["decision"], "reviewer": "demo_concierge"})
            req = {"endpoint": "POST /api/v1/hitl/approve", "body": {"thread_id": thread, "decision": step["decision"]}}
        thread = res["thread_id"]
        resp = to_response(res)
        st = res["state"]
        resp["intent"] = st.get("intent")
        resp["retries"] = st.get("retries", 0)
        resp["violations_seen"] = st.get("feedback") or []
        out_steps.append({"n": i, "title": step["title"], "kind": step["kind"], "request": req, "response": resp,
                          "seconds": round(time.perf_counter() - ts, 1)})
    return {"scenario": {k: sc[k] for k in ("id", "title", "persona", "story", "outcome")},
            "guest": {k: guest[k] for k in ("guest_id", "first_name", "segment", "loyalty_tier", "ltv_usd")},
            "reservation": r, "steps": out_steps, "folio": pms.get_folio(r["reservation_id"]),
            "total_seconds": round(time.perf_counter() - t0, 1)}
