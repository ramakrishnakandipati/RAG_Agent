"""The six LangGraph nodes, each wrapped with pre/post validators and structured telemetry."""
from __future__ import annotations
import time
from datetime import datetime
from functools import wraps

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from src.config import get_settings
from src.graph import drafting
from src.tools import compliance_tools as comp
from src.tools import environment_tools as envt
from src.tools import nba_rms_engine as nba
from src.tools import pms_crm_tools as pms
from src.tools.rag_retriever import get_retriever
from src.utils.logger import get_logger, telemetry
from src.validation.step_validators import VALIDATORS, ValidationFailure

log = get_logger("nodes")

DOC_TYPES = {"spa": ["spa"], "dining": ["dining"], "room_upgrade": ["rooms"], "late_checkout": ["policy"], "loyalty": ["policy"]}


def instrument(name: str):
    pre, post = VALIDATORS[name]

    def deco(fn):
        @wraps(fn)
        def wrapped(state: dict) -> dict:
            t0 = time.perf_counter()
            rec = {"node": name, "thread_id": state.get("thread_id"), "ts": datetime.now().isoformat(timespec="seconds")}
            try:
                pre(state)
                out = fn(state)
                post({**state, **out})
                rec.update(status="OK", pre_ok=True, post_ok=True)
                rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                telemetry(rec)
                return {**out, "trace": [rec]}
            except GraphBubbleUp:
                raise
            except ValidationFailure as e:
                rec.update(status="FAIL_VALIDATION", pre_ok=e.phase != "pre", post_ok=False, note="; ".join(e.errors))
                msg = str(e)
            except Exception as e:  # noqa: BLE001
                rec.update(status="ERROR", pre_ok=True, post_ok=False, note=f"{type(e).__name__}: {e}")
                msg = f"[{name}] {type(e).__name__}: {e}"
                log.error("node_error", node=name, error=msg)
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            telemetry(rec)
            return {"status": "FAIL_VALIDATION", "errors": [msg], "trace": [rec]}
        wrapped.__name__ = name
        return wrapped
    return deco


# --------------------------------------------------------------------------------------------- 1
@instrument("enrich_context")
def enrich_context(state: dict) -> dict:
    guest = pms.get_guest_360(state["guest_id"])
    stay = pms.get_stay(state["reservation_id"])
    env = envt.get_environmental_signals(str(stay["check_in"]), str(stay["check_out"])) if stay else None
    return {"guest": guest, "stay": stay, "env": env, "status": "RUNNING", "retries": state.get("retries", 0), "feedback": []}


# --------------------------------------------------------------------------------------------- 2
@instrument("retrieve_knowledge_and_offers")
def retrieve_knowledge_and_offers(state: dict) -> dict:
    g, st, env = state["guest"], state["stay"], state["env"]
    inbound = state.get("trigger_type") == "inbound_reply"
    last_sku = None
    if inbound:
        with pms.connect() as con:
            row = con.execute("SELECT sku FROM outbox WHERE reservation_id=? AND sku IS NOT NULL ORDER BY id DESC LIMIT 1",
                              (st["reservation_id"],)).fetchone()
            last_sku = row["sku"] if row else None
    offers = nba.generate_offers(g, st, env, top_k=8 if inbound else 3, threshold=0.0 if inbound else None)
    selected = None
    if offers:
        selected = next((o for o in offers if o["sku"] == last_sku), None) if last_sku else None
        selected = selected or offers[0]
    ret = get_retriever()
    cat = selected["category"] if selected else "spa"
    q = f"{selected['name'] if selected else 'resort'} {cat.replace('_', ' ')} details hours policy"
    if inbound and state.get("inbound_text"):
        q = f"{state['inbound_text']} {q}"
    snippets = ret.search(q, k=3, doc_types=DOC_TYPES.get(cat)) + ret.search("brand voice tone guidelines", k=2, doc_types=["brand"])
    keep = [{"text": s["text"], "source": s["source"], "doc_type": s["doc_type"], "score": s.get("rerank", s["rrf"])} for s in snippets]
    out = {"candidates": offers, "selected_offer": selected, "kb_snippets": keep}
    if not offers:
        out["status"] = "NO_OFFER"
    return out


# --------------------------------------------------------------------------------------------- 3
@instrument("draft_message")
def draft_message(state: dict) -> dict:
    offer = state["selected_offer"]
    out: dict = {}
    action = "offer"
    if state.get("trigger_type") == "inbound_reply":
        intent = state.get("intent") or drafting.classify_reply(state.get("inbound_text", ""), offer)
        out["intent"] = intent
        i = intent["intent"]
        if i == "accept":
            avail = pms.get_availability(offer["sku"], str(offer["service_date"]))
            action = "confirm" if (avail > 0 or offer["category"] == "loyalty") else "ack_decline"
        elif i == "decline":
            action = "ack_decline"
        elif i == "question":
            action = "answer"
        elif i in ("complaint", "custom_request"):
            action = "holding"
        else:
            action = "other"
    retries = state.get("retries", 0)
    mode = state.get("adversarial", "none")
    if mode in ("once", "persistent", True) and (retries == 0 or mode == "persistent"):
        d = drafting.poison_draft(action)
    else:
        d = drafting.draft_with_llm(state, action)
    out["draft"] = d
    out["draft"]["body"] = ""  # rendered by compliance_check
    return out


# --------------------------------------------------------------------------------------------- 4
@instrument("compliance_check")
def compliance_check(state: dict) -> dict:
    d = dict(state["draft"])
    facts = comp.build_facts(state)
    d["body"] = comp.render(d["body_template"], facts)
    d["subject"] = comp.render(d["subject"], facts)
    res = comp.verify_compliance_and_caps({**state, "draft": d})
    out = {"draft": d, "guardrail": res}
    if res["status"] == "FAIL_POLICY":
        out["retries"] = state.get("retries", 0) + 1
        out["feedback"] = res["violations"]
        out["status"] = "RUNNING"
    elif res["status"] == "FAIL_VALIDATION":
        out["status"] = "FAIL_VALIDATION"
        out["errors"] = res["violations"]
    return out


# --------------------------------------------------------------------------------------------- 5
@instrument("hitl_gate")
def hitl_gate(state: dict) -> dict:
    g, offer = state["guest"], state["selected_offer"]
    intent = state.get("intent") or {}
    reasons: list[str] = []
    draft = state["draft"]
    if state["guardrail"]["status"] != "PASS":
        reasons.append("repeated_policy_failure")
        action = draft.get("action", "offer")
        facts = comp.build_facts(state)
        fb = drafting.safe_fallback(state, "holding" if state.get("trigger_type") == "inbound_reply" else "offer")
        fb["body"], fb["subject"] = comp.render(fb["body_template"], facts), comp.render(fb["subject"], facts)
        draft = fb
    if intent.get("sentiment") == "negative" or intent.get("sentiment_score", 0) < -0.3:
        reasons.append("negative_sentiment")
    if g["loyalty_tier"] == "Platinum" and intent.get("custom_request"):
        reasons.append("vip_platinum_custom_request")
    if not reasons:
        return {"hitl_required": False, "hitl_reasons": []}
    pms.enqueue_hitl(state["thread_id"], state["reservation_id"], g["guest_id"], reasons, draft, offer)
    decision = interrupt({"thread_id": state["thread_id"], "reasons": reasons, "draft": draft, "offer": offer,
                          "inbound_text": state.get("inbound_text")})
    decision = decision if isinstance(decision, dict) else {"decision": str(decision)}
    pms.resolve_hitl(state["thread_id"], decision)
    pms.write_audit(state["thread_id"], state["reservation_id"], g["guest_id"], "hitl_decision", decision.get("decision", "?"),
                    offer["sku"], None, {"reasons": reasons, "reviewer": decision.get("reviewer")})
    if decision.get("decision") == "approve" and decision.get("edited_body"):
        draft = dict(draft, body=decision["edited_body"], source="human_edited")
    out = {"hitl_required": True, "hitl_reasons": reasons, "hitl_decision": decision, "draft": draft}
    if decision.get("decision") == "reject":
        out["status"] = "REJECTED"
    return out


# --------------------------------------------------------------------------------------------- 6
@instrument("dispatch_and_log")
def dispatch_and_log(state: dict) -> dict:
    d = state["draft"]
    # last line of defence: re-scan the final text (covers human edits and fallback drafts)
    viol = comp.check_text_tokens(f"{d['subject']}\n{d['body']}", state["selected_offer"])
    if viol:
        pms.write_audit(state["thread_id"], state["reservation_id"], state["guest_id"], "dispatch_blocked", "BLOCKED",
                        detail={"violations": viol})
        return {"status": "BLOCKED", "dispatched": False, "errors": viol}
    return comp.dispatch_and_log(state)
