"""Compiled LangGraph StateGraph with SQLite checkpointer (HITL interrupts survive restarts)."""
from __future__ import annotations
import sqlite3
import uuid
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.config import get_settings
from src.graph import edges, nodes
from src.schemas.state import AgentState

_graph = None


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for n in (nodes.enrich_context, nodes.retrieve_knowledge_and_offers, nodes.draft_message,
              nodes.compliance_check, nodes.hitl_gate, nodes.dispatch_and_log):
        g.add_node(n.__name__, n)
    g.add_edge(START, "enrich_context")
    g.add_conditional_edges("enrich_context", edges.route_after_enrich, ["retrieve_knowledge_and_offers", END])
    g.add_conditional_edges("retrieve_knowledge_and_offers", edges.route_after_retrieve, ["draft_message", END])
    g.add_conditional_edges("draft_message", edges.route_after_draft, ["compliance_check", END])
    g.add_conditional_edges("compliance_check", edges.route_after_compliance, ["hitl_gate", "draft_message", END])
    g.add_conditional_edges("hitl_gate", edges.route_after_hitl, ["dispatch_and_log", END])
    g.add_edge("dispatch_and_log", END)
    return g.compile(checkpointer=checkpointer)


def get_graph():
    global _graph
    if _graph is None:
        s = get_settings()
        conn = sqlite3.connect(s.checkpoint_path, check_same_thread=False)
        _graph = build_graph(SqliteSaver(conn))
    return _graph


def new_thread_id(reservation_id: str) -> str:
    return f"{reservation_id}-{uuid.uuid4().hex[:8]}"


def run_workflow(reservation_id: str, guest_id: Optional[str] = None, trigger_type: str = "pre_arrival",
                 inbound_text: Optional[str] = None, adversarial: str = "none", thread_id: Optional[str] = None,
                 now_hour: int = 10) -> dict:
    from src.tools import pms_crm_tools as pms
    if guest_id is None:
        st = pms.get_stay(reservation_id)
        guest_id = st["guest_id"] if st else ""
    tid = thread_id or new_thread_id(reservation_id)
    init = {"thread_id": tid, "guest_id": guest_id, "reservation_id": reservation_id, "trigger_type": trigger_type,
            "inbound_text": inbound_text, "adversarial": adversarial, "now_hour": now_hour, "retries": 0}
    return _result(get_graph().invoke(init, {"configurable": {"thread_id": tid}}), tid)


def resume_workflow(thread_id: str, decision: dict) -> dict:
    out = get_graph().invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})
    return _result(out, thread_id)


def _result(out: dict, tid: str) -> dict:
    snap = get_graph().get_state({"configurable": {"thread_id": tid}})
    interrupted = bool(snap.next)
    v = dict(snap.values if snap and snap.values else out)
    payload = None
    if interrupted:
        for t in snap.tasks:
            if t.interrupts:
                payload = t.interrupts[0].value
                break
        if payload:
            v["hitl_reasons"] = payload.get("reasons", [])
            v["status"] = "ESCALATED"
    return {"thread_id": tid, "interrupted": interrupted, "state": v, "interrupt": payload}
