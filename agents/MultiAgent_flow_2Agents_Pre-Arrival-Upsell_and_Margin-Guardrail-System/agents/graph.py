"""LangGraph cyclic Generator-Critic state machine with bounded revisions and a Human-in-the-Loop interrupt.

        START -> strategist -> auditor --APPROVED--------------------> finalize -> END
                    ^             |  \\--REJECTED & r<R_max (r:=r+1)-> strategist
                    |             \\----ESCALATE_HITL / unresolved ----> hitl_gate (interrupt) -> finalize -> END

Termination (docs L8): r increments only on REJECTED and the auditor escalates when r >= R_max, so at most R_max+1 drafts are produced.
HITL: `interrupt()` pauses the thread (checkpointed in SQLite); resume with Command(resume={"decision": "approve"|"reject", "reviewer": ...}).
"""
from __future__ import annotations
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .auditor_agent import auditor_node
from .schemas import GraphState
from .strategist_agent import strategist_node
from .tools import trace
from models.utils import jsonl_append, load_config, path


def route_after_audit(state: GraphState) -> str:
    d = state["verdict"]["decision"]
    return {"APPROVED": "finalize", "REJECTED": "strategist", "ESCALATE_HITL": "hitl_gate"}[d]


def hitl_gate(state: GraphState) -> dict:
    payload = {"thread_id": state.get("thread_id"), "reservation_id": state["guest"]["reservation_id"], "reason": state["verdict"].get("hitl_reason"),
               "draft": state["draft"], "violations": state["verdict"].get("violations"), "vip_value": state["verdict"].get("vip_value")}
    decision = interrupt(payload)                                       # <- pauses here until a human resumes the thread
    approved = str(decision.get("decision", "reject")).lower() == "approve"
    ev = {"agent": "human", "decision": "APPROVE" if approved else "REJECT", "reviewer": decision.get("reviewer", "unknown"), "note": decision.get("note", "")}
    trace({"thread_id": state.get("thread_id"), **ev})
    return {"final_status": "HITL_APPROVED" if approved else "HITL_REJECTED", "hitl": ev, "history": [ev]}


def finalize(state: GraphState) -> dict:
    status = state.get("final_status") or "APPROVED"
    trace({"thread_id": state.get("thread_id"), "agent": "graph", "final_status": status, "revisions": state.get("revision", 0)})
    return {"final_status": status}


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)
    g.add_node("strategist", strategist_node); g.add_node("auditor", auditor_node)
    g.add_node("hitl_gate", hitl_gate); g.add_node("finalize", finalize)
    g.add_edge(START, "strategist"); g.add_edge("strategist", "auditor")
    g.add_conditional_edges("auditor", route_after_audit, {"finalize": "finalize", "strategist": "strategist", "hitl_gate": "hitl_gate"})
    g.add_edge("hitl_gate", "finalize"); g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


_graph = None


def get_graph():
    """Process-wide graph backed by a SQLite checkpointer (so HITL threads survive restarts)."""
    global _graph
    if _graph is None:
        conn = sqlite3.connect(str(path("checkpoints") / "graph.sqlite"), check_same_thread=False)
        _graph = build_graph(SqliteSaver(conn))
    return _graph


def run_offer(guest: dict, injections: Optional[dict] = None, pms_override: Optional[dict] = None, thread_id: Optional[str] = None,
              graph=None, max_revisions: Optional[int] = None) -> dict:
    """Run one offer through the loop. Returns the final state; if a human is required, final_status == 'HITL_PENDING' and the thread is paused."""
    graph = graph or get_graph(); thread_id = thread_id or f"th-{uuid.uuid4().hex[:10]}"
    cfg = {"configurable": {"thread_id": thread_id}}
    init = {"thread_id": thread_id, "guest": guest, "injections": injections or {}, "pms_override": pms_override or {}, "revision": 0,
            "max_revisions": max_revisions if max_revisions is not None else load_config()["agents"]["max_revisions"], "directives": {}, "history": []}
    out = graph.invoke(init, cfg)
    if "__interrupt__" in out:
        out = dict(graph.get_state(cfg).values); out["final_status"] = "HITL_PENDING"
        out["interrupt"] = graph.get_state(cfg).tasks[0].interrupts[0].value if graph.get_state(cfg).tasks else {}
    out["thread_id"] = thread_id
    jsonl_append("offer_runs.jsonl", {"thread_id": thread_id, "reservation_id": guest["reservation_id"], "status": out.get("final_status"), "revisions": out.get("revision", 0),
                                      "sku": (out.get("draft") or {}).get("sku"), "price": (out.get("draft") or {}).get("offer_price")})
    return out


def resume_hitl(thread_id: str, decision: str, reviewer: str = "ops", note: str = "", graph=None) -> dict:
    graph = graph or get_graph(); cfg = {"configurable": {"thread_id": thread_id}}
    out = graph.invoke(Command(resume={"decision": decision, "reviewer": reviewer, "note": note}), cfg)
    out["thread_id"] = thread_id
    jsonl_append("offer_runs.jsonl", {"thread_id": thread_id, "event": "hitl_resolved", "status": out.get("final_status"), "reviewer": reviewer})
    return out
