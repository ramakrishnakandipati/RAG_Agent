"""Bounded 6-node LangGraph state machine with pre/post validation gates and a guardrail retry loop.

START -> ingest_and_resolve -> score_guest -> select_strategy -> apply_guardrails -(retry<=2)-> select_strategy
                                                                      |-> compose_and_dispatch -> log_and_attribute -> END
Any failed gate short-circuits to log_and_attribute with state.error set.
"""
from __future__ import annotations
import time, uuid
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from agent import validators as V, tools as T, policy_engine as P
from agent.llm_client import structured
from agent.state import AgentState, BookingWebhookSchema, OfferStrategySchema, DowngradeReasonSchema
from config.settings import get_settings, load_catalog
from models.uplift_pipeline import row_features

MAX_GUARDRAIL_RETRIES = 2
CHANNEL = {"pre_arrival": "email", "on_property": "wifi_portal", "post_stay": "email"}


def _step(node, pre, post):
    """Decorator: run pre gate, node body, post gate; record trace; convert failures into state.error."""
    def deco(fn):
        def wrapped(state: AgentState) -> dict:
            t0 = time.perf_counter(); gates = []
            g = pre(state); gates.append(g)
            if not g["passed"]:
                return {"gates": gates, "error": f"{node}: pre-condition failed", "trace": [{"node": node, "ok": False, "ms": 0}]}
            try:
                upd = fn(state)
            except Exception as e:  # defensive: never crash the graph
                return {"gates": gates, "error": f"{node}: {type(e).__name__}: {e}", "trace": [{"node": node, "ok": False, "ms": 0}]}
            merged = {**state, **upd}; g2 = post(merged); gates.append(g2)
            upd["gates"] = gates
            upd["trace"] = [{"node": node, "ok": g2["passed"], "ms": round((time.perf_counter() - t0) * 1000, 2), "summary": upd.get("_summary", "")}]
            upd.pop("_summary", None)
            if not g2["passed"]:
                upd["error"] = f"{node}: post-condition failed: " + ", ".join(c["check"] for c in g2["checks"] if not c["passed"])
            return upd
        wrapped.__name__ = node
        return wrapped
    return deco


# ---------------------------------------------------------------- node 1
@_step("ingest_and_resolve", V.pre_ingest, V.post_ingest)
def ingest_and_resolve(state):
    wh = BookingWebhookSchema.model_validate(state["webhook"]).model_dump()
    prof, meta = T.resolve_identity(wh)
    holdout = wh["is_ghost_holdout"] if wh["is_ghost_holdout"] is not None else T.assign_holdout(wh["booking_id"])
    if prof is None:
        prof = {"profile_id": f"SHADOW-{wh['booking_id']}", "shadow": True}      # new-to-CDP guest: shadow profile
    sc = round(wh["adr"] * wh["los"] * wh["ota_commission"], 2)
    return {"webhook": wh, "resolved_profile": prof, "resolution": meta, "saved_commission_usd": sc, "is_holdout": bool(holdout),
            "_summary": f"{meta['method']} -> {prof['profile_id']}; saved commission ${sc}"}


# ---------------------------------------------------------------- node 2 (score_guest; custom wrapper so features are built before the pre-gate)
def _features_and_scores(state):
    wh = state["webhook"]; prof = state["resolved_profile"]
    return row_features(wh["ota"], wh["adr"], wh["los"], wh["lead_days"], wh["ota_commission"], wh["free_cancellation"], wh["trip_purpose"],
                        None if prof.get("shadow") else prof)


def _score_node(state):  # features computed before pre-gate so the gate can validate completeness
    st = dict(state); st["features"] = _features_and_scores(state)
    gates0 = V.pre_score(st)
    if not gates0["passed"]:
        return {"features": st["features"], "gates": [gates0], "error": "score_guest: pre-condition failed", "trace": [{"node": "score_guest", "ok": False, "ms": 0}]}
    t0 = time.perf_counter(); scores = T.scoring_service().score(st["features"])
    merged = {**st, "scores": scores}; g2 = V.post_score(merged)
    out = {"features": st["features"], "scores": scores, "gates": [gates0, g2],
           "trace": [{"node": "score_guest", "ok": g2["passed"], "ms": round((time.perf_counter() - t0) * 1000, 2),
                      "summary": f"tau={scores['tau']:.3f} p_cancel={scores['p_cancel']:.3f} ({scores['device']}, {scores['latency_ms']}ms)"}]}
    if not g2["passed"]:
        out["error"] = "score_guest: post-condition failed: " + ", ".join(c["check"] for c in g2["checks"] if not c["passed"])
    return out


# ---------------------------------------------------------------- node 3
def _template_copy(state, offer_id: str) -> str:
    wh = state["webhook"]; o = P.catalog_offers()[offer_id]
    if offer_id == P.NO_OFFER:
        return f"Hi {wh['first_name']}, thank you for choosing us - we look forward to hosting you."
    lead = {"pre_arrival": f"We're looking forward to welcoming you on {wh['checkin_date']}.",
            "on_property": "Welcome! Thanks for connecting to our Wi-Fi.",
            "post_stay": "Thank you for staying with us - we hope you enjoyed it."}[wh["window"]]
    return f"Hi {wh['first_name']}, {lead} As a thank-you for booking with us, enjoy {o['label']} when you book direct as a member."


@_step("select_strategy", V.pre_strategy, V.post_strategy)
def select_strategy(state):
    wh, sc, tau, pc = state["webhook"], state["saved_commission_usd"], state["scores"]["tau"], state["scores"]["p_cancel"]
    rejected = state.get("rejected_offers", [])
    cands = {k: v for k, v in P.catalog_offers().items() if v["window"] in (wh["window"], "any") and k not in rejected}
    cands = {k: v for k, v in cands.items() if v["kind"] != "public_rate"}   # parity-breaking offers are never offered to the model

    def fallback():
        act = {k: P.expected_net_revenue_delta(tau, pc, sc, v["cost_usd"]) for k, v in cands.items() if k != P.NO_OFFER}
        best = max(act, key=act.get) if act and max(act.values()) > 0 else P.NO_OFFER   # NO_OFFER is worth exactly 0
        return OfferStrategySchema(window=wh["window"], offer_id=best, channel=CHANNEL[wh["window"]],
                                   rationale=f"Offline policy: highest expected net revenue delta (tau={tau:.3f}, p_cancel={pc:.3f}).",
                                   personalized_copy=_template_copy(state, best))
    system = ("You are a hospitality direct-booking strategist. Choose exactly ONE offer_id from the candidate list for the active window and write a short, "
              "warm, personalised message (no prices, no public discounts). Respond with JSON only.")
    user = (f"Guest: {wh['first_name']} ({wh['trip_purpose']}, via {wh['ota']}, {wh['los']} nights, ADR ${wh['adr']}). Window: {wh['window']}. "
            f"Uplift tau={tau:.3f}, cancel risk={pc:.3f}, saved commission=${sc}. Channel hint: {CHANNEL[wh['window']]}.\n"
            f"Candidates: {[{k: (v['label'], v['cost_usd']) for k, v in cands.items()}]}\nAlready rejected by policy: {rejected}")
    obj, meta = structured(system, user, OfferStrategySchema, fallback, max_retries=2)
    return {"strategy": obj.model_dump(), "llm_mode": meta["mode"], "llm_attempts": meta["attempts"],
            "_summary": f"{obj.offer_id} via {meta['mode']} (attempts={meta['attempts']})"}


# ---------------------------------------------------------------- node 4
def _verdict(state, offer_id):
    wh, prof = state["webhook"], state["resolved_profile"]; sh = bool(prof.get("shadow"))
    return dict(tau=state["scores"]["tau"], p_cancel=state["scores"]["p_cancel"], saved_commission=state["saved_commission_usd"],
                stay_revenue=wh["adr"] * wh["los"], ota_rate=wh.get("ota_public_rate"), direct_rate=wh.get("direct_public_rate"),
                msgs_30d=0 if sh else int(prof.get("msgs_30d", 0)), opt_in=False if sh else bool(prof.get("email_opt_in", False)), resolved=not sh)


@_step("apply_guardrails", V.pre_guardrails, V.post_guardrails)
def apply_guardrails(state):
    wh, offer = state["webhook"], state["strategy"]["offer_id"]; kw = _verdict(state, offer)
    v = P.evaluate(offer, wh["window"], **kw); retries = state.get("guardrail_retries", 0)
    audit = {"run_id": state["run_id"], "ts": T.now(), "booking_id": wh["booking_id"], "proposed_offer": offer, "verdict": v.asdict(), "retry": retries}
    T.jsonl_append("guardrail_audits.jsonl", audit)
    if v.approved:
        return {"guardrail": {**v.asdict(), "retry": False}, "final_offer_id": offer, "downgraded": state.get("downgraded", False), "_summary": f"approved {offer}"}
    rejected = state.get("rejected_offers", []) + [offer]
    if retries < MAX_GUARDRAIL_RETRIES:
        return {"guardrail": {**v.asdict(), "approved": False, "retry": True}, "rejected_offers": rejected, "guardrail_retries": retries + 1,
                "final_offer_id": "", "_summary": f"blocked {offer} by {v.blocking_rules}; retry {retries + 1}"}
    # bounded: deterministic downgrade
    final, fv = P.best_compliant_offer(wh["window"], rejected, **kw)
    reason_txt = f"{offer} blocked by {', '.join(v.blocking_rules)}; downgraded to {final}."
    obj, _ = structured("Explain in one concise sentence why the offer was downgraded. JSON only.", reason_txt, DowngradeReasonSchema,
                        lambda: DowngradeReasonSchema(summary=reason_txt), max_retries=1)
    T.jsonl_append("guardrail_audits.jsonl", {**audit, "downgraded_to": final, "reason": obj.summary, "final_verdict": fv.asdict()})
    return {"guardrail": {**fv.asdict(), "retry": False, "downgrade_reason": obj.summary, "initial_blocking": v.blocking_rules}, "rejected_offers": rejected,
            "final_offer_id": final, "downgraded": True, "_summary": f"downgraded {offer} -> {final}"}


# ---------------------------------------------------------------- node 5
@_step("compose_and_dispatch", V.pre_dispatch, V.post_dispatch)
def compose_and_dispatch(state):
    wh = state["webhook"]; offer = state["final_offer_id"]; inv = T.crs_inventory(wh["room_type"], wh["checkin_date"])
    if offer == "CUG_ROOM_UPGRADE" and not inv["upgrade_available"]:               # CRS inventory gate
        alt = "CUG_LATE_CHECKOUT"; v = P.evaluate(alt, wh["window"], **_verdict(state, alt)); offer = alt if v.approved else P.NO_OFFER
    strat = state["strategy"]
    copy = strat["personalized_copy"] if offer == strat["offer_id"] else _template_copy(state, offer)
    if offer == P.NO_OFFER:
        return {"final_offer_id": offer, "dispatch": {"status": "no_action", "channel": None, "link": None, "link_valid": True, "inventory": inv}, "_summary": "no offer"}
    link = T.gated_link(wh["booking_id"], offer)
    d = T.dispatch(strat["channel"], wh["booking_id"], f"{copy} {link}", link, state["is_holdout"])
    return {"final_offer_id": offer, "dispatch": {**d, "link": link, "link_valid": T.verify_link(link), "copy": copy, "inventory": inv},
            "_summary": f"{d['status']} {offer} via {strat['channel']}"}


# ---------------------------------------------------------------- node 6
@_step("log_and_attribute", V.pre_log, V.post_log)
def log_and_attribute(state):
    wh, g = state.get("webhook", {}), state.get("guardrail", {})
    offer = state.get("final_offer_id") or P.NO_OFFER
    sent = state.get("dispatch", {}).get("status") == "sent"
    attr = {"run_id": state["run_id"], "ts": T.now(), "booking_id": wh.get("booking_id"), "profile_id": (state.get("resolved_profile") or {}).get("profile_id"),
            "ota": wh.get("ota"), "window": wh.get("window"), "treatment": bool(sent), "ghost_holdout": bool(state.get("is_holdout")), "offer_id": offer,
            "tau": state.get("scores", {}).get("tau"), "p_cancel": state.get("scores", {}).get("p_cancel"), "saved_commission_usd": state.get("saved_commission_usd"),
            "expected_net_revenue_delta": g.get("enrd"), "downgraded": state.get("downgraded", False), "llm_mode": state.get("llm_mode"), "error": state.get("error")}
    T.jsonl_append("agent_decisions.jsonl", {**attr, "strategy": state.get("strategy"), "guardrail": g, "trace": state.get("trace")})
    return {"attribution": attr, "_summary": "logged"}


# ---------------------------------------------------------------- routing / build
def _after_gate(nxt):
    return lambda s: "log_and_attribute" if s.get("error") else nxt


def _route_guardrails(s):
    if s.get("error"): return "log_and_attribute"
    return "select_strategy" if s.get("guardrail", {}).get("retry") else "compose_and_dispatch"


@lru_cache
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("ingest_and_resolve", ingest_and_resolve); g.add_node("score_guest", _score_node); g.add_node("select_strategy", select_strategy)
    g.add_node("apply_guardrails", apply_guardrails); g.add_node("compose_and_dispatch", compose_and_dispatch); g.add_node("log_and_attribute", log_and_attribute)
    g.add_edge(START, "ingest_and_resolve")
    g.add_conditional_edges("ingest_and_resolve", _after_gate("score_guest"), ["score_guest", "log_and_attribute"])
    g.add_conditional_edges("score_guest", _after_gate("select_strategy"), ["select_strategy", "log_and_attribute"])
    g.add_conditional_edges("select_strategy", _after_gate("apply_guardrails"), ["apply_guardrails", "log_and_attribute"])
    g.add_conditional_edges("apply_guardrails", _route_guardrails, ["select_strategy", "compose_and_dispatch", "log_and_attribute"])
    g.add_conditional_edges("compose_and_dispatch", _after_gate("log_and_attribute"), ["log_and_attribute"])
    g.add_edge("log_and_attribute", END)
    return g.compile()


def run_agent(webhook: dict, run_id: str | None = None) -> dict:
    run_id = run_id or uuid.uuid4().hex[:12]
    final = build_graph().invoke({"run_id": run_id, "webhook": webhook, "rejected_offers": [], "guardrail_retries": 0, "gates": [], "trace": [], "error": None},
                                 {"recursion_limit": 40})
    V.write_step_gates(run_id, final.get("gates", []))
    return final
