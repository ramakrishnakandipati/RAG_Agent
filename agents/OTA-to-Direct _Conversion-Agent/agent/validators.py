"""Pre/post-condition validators for the six LangGraph nodes. Each returns a GateResult; failures are logged to step_gates."""
from __future__ import annotations
import math, time
from pydantic import ValidationError
from agent.state import BookingWebhookSchema, OfferStrategySchema, GateResult
from agent.policy_engine import catalog_offers
from config.settings import get_settings
from models.uplift_pipeline import FEATURES

NODES = ["ingest_and_resolve", "score_guest", "select_strategy", "apply_guardrails", "compose_and_dispatch", "log_and_attribute"]


def _g(node, phase, checks, t0):
    return GateResult(node=node, phase=phase, passed=all(c["passed"] for c in checks), checks=checks, ms=round((time.perf_counter() - t0) * 1000, 3)).model_dump()


def c(name, ok, detail=None):
    return {"check": name, "passed": bool(ok), "detail": detail}


def pre_ingest(state):
    t0 = time.perf_counter()
    try:
        BookingWebhookSchema.model_validate(state.get("webhook", {})); ok, d = True, None
    except ValidationError as e:
        ok, d = False, [x["loc"] for x in e.errors()][:5]
    return _g("ingest_and_resolve", "pre", [c("webhook_schema", ok, d)], t0)


def post_ingest(state):
    t0 = time.perf_counter()
    r = state.get("resolution", {})
    return _g("ingest_and_resolve", "post", [
        c("resolution_attempted", "method" in r, r.get("method")),
        c("resolved_profile_id_present", bool(state.get("resolved_profile")), r.get("profile_id")),
        c("saved_commission_usd_gt_0", state.get("saved_commission_usd", 0) > 0, state.get("saved_commission_usd")),
        c("holdout_flag_set", isinstance(state.get("is_holdout"), bool), state.get("is_holdout"))], t0)


def pre_score(state):
    t0 = time.perf_counter(); f = state.get("features", {})
    miss = [k for k in FEATURES if k not in f]
    return _g("score_guest", "pre", [c("feature_vector_complete", not miss, miss)], t0)


def post_score(state):
    t0 = time.perf_counter(); f = state.get("features", {}); s = state.get("scores", {})
    nans = [k for k, v in f.items() if isinstance(v, float) and math.isnan(v)]
    tau, pc = s.get("tau", float("nan")), s.get("p_cancel", float("nan"))
    return _g("score_guest", "post", [
        c("zero_nans", not nans, nans),
        c("tau_in_[-1,1]", -1 <= tau <= 1, tau), c("p_cancel_in_[0,1]", 0 <= pc <= 1, pc),
        c("inference_latency_lt_1000ms", s.get("latency_ms", 1e9) < 1000, {"ms": s.get("latency_ms"), "device": s.get("device")})], t0)


def pre_strategy(state):
    t0 = time.perf_counter()
    return _g("select_strategy", "pre", [c("scores_present", "tau" in state.get("scores", {})), c("window_valid", state["webhook"]["window"] in ("pre_arrival", "on_property", "post_stay"))], t0)


def post_strategy(state):
    t0 = time.perf_counter(); st = state.get("strategy", {})
    try:
        o = OfferStrategySchema.model_validate(st); ok, d = True, None
    except ValidationError as e:
        o, ok, d = None, False, str(e)[:200]
    cat = catalog_offers()
    return _g("select_strategy", "post", [
        c("offer_strategy_schema", ok, d),
        c("valid_offer_id_in_catalog", bool(o) and o.offer_id in cat, st.get("offer_id")),
        c("valid_window", bool(o) and o.window == state["webhook"]["window"], st.get("window")),
        c("non_empty_personalized_copy", bool(o) and len(o.personalized_copy.strip()) >= 10),
        c("llm_retries_le_2", state.get("llm_attempts", 0) <= 3, state.get("llm_attempts"))], t0)


def pre_guardrails(state):
    t0 = time.perf_counter()
    return _g("apply_guardrails", "pre", [c("strategy_present", bool(state.get("strategy")))], t0)


def post_guardrails(state):
    t0 = time.perf_counter(); g = state.get("guardrail", {})
    return _g("apply_guardrails", "post", [
        c("verdict_present", "approved" in g),
        c("final_offer_policy_approved_or_retrying", g.get("approved") is True or bool(g.get("retry")), state.get("final_offer_id")),
        c("retries_bounded_le_2", state.get("guardrail_retries", 0) <= 2, state.get("guardrail_retries"))], t0)


def pre_dispatch(state):
    t0 = time.perf_counter()
    return _g("compose_and_dispatch", "pre", [c("final_offer_set", bool(state.get("final_offer_id"))), c("guardrail_ok", state.get("guardrail", {}).get("approved") is True)], t0)


def post_dispatch(state):
    t0 = time.perf_counter(); d = state.get("dispatch", {})
    holdout_ok = (not state.get("is_holdout")) or d.get("status") != "sent"   # holdout may be suppressed or no_action, never sent
    return _g("compose_and_dispatch", "post", [
        c("dispatch_recorded", "status" in d, d.get("status")),
        c("ghost_holdout_never_messaged", holdout_ok, {"holdout": state.get("is_holdout"), "status": d.get("status")}),
        c("gated_link_signed", d.get("link_valid", True), d.get("link"))], t0)


def pre_log(state):
    t0 = time.perf_counter()
    return _g("log_and_attribute", "pre", [c("run_id_present", bool(state.get("run_id")))], t0)


def post_log(state):
    t0 = time.perf_counter(); a = state.get("attribution", {})
    return _g("log_and_attribute", "post", [c("attribution_written", bool(a)), c("attribution_has_treatment_flag", "treatment" in a)], t0)


def write_step_gates(run_id: str, gates: list[dict]) -> None:
    from agent.tools import jsonl_append
    import json
    jsonl_append("step_gates.jsonl", {"run_id": run_id, "gates": gates})
    p = get_settings().path("validation", "step_gates.json")
    cur = json.loads(p.read_text()) if p.exists() else {"runs": 0, "nodes": {n: {"pass": 0, "fail": 0} for n in NODES}, "recent": []}
    cur["runs"] += 1
    for g in gates:
        k = "pass" if g["passed"] else "fail"; cur["nodes"][g["node"]][k] += 1
    cur["recent"] = (cur["recent"] + [{"run_id": run_id, "all_passed": all(g["passed"] for g in gates), "gates": gates}])[-25:]
    cur["all_passed_last_run"] = all(g["passed"] for g in gates)
    p.write_text(json.dumps(cur, indent=2, default=str))
