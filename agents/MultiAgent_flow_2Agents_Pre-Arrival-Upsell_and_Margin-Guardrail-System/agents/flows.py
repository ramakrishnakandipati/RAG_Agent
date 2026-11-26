"""End-to-end flow catalogue and per-stage / overall metrics.

Three flow types, each with >= 2 worked examples (scenario keys from agents.tools.SCENARIOS):
  STRAIGHT_THROUGH  : approved on the first draft                      (clean_approval, corporate_late_checkout)
  REJECT_REVISE     : auditor rejects, strategist repairs, approved    (below_floor_price, spammy_copy, last_room_upgrade)
  HITL_ESCALATION   : human sign-off needed / loop exhausted at R_max  (vip_duplicate_perk, unresolvable_escalation)

Stage metrics (per run)       S1 CRM context -> S2 GPU quant inference -> S3 Strategist draft -> S4 Auditor contracts -> S5 revision loop -> S6 outcome
Overall metrics (all runs)    first-pass rate, final-approval rate, HITL rate, mean revisions, violation histogram, latency percentiles, margin protected.
Math: first_pass = (1/N) sum 1[r_i = 0 and APPROVED];  mean_rev = (1/N) sum r_i;  margin_protected = sum_i (p_i^{draft0} - floor)^- (docs L8).
"""
from __future__ import annotations
import json
import time
from collections import Counter

import numpy as np

from .graph import build_graph, run_offer, resume_hitl
from .tools import SCENARIOS
from models.utils import path

FLOWS = {
    "STRAIGHT_THROUGH": {"title": "Straight-through approval", "scenarios": ["clean_approval", "corporate_late_checkout"],
                         "path": "Strategist → Auditor ✔ → Finalize"},
    "REJECT_REVISE": {"title": "Reject → revise → approve", "scenarios": ["below_floor_price", "spammy_copy", "last_room_upgrade"],
                      "path": "Strategist → Auditor ✘ → Strategist (directives) → Auditor ✔ → Finalize"},
    "HITL_ESCALATION": {"title": "Human-in-the-loop escalation", "scenarios": ["vip_duplicate_perk", "unresolvable_escalation"],
                        "path": "… → Auditor ESCALATE → HITL gate → Finalize"},
}
FLOW_OF = {s: f for f, v in FLOWS.items() for s in v["scenarios"]}


def stage_metrics(out: dict) -> dict:
    """Break one finished/paused run into the six stage records."""
    h = out["history"]; strat = [e for e in h if e["agent"] == "strategist"]; aud = [e for e in h if e["agent"] == "auditor"]
    g = out["guest"]; first = strat[0]; c0 = first.get("candidates", [])
    live = [c for c in c0 if not c["excluded"]]                                  # same rule as the strategist: margin must clear the shadow-price hurdle
    top = next((c for c in live if c["unit_margin"] > c["hurdle"]), live[0] if live else (c0[0] if c0 else {}))
    s1 = {"reservation": g["reservation_id"], "segment": g["segment"], "tier": g["loyalty_tier"], "channel": g["channel"], "room": g["room_type"],
          "lead_time_d": g["lead_time"], "tier_perks": len(g["existing_tier_perks"]), "contacts_7d": g["contacts_7d"]}
    s2 = {"candidates_ranked": len(c0), "top_sku": top.get("sku"), "top_score": top.get("score"), "top_cate": top.get("cate_dragon"), "top_p_accept": top.get("p_accept"),
          "top_unit_margin": top.get("unit_margin"), "top_hurdle": top.get("hurdle"), "gpu_latency_ms": first.get("gpu_latency_ms")}
    s3 = [{"rev": e["revision"], "sku": e["sku"], "price": e["price"], "bundle": e["bundle"], "channel": e["channel"], "ms": e["latency_ms"], "llm": e["llm_mode"]} for e in strat]
    s4 = [{"rev": e["revision"], "decision": e["decision"], "violations": e["violations"], "tone": e["tone_score"], "ms": e["latency_ms"],
           "checks_pass": sum(bool(v) for v in e.get("checks", {}).values()), "checks_total": len(e.get("checks", {}))} for e in aud]
    s5 = {"drafts": len(strat), "revisions": out.get("revision", 0), "rejections": sum(e["decision"] == "REJECTED" for e in aud),
          "violations_fixed": sum(len(e["violations"]) for e in aud if e["decision"] == "REJECTED")}
    s6 = {"final_status": out.get("final_status"), "final_sku": out["draft"]["sku"], "final_price": out["draft"]["offer_price"],
          "hitl_reason": out["verdict"].get("hitl_reason"), "vip_value": out["verdict"].get("vip_value"), "unit_margin": out["draft"]["quant"].get("unit_margin")}
    s6["contribution_margin_pct"] = round(100 * (out["draft"]["quant"].get("unit_margin", 0)) / max(out["draft"]["offer_price"], 1e-9), 1)
    tot = round(sum(e["latency_ms"] for e in h if "latency_ms" in e), 1)
    return {"S1_context": s1, "S2_quant": s2, "S3_draft": s3, "S4_audit": s4, "S5_loop": s5, "S6_outcome": s6, "total_latency_ms": tot}


def run_scenario(key: str, graph=None, auto_hitl: bool | None = None) -> dict:
    sc = SCENARIOS[key]; g = graph or build_graph()
    out = run_offer(sc["guest"].model_dump(), sc["injections"], sc["pms_override"], graph=g)
    m = stage_metrics(out); m.update(key=key, flow=FLOW_OF.get(key, "OTHER"), title=sc["title"])
    return {"out": out, "metrics": m}


def run_all(save: bool = True) -> dict:
    """Run every scenario (isolated in-memory graph; HITL threads are left pending), aggregate overall metrics."""
    from .tools import get_engine
    get_engine(); g = build_graph()
    run_scenario("clean_approval", g)      # warm-up so latencies exclude one-off GPU/model load
    runs = []
    for k in SCENARIOS:
        t0 = time.perf_counter(); r = run_scenario(k, g); r["metrics"]["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1); runs.append(r["metrics"])
    n = len(runs); viol = Counter(v for r in runs for a in r["S4_audit"] for v in a["violations"])
    lat = np.array([r["total_latency_ms"] for r in runs])
    first_pass = sum(r["S5_loop"]["revisions"] == 0 and r["S6_outcome"]["final_status"] == "APPROVED" for r in runs)
    approved = sum(r["S6_outcome"]["final_status"] == "APPROVED" for r in runs); hitl = sum(r["S6_outcome"]["final_status"].startswith("HITL") for r in runs)
    first_dec = Counter(r["S4_audit"][0]["decision"] for r in runs)
    # revenue/margin protected: first drafts the auditor would have let through unchecked, valued at unit margin of the draft
    unsafe = [r for r in runs if r["S4_audit"][0]["decision"] == "REJECTED"]
    overall = {"n_runs": n, "first_pass_approval_rate": first_pass / n, "final_auto_approval_rate": approved / n, "hitl_rate": hitl / n,
               "mean_revisions": float(np.mean([r["S5_loop"]["revisions"] for r in runs])), "max_revisions_seen": max(r["S5_loop"]["revisions"] for r in runs),
               "first_draft_decisions": dict(first_dec), "first_draft_rejection_rate": len(unsafe) / n,
               "violations_by_code": dict(viol), "violations_total": int(sum(viol.values())),
               "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
               "mean_gpu_quant_latency_ms": float(np.mean([r["S2_quant"]["gpu_latency_ms"] or 0 for r in runs])),
               "mean_final_price": float(np.mean([r["S6_outcome"]["final_price"] for r in runs])),
               "mean_contribution_margin_pct": float(np.mean([r["S6_outcome"]["contribution_margin_pct"] for r in runs])),
               "llm_mode": runs[0]["S3_draft"][0]["llm"]}
    by_flow = {}
    for f in FLOWS:
        rs = [r for r in runs if r["flow"] == f]
        by_flow[f] = {"n": len(rs), "mean_revisions": float(np.mean([r["S5_loop"]["revisions"] for r in rs])), "mean_latency_ms": float(np.mean([r["total_latency_ms"] for r in rs])),
                      "statuses": dict(Counter(r["S6_outcome"]["final_status"] for r in rs)), "examples": [r["key"] for r in rs]}
    rep = {"overall": overall, "by_flow": by_flow, "runs": runs}
    if save:
        (path("metrics") / "flow_metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    r = run_all(); print(json.dumps(r["overall"], indent=2)); print(json.dumps(r["by_flow"], indent=2))
