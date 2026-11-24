"""End-to-end offline replay over N guest journeys -> metrics/, evaluation/, then charts + README update."""
from __future__ import annotations
import argparse
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from data.synthetic_generator import build_feature_row, sigmoid, true_accept_logit
from src.config import get_settings
from src.evaluation import judge, model_eval
from src.graph import edges
from src.graph.workflow import resume_workflow, run_workflow
from src.tools import environment_tools as envt
from src.tools import nba_rms_engine as nba
from src.tools import pms_crm_tools as pms
from src.utils import device as dev
from src.utils.logger import get_logger

log = get_logger("replay")
BASELINE_SKU = "SPA-RELAX"


def _sample_reservations(n: int, rng: np.random.Generator) -> list[dict]:
    pool = pms.list_upcoming_reservations(limit=5000, min_days=1, max_days=14)
    df = pd.DataFrame(pool)
    plat = df[df.loyalty_tier == "Platinum"]
    take_plat = plat.sample(min(len(plat), max(n // 8, 1)), random_state=int(rng.integers(1e6)))
    rest = df.drop(take_plat.index).sample(min(len(df) - len(take_plat), n - len(take_plat)), random_state=int(rng.integers(1e6)))
    return pd.concat([take_plat, rest]).sample(frac=1, random_state=1).head(n).to_dict("records")


def _resolve_hitl(res: dict) -> dict:
    guard = 0
    while res["interrupted"] and guard < 3:
        res = resume_workflow(res["thread_id"], {"decision": "approve", "reviewer": "sim_concierge"})
        guard += 1
    return res


def _accept_prob(guest, stay, env, offer, fx) -> float:
    cat = next(s for s in nba.load_catalog() if s["sku"] == offer["sku"])
    row = build_feature_row(guest, stay, cat, {"precip_prob": env["precip_prob"], "temp_c": env["temp_c"]},
                            env["has_event"], env["occupancy_pct"], offer["discount_pct"])
    return sigmoid(true_accept_logit(row, fx))


def run_replay(n: int = 120, seed: int = 42, llm_judge: bool = True, adv_once: float = 0.10, adv_persist: float = 0.04,
               progress: bool = True) -> dict:
    s = get_settings()
    rng = np.random.default_rng(seed)
    pms.init_pms()
    pms.reset_runtime_tables()
    (s.logs_dir / 'telemetry.jsonl').unlink(missing_ok=True)
    fx = json.loads((s.raw_dir / "guest_effects.json").read_text())
    res_rows = _sample_reservations(n, rng)
    cat = {c["sku"]: c for c in nba.load_catalog()}
    journeys, lat = [], defaultdict(list)
    t_start = time.perf_counter()
    for i, r in enumerate(res_rows):
        mode = "persistent" if rng.random() < adv_persist else ("once" if rng.random() < adv_once else "none")
        t0 = time.perf_counter()
        try:
            res = _resolve_hitl(run_workflow(r["reservation_id"], adversarial=mode))
        except Exception as e:  # noqa: BLE001
            log.error("journey_crash", reservation=r["reservation_id"], error=str(e))
            journeys.append({"reservation_id": r["reservation_id"], "outcome": "CRASH", "error": str(e)})
            continue
        st = res["state"]
        for t in st.get("trace", []):
            lat[t["node"]].append(t["latency_ms"])
        guest, stay, env = st.get("guest"), st.get("stay"), st.get("env")
        offer = st.get("selected_offer")
        j = {"reservation_id": r["reservation_id"], "guest_id": r["guest_id"], "tier": r["loyalty_tier"], "segment": r["segment"],
             "channel_group": r["channel_group"], "adversarial": mode, "status": st.get("status"), "route": edges.classify_route(st),
             "retries": st.get("retries", 0), "hitl": bool(st.get("hitl_required")), "hitl_reasons": st.get("hitl_reasons", []),
             "guardrail": (st.get("guardrail") or {}).get("status"), "guardrail_violations": (st.get("guardrail") or {}).get("violations", []), "violations_seen": (st.get("feedback") or []),
             "source": (st.get("draft") or {}).get("source"), "sku": offer["sku"] if offer else None,
             "nights": stay["nights"] if stay else 0, "journey_s": round(time.perf_counter() - t0, 2),
             "p_accept_model": offer["propensity"] if offer else None, "sent": bool(st.get("dispatched")),
             "booked_amount": 0.0, "est_commission_avoided": 0.0, "exp_revenue": 0.0, "exp_margin": 0.0, "base_exp_margin": 0.0, "judge": None, "inbound": None}
        if offer and guest:
            p_true = _accept_prob(guest, stay, env, offer, fx.get(guest["guest_id"], 0.0))
            j["p_accept_true"] = round(p_true, 4)
            # --- baseline: static spa offer at list price to every opted-in guest (same behaviour model)
            base_offer = {"sku": BASELINE_SKU, "discount_pct": 0.0}
            pb = _accept_prob(guest, stay, env, base_offer, fx.get(guest["guest_id"], 0.0)) if guest["opt_in_marketing"] else 0.0
            j["base_p"], j["base_exp_revenue"] = round(pb, 4), round(pb * cat[BASELINE_SKU]["price"], 2)
            j["base_exp_margin"] = round(pb * (cat[BASELINE_SKU]["price"] - cat[BASELINE_SKU]["cost"]), 2)
            j["base_realized"] = cat[BASELINE_SKU]["price"] if rng.random() < pb else 0.0
        else:
            j["base_p"] = j["base_exp_revenue"] = j["base_realized"] = 0.0
        if j["sent"] and st.get("draft"):
            j["judge"] = judge.judge_message(f"{st['draft']['subject']}\n{st['draft']['body']}", offer, use_llm=llm_judge)
            if offer["category"] == "loyalty":
                j["est_commission_avoided"] = round(p_true * 0.18 * stay["adr"] * stay["nights"], 2)
            else:
                j["exp_revenue"] = round(p_true * offer["final_price"], 2)
                j["exp_margin"] = round(p_true * (offer["final_price"] - offer["cost"]), 2)
            # --- two-way: guest reply
            if rng.random() < p_true:
                rep = _resolve_hitl(run_workflow(r["reservation_id"], trigger_type="inbound_reply", inbound_text="Yes please, add it to my booking!"))
                rs = rep["state"]
                j["inbound"] = {"kind": "accept", "status": rs.get("status"), "booking": rs.get("booking")}
                if rs.get("status") == "BOOKED" and rs.get("booking", {}).get("ok"):
                    j["booked_amount"] = float(rs["booking"]["amount"])
                    if offer["category"] == "loyalty":
                        j["est_commission_avoided"] = round(0.18 * stay["adr"] * stay["nights"], 2)
            elif rng.random() < 0.06:
                rep = _resolve_hitl(run_workflow(r["reservation_id"], trigger_type="inbound_reply", inbound_text="This is terrible, I want a refund."))
                j["inbound"] = {"kind": "complaint", "status": rep["state"].get("status"), "hitl": bool(rep["state"].get("hitl_required")),
                                "reasons": rep["state"].get("hitl_reasons", [])}
            elif r["loyalty_tier"] == "Platinum" and rng.random() < 0.4:
                rep = _resolve_hitl(run_workflow(r["reservation_id"], trigger_type="inbound_reply",
                                                 inbound_text="Could you arrange a private surprise anniversary dinner on the beach?"))
                j["inbound"] = {"kind": "vip_custom", "status": rep["state"].get("status"), "hitl": bool(rep["state"].get("hitl_required")),
                                "reasons": rep["state"].get("hitl_reasons", [])}
        journeys.append(j)
        if progress and (i + 1) % 10 == 0:
            print(f"  replay {i + 1}/{len(res_rows)}  ({time.perf_counter() - t_start:.0f}s)", flush=True)

    metrics = aggregate(journeys, lat, time.perf_counter() - t_start, n, seed)
    mm = model_eval.evaluate()
    metrics["model"] = {k: v for k, v in mm.items() if k not in ("curves",)}
    s.metrics_dir.mkdir(exist_ok=True)
    (s.metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    with open(s.metrics_dir / "journeys.jsonl", "w", encoding="utf-8") as f:
        for j in journeys:
            f.write(json.dumps(j, default=str) + "\n")
    pd.DataFrame([{"node": k, "latency_ms": v} for k, vs in lat.items() for v in vs]).to_csv(s.metrics_dir / "node_latency.csv", index=False)
    (s.eval_dir / "judge_results.json").write_text(json.dumps(
        [{"reservation_id": j["reservation_id"], **j["judge"]} for j in journeys if j.get("judge")], indent=2, default=str))
    log.info("replay_done", **{k: metrics[k] for k in ("n_journeys", "hallucination_rate_pct")})
    return metrics


def aggregate(journeys: list[dict], lat: dict, wall_s: float, n_req: int, seed: int) -> dict:
    ok = [j for j in journeys if j.get("outcome") != "CRASH"]
    sent = [j for j in ok if j["sent"]]
    judged = [j["judge"] for j in sent if j.get("judge")]
    halluc = sum(1 for jd in judged if not jd["hallucination_free"])
    funnel = Counter()
    for j in ok:
        if j["status"] == "NO_OFFER": funnel["no_offer"] += 1
        elif j["status"] == "FAIL_VALIDATION":
            hard = ("guest_not_opted_in", "frequency_cap_exceeded", "quiet_hours", "inventory_unavailable")
            vs = [x.split(":")[0] for x in (j.get("guardrail_violations") or [])]
            v = next((x for x in vs if x in hard), vs[0] if vs else "other")
            funnel["hard_stop_" + v] += 1
        elif j["hitl"]: funnel["escalated_hitl"] += 1
        elif j["retries"] > 0: funnel["retried_then_passed"] += 1
        else: funnel["first_pass"] += 1
    adv = [j for j in ok if j["adversarial"] != "none"]
    caught = [j for j in adv if j["retries"] > 0 or j["hitl"] or j["status"] == "FAIL_VALIDATION"]
    poisoned_sent = [j for j in sent if j["source"] == "adversarial"]
    hitl_reasons = Counter(r for j in ok for r in j["hitl_reasons"])
    for j in ok:
        if j.get("inbound") and j["inbound"].get("hitl"):
            hitl_reasons.update(j["inbound"].get("reasons", []))
    nights = sum(j["nights"] for j in ok)
    agent_real = sum(j["booked_amount"] for j in ok)
    agent_exp = sum(j["exp_revenue"] for j in ok)
    base_real = sum(j["base_realized"] for j in ok)
    base_exp = sum(j["base_exp_revenue"] for j in ok)
    lat_s = {k: {"mean_ms": round(float(np.mean(v)), 1), "p50_ms": round(float(np.percentile(v, 50)), 1),
                 "p95_ms": round(float(np.percentile(v, 95)), 1), "n": len(v)} for k, v in lat.items()}
    inbound = [j["inbound"] for j in ok if j.get("inbound")]
    tones = [jd["tone_score"] for jd in judged]
    cats = Counter(j["sku"] for j in sent)
    return {
        "n_requested": n_req, "n_journeys": len(ok), "n_crashed": len(journeys) - len(ok), "seed": seed, "wall_seconds": round(wall_s, 1),
        "gpu": dev.gpu_info(), "llm_sources": dict(Counter(j["source"] for j in sent)),
        "messages_sent": len(sent), "messages_judged": len(judged),
        "hallucinated_messages": halluc, "hallucination_rate_pct": round(100 * halluc / max(len(judged), 1), 3),
        "adversarial_injected": len(adv), "adversarial_caught": len(caught), "adversarial_sent_to_guest": len(poisoned_sent),
        "adversarial_catch_rate_pct": round(100 * len(caught) / max(len(adv), 1), 1) if adv else None,
        "funnel": dict(funnel), "hitl_reasons": dict(hitl_reasons),
        "inbound_replies": len(inbound), "inbound_accepts": sum(1 for x in inbound if x["kind"] == "accept"),
        "inbound_booked": sum(1 for x in inbound if x["kind"] == "accept" and x["status"] == "BOOKED"),
        "inbound_sold_out": sum(1 for x in inbound if x["status"] == "SOLD_OUT"),
        "avg_tone_score": round(float(np.mean(tones)), 3) if tones else None, "tone_source": dict(Counter(jd["tone_source"] for jd in judged)),
        "offers_by_sku": dict(cats), "loyalty_offers_sent": cats.get("LOY-DIRECT", 0),
        "node_latency": lat_s, "avg_journey_s": round(float(np.mean([j["journey_s"] for j in ok])), 2),
        "revenue": {
            "room_nights": nights, "agent_realized_usd": round(agent_real, 2), "baseline_realized_usd": round(base_real, 2),
            "agent_expected_usd": round(agent_exp, 2), "baseline_expected_usd": round(base_exp, 2),
            "agent_anc_revpar": round(agent_exp / max(nights, 1), 3), "baseline_anc_revpar": round(base_exp / max(nights, 1), 3),
            "anc_revpar_lift_usd": round((agent_exp - base_exp) / max(nights, 1), 3),
            "anc_revpar_lift_pct": round(100 * (agent_exp - base_exp) / max(base_exp, 1e-9), 1),
            "realized_lift_pct": round(100 * (agent_real - base_real) / max(base_real, 1e-9), 1) if base_real else None,
            "agent_exp_margin_usd": round(sum(j["exp_margin"] for j in ok), 2), "baseline_exp_margin_usd": round(sum(j["base_exp_margin"] for j in ok), 2),
            "margin_lift_pct": round(100 * (sum(j["exp_margin"] for j in ok) - sum(j["base_exp_margin"] for j in ok)) / max(sum(j["base_exp_margin"] for j in ok), 1e-9), 1),
            "est_ota_commission_avoided_usd": round(sum(j["est_commission_avoided"] for j in ok), 2),
            "basis": "expected value = P(accept | hidden behaviour model) x final price; realized = Bernoulli draws; baseline = static SPA-RELAX at list price to every opted-in guest",
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm-judge", action="store_true")
    ap.add_argument("--no-readme", action="store_true")
    a = ap.parse_args()
    m = run_replay(a.n, a.seed, llm_judge=not a.no_llm_judge)
    if not a.no_readme:
        from src.visualization.generate_charts import generate_all_charts
        from src.utils.readme_updater import update_readme
        generate_all_charts()
        update_readme()
    print(json.dumps({k: m[k] for k in ("n_journeys", "hallucination_rate_pct", "funnel", "adversarial_catch_rate_pct", "revenue")}, indent=2))
