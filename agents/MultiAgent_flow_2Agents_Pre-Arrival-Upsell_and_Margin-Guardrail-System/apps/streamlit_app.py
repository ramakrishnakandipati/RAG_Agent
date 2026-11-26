"""Executive & Simulation dashboard (Streamlit). Run:  streamlit run apps/streamlit_app.py"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agents.graph import resume_hitl, run_offer  # noqa: E402
from agents.llm_client import get_llm  # noqa: E402
from agents.tools import PMS, SCENARIOS, get_engine  # noqa: E402
from models.utils import gpu_snapshot, load_config, path  # noqa: E402

st.set_page_config(page_title="Upsell Guardrail MAS", layout="wide", page_icon="🛎️")
CFG = load_config()


def _json(name, default=None):
    p = path("metrics") / name
    return json.loads(p.read_text()) if p.exists() else default


@st.cache_resource(show_spinner="Loading GPU models...")
def _engine():
    return get_engine()


def tag(status: str) -> str:
    colour = {"APPROVED": "green", "HITL_APPROVED": "green", "REJECTED": "red", "HITL_REJECTED": "red", "ESCALATE_HITL": "orange", "HITL_PENDING": "orange"}.get(status, "gray")
    return f":{colour}[**{status}**]"


st.title("Pre-Arrival Upsell & Margin Guardrail - 2-Agent System")
llm = get_llm(); g = gpu_snapshot()
st.caption(f"GPU: **{g.get('device')}** · VRAM peak {g.get('vram_peak_mib', 0)} MiB · LLM: **{llm.mode}** (`{CFG['llm']['model']}` @ {CFG['llm']['base_url']}) · R_max = {CFG['agents']['max_revisions']}")
t_exec, t_sim, t_flow, t_quant, t_ope, t_hitl = st.tabs(["Executive KPIs", "End-to-End Flow", "Flow Metrics", "Quant Models", "Offline Policy Evaluation", "HITL Queue"])

# ------------------------------------------------------------------ executive
with t_exec:
    ope, qs = _json("ope_report.json"), _json("quant_suite_report.json")
    if not ope or not qs:
        st.warning("Run `python models/train_quant_suite.py` and `python evaluation/offline_replay.py` first.")
    else:
        pol = {p["policy"]: p for p in ope["policies"]}
        a, b, c, d = st.columns(4)
        a.metric("Logged policy value ($/guest)", f"{pol['logged_behaviour']['true_value']:.2f}")
        b.metric("ML + guardrails ($/guest)", f"{pol['ml+agent_guardrails']['true_value']:.2f}", f"{ope['improvement_vs_logged_pct']['ml+agent_guardrails']:+.0f}% vs logged")
        c.metric("Guardrail violations (ML-only → guarded)", f"{pol['ml_only(elasticity greedy)']['guardrail_violation_rate']:.1%} → {pol['ml+agent_guardrails']['guardrail_violation_rate']:.1%}")
        d.metric("Elasticity ECE / monotonic violations", f"{qs['elasticity']['ece']:.4f} / {qs['elasticity']['monotonicity_violations']}")
        e, f, h, i = st.columns(4)
        e.metric("Elasticity RMSE vs truth", f"{qs['elasticity']['rmse_vs_true_prob']:.3f}")
        f.metric("DragonNet √PEHE (avg)", f"{np.mean([v['sqrt_pehe']['dragonnet'] for v in qs['uplift'].values()]):.3f}")
        h.metric("LinUCB regret vs random", f"{qs['bandit']['linucb']['final_cumulative_regret'] / qs['bandit']['random']['final_cumulative_regret']:.0%}")
        i.metric("Allocator KKT violation", f"{qs['allocator']['primal_violation_max']:.1e}")
        st.subheader("Policy comparison")
        st.dataframe(pd.DataFrame(ope["policies"])[["policy", "true_value", "ips", "snips", "dr", "ess", "regret", "guardrail_violation_rate", "cannibalisation_rate"]].round(3), use_container_width=True)
        runs = path("logs") / "offer_runs.jsonl"
        if runs.exists():
            df = pd.DataFrame([json.loads(x) for x in runs.read_text().splitlines() if x.strip()]); df = df[df.get("status").notna()] if "status" in df else df
            st.subheader("Agent outcomes (all logged runs)"); st.bar_chart(df["status"].value_counts())

# ------------------------------------------------------------------ end-to-end flow
STAGES = ["S1 Guest context (CRM/PMS/RMS)", "S2 GPU quant inference", "S3 Strategist draft", "S4 Auditor contracts", "S5 Revision loop", "S6 Outcome"]


def _dot(m):
    s6 = m["S6_outcome"]["final_status"]; col = {"APPROVED": "#2e7d4f", "HITL_PENDING": "#c98a00", "HITL_APPROVED": "#2e7d4f", "HITL_REJECTED": "#b3402f"}.get(s6, "#555")
    loop = f'A -> S [label="REJECTED x{m["S5_loop"]["rejections"]}", color="#b3402f", fontcolor="#b3402f"];' if m["S5_loop"]["rejections"] else ""
    hitl = 'A -> H [label="ESCALATE", color="#c98a00"]; H -> F;' if s6.startswith("HITL") else ""
    ok = 'A -> F [label="APPROVED", color="#2e7d4f"];' if s6 == "APPROVED" else ""
    return ('digraph{rankdir=LR;node[shape=box,style="rounded,filled",fillcolor="#f6f4ef"];'
            f'C[label="CRM / PMS / RMS\\n{m["S1_context"]["tier"]} - {m["S1_context"]["segment"]}"];'
            f'Q[label="GPU quant\\n{m["S2_quant"]["gpu_latency_ms"]:.0f} ms"];S[label="Agent 1 Strategist\\n{m["S5_loop"]["drafts"]} draft(s)"];'
            f'A[label="Agent 2 Auditor\\n{m["S5_loop"]["rejections"]} rejection(s)"];H[label="HITL gate"];F[label="{s6}",fillcolor="{col}",fontcolor=white];'
            f'C->Q->S->A;{loop}{ok}{hitl}}}')


with t_sim:
    from agents.flows import FLOWS, FLOW_OF, stage_metrics
    from agents.graph import run_offer
    st.markdown("#### End-to-end flow: CRM → GPU quant → Strategist ⇄ Auditor → HITL / Finalize")
    flow = st.radio("Flow type", list(FLOWS), format_func=lambda f: f"{FLOWS[f]['title']}  ·  {FLOWS[f]['path']}")
    ex = st.selectbox("Example", FLOWS[flow]["scenarios"], format_func=lambda k: SCENARIOS[k]["title"])
    sc = SCENARIOS[ex]; st.info(sc["description"])
    c1, c2 = st.columns([1, 3])
    with c1:
        with st.expander("Guest profile", expanded=False):
            st.json(sc["guest"].model_dump())
        jv = st.slider("Live PMS: JUNIOR_OCEAN_VIEW available", 0, 30, int(sc["pms_override"].get("JUNIOR_OCEAN_VIEW", PMS.available["JUNIOR_OCEAN_VIEW"])))
        st.caption(f"Dynamic safety buffer = {PMS.dynamic_safety_buffer('JUNIOR_OCEAN_VIEW')} rooms")
        go = st.button("Run end-to-end flow", type="primary")
    if go:
        _engine()
        with st.spinner("Running LangGraph loop on GPU..."):
            out = run_offer(sc["guest"].model_dump(), sc["injections"], {**sc["pms_override"], "JUNIOR_OCEAN_VIEW": jv})
            m = stage_metrics(out); m.update(key=ex, flow=FLOW_OF[ex], title=sc["title"])
        st.session_state["last"] = {"out": out, "m": m}
        if out["final_status"] == "HITL_PENDING":
            q = path("logs") / "hitl_queue.json"; qd = json.loads(q.read_text()) if q.exists() else {}
            qd[out["thread_id"]] = {"thread_id": out["thread_id"], "reservation_id": sc["guest"].reservation_id, "reason": (out.get("interrupt") or {}).get("reason"),
                                    "draft": out["draft"], "status": "PENDING"}; q.write_text(json.dumps(qd, indent=2))
    last = st.session_state.get("last")
    if last and last["m"]["key"] == ex:
        out, m = last["out"], last["m"]
        with c2:
            st.markdown(f"### Result: {tag(out['final_status'])} · {out.get('revision', 0)} revision(s) · {m['total_latency_ms']} ms total")
            st.graphviz_chart(_dot(m))
        s1, s2, s5, s6 = m["S1_context"], m["S2_quant"], m["S5_loop"], m["S6_outcome"]
        with st.container(border=True):
            st.markdown(f"**{STAGES[0]}**"); a, b, c, d = st.columns(4)
            a.metric("Segment / tier", f"{s1['segment']} / {s1['tier']}"); b.metric("Channel · room", f"{s1['channel']} · {s1['room']}"); c.metric("Lead time", f"T-{s1['lead_time_d']}"); d.metric("Free tier perks", s1["tier_perks"])
        with st.container(border=True):
            st.markdown(f"**{STAGES[1]}** - elasticity net + DragonNet/X-Learner + LinUCB bonus − shadow-price hurdle")
            a, b, c, d, e = st.columns(5); a.metric("GPU latency", f"{s2['gpu_latency_ms']:.0f} ms"); b.metric("Top SKU", s2["top_sku"]); c.metric("CATE τ̂", f"{s2['top_cate']:+.3f}")
            d.metric("P(accept)", f"{s2['top_p_accept']:.2f}"); e.metric("Margin − hurdle", f"${s2['top_unit_margin'] - s2['top_hurdle']:.0f}")
            st.dataframe(pd.DataFrame(out["history"][0]["candidates"]).round(3), use_container_width=True, hide_index=True)
        with st.container(border=True):
            st.markdown(f"**{STAGES[2]}** per revision")
            st.dataframe(pd.DataFrame(m["S3_draft"]), use_container_width=True, hide_index=True)
            d = out["draft"]; st.text(f"Subject: {d['subject']}\n\n{d['body_copy']}"); st.caption(d["quant_rationale"])
        with st.container(border=True):
            st.markdown(f"**{STAGES[3]}** per revision")
            st.dataframe(pd.DataFrame(m["S4_audit"]), use_container_width=True, hide_index=True)
            chk = out["verdict"].get("checks", {}); st.caption("Final-verdict contract checks: " + " · ".join(f"{'✅' if v else '❌'} {k}" for k, v in chk.items()))
            if out["verdict"]["violations"]:
                st.table(pd.DataFrame([{"code": x["code"], "detail": x["detail"], "remediation": x["remediation"]} for x in out["verdict"]["violations"]]))
        with st.container(border=True):
            st.markdown(f"**{STAGES[4]}**"); a, b, c, d = st.columns(4)
            a.metric("Drafts produced", s5["drafts"], help="at most R_max + 1"); b.metric("Revisions", f"{s5['revisions']} / {CFG['agents']['max_revisions']}")
            c.metric("Rejections", s5["rejections"]); d.metric("Violations repaired", s5["violations_fixed"])
        with st.container(border=True):
            st.markdown(f"**{STAGES[5]}** {tag(s6['final_status'])}"); a, b, c, d = st.columns(4)
            a.metric("Final offer", f"{s6['final_sku']} ${s6['final_price']:.0f}"); b.metric("Contribution margin", f"{s6['contribution_margin_pct']}%", help="floor 30%")
            c.metric("VIP value (w·p)", f"${s6['vip_value']:.0f}", help="HITL at >= $250"); d.metric("Pipeline latency", f"{m['total_latency_ms']} ms")
            if s6["hitl_reason"]:
                st.warning(f"HITL: {s6['hitl_reason']}")

# ------------------------------------------------------------------ overall flow metrics
with t_flow:
    from agents.flows import FLOWS, run_all
    fm_path = path("metrics") / "flow_metrics.json"
    if st.button("Re-run all flows (GPU)") or not fm_path.exists():
        with st.spinner("Running all flows..."):
            run_all()
    fm = json.loads(fm_path.read_text()); ov = fm["overall"]
    st.markdown("#### Overall metrics across all end-to-end flows")
    a, b, c, d, e = st.columns(5)
    a.metric("Runs", ov["n_runs"]); b.metric("First-pass approval", f"{ov['first_pass_approval_rate']:.0%}"); c.metric("Auto-approved (final)", f"{ov['final_auto_approval_rate']:.0%}")
    d.metric("HITL escalation", f"{ov['hitl_rate']:.0%}"); e.metric("Mean revisions", f"{ov['mean_revisions']:.2f}")
    a, b, c, d, e = st.columns(5)
    a.metric("First drafts rejected", f"{ov['first_draft_rejection_rate']:.0%}"); b.metric("Violations caught", ov["violations_total"])
    c.metric("Latency p50 / p95", f"{ov['latency_ms']['p50']:.0f} / {ov['latency_ms']['p95']:.0f} ms")
    d.metric("GPU quant latency", f"{ov['mean_gpu_quant_latency_ms']:.0f} ms"); e.metric("Mean contribution margin", f"{ov['mean_contribution_margin_pct']:.0f}%")
    l, r = st.columns(2)
    with l:
        st.markdown("**Violations caught by the Auditor (by code)**"); st.bar_chart(pd.Series(ov["violations_by_code"]))
    with r:
        st.markdown("**Mean latency by flow (ms)**"); st.bar_chart(pd.Series({FLOWS[k]["title"]: v["mean_latency_ms"] for k, v in fm["by_flow"].items()}))
    st.markdown("**By flow type**")
    st.dataframe(pd.DataFrame([{"flow": FLOWS[k]["title"], "examples": ", ".join(v["examples"]), "n": v["n"], "mean revisions": round(v["mean_revisions"], 2),
                                "mean latency ms": round(v["mean_latency_ms"], 1), "statuses": json.dumps(v["statuses"])} for k, v in fm["by_flow"].items()]), use_container_width=True, hide_index=True)
    st.markdown("**Per-example stage metrics (S1 context → S6 outcome)**")
    rows = [{"example": r["key"], "flow": r["flow"], "tier": r["S1_context"]["tier"], "top SKU (S2)": r["S2_quant"]["top_sku"], "CATE": r["S2_quant"]["top_cate"], "quant ms": r["S2_quant"]["gpu_latency_ms"],
             "drafts (S3)": r["S5_loop"]["drafts"], "first audit (S4)": r["S4_audit"][0]["decision"], "violations": ",".join(r["S4_audit"][0]["violations"]) or "-",
             "revisions (S5)": r["S5_loop"]["revisions"], "outcome (S6)": r["S6_outcome"]["final_status"], "price": r["S6_outcome"]["final_price"],
             "CM %": r["S6_outcome"]["contribution_margin_pct"], "total ms": r["total_latency_ms"]} for r in fm["runs"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("**System-level context (models & policy evaluation)**")
    qs, ope = _json("quant_suite_report.json"), _json("ope_report.json")
    if qs and ope:
        pol = {p["policy"]: p for p in ope["policies"]}
        a, b, c, d = st.columns(4)
        a.metric("Policy value: logged → ML+guardrails", f"${pol['logged_behaviour']['true_value']:.1f} → ${pol['ml+agent_guardrails']['true_value']:.1f}")
        b.metric("Guardrail violations: ML-only → guarded", f"{pol['ml_only(elasticity greedy)']['guardrail_violation_rate']:.0%} → {pol['ml+agent_guardrails']['guardrail_violation_rate']:.0%}")
        c.metric("DR estimate (guarded)", f"${pol['ml+agent_guardrails']['dr']:.1f}"); d.metric("Elasticity ECE / RMSE", f"{qs['elasticity']['ece']:.3f} / {qs['elasticity']['rmse_vs_true_prob']:.3f}")

# ------------------------------------------------------------------ quant
with t_quant:
    qs = _json("quant_suite_report.json")
    if qs:
        st.image(str(path("plots") / "elasticity_diagnostics.png"), caption="Elasticity net: loss, calibration, monotone demand curves")
        st.image(str(path("plots") / "uplift_qini_curves.png"), caption="Uplift Qini curves (DragonNet vs X-Learner vs random)")
        l, r = st.columns(2); l.image(str(path("plots") / "bandit_regret.png")); r.image(str(path("plots") / "allocator_shadow_prices.png"))
        st.subheader("Uplift √PEHE per category"); st.dataframe(pd.DataFrame({k: v["sqrt_pehe"] for k, v in qs["uplift"].items()}).T.round(4))
        st.subheader("Shadow prices"); st.json(qs["allocator"])

# ------------------------------------------------------------------ ope
with t_ope:
    ope = _json("ope_report.json")
    if ope:
        st.image(str(path("plots") / "ope_policy_values.png")); st.dataframe(pd.DataFrame(ope["policies"]).drop(columns=["dr_ci95"]).round(3))
        if ope.get("uplift_qini"):
            st.subheader("Qini coefficient / AUUC (held-out)"); st.dataframe(pd.DataFrame({k: {m: v[m]["qini_coefficient"] for m in ("dragonnet", "x_learner", "random")} for k, v in ope["uplift_qini"].items()}).T.round(5))

# ------------------------------------------------------------------ hitl
with t_hitl:
    q = path("logs") / "hitl_queue.json"; qd = json.loads(q.read_text()) if q.exists() else {}
    pend = [v for v in qd.values() if v["status"] == "PENDING"]
    st.write(f"**{len(pend)}** pending review(s)")
    for v in pend:
        with st.container(border=True):
            st.markdown(f"**{v['reservation_id']}** · `{v['thread_id']}` · {v['draft']['sku']} ${v['draft']['offer_price']}"); st.caption(v.get("reason") or "")
            c1, c2, _ = st.columns([1, 1, 6])
            for lab, dec, col in (("Approve", "approve", c1), ("Reject", "reject", c2)):
                if col.button(lab, key=f"{dec}{v['thread_id']}"):
                    o = resume_hitl(v["thread_id"], dec, "streamlit-ops"); qd[v["thread_id"]].update(status="RESOLVED", final_status=o["final_status"]); q.write_text(json.dumps(qd, indent=2)); st.rerun()
