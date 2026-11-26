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
t_exec, t_sim, t_quant, t_ope, t_hitl = st.tabs(["Executive KPIs", "Scenario Simulator", "Quant Models", "Offline Policy Evaluation", "HITL Queue"])

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

# ------------------------------------------------------------------ simulator
with t_sim:
    names = list(SCENARIOS); sel = st.selectbox("Scenario", names, format_func=lambda k: SCENARIOS[k]["title"])
    sc = SCENARIOS[sel]; st.info(sc["description"])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.json(sc["guest"].model_dump(), expanded=False)
        jv = st.slider("Live PMS: JUNIOR_OCEAN_VIEW available", 0, 30, int(sc["pms_override"].get("JUNIOR_OCEAN_VIEW", PMS.available["JUNIOR_OCEAN_VIEW"])))
        st.caption(f"Dynamic safety buffer = {PMS.dynamic_safety_buffer('JUNIOR_OCEAN_VIEW')} rooms")
        go = st.button("Run Strategist ⇄ Auditor loop", type="primary")
    if go:
        _engine()
        with st.spinner("Running LangGraph loop on GPU..."):
            out = run_offer(sc["guest"].model_dump(), sc["injections"], {**sc["pms_override"], "JUNIOR_OCEAN_VIEW": jv})
        st.session_state["last"] = out
        if out["final_status"] == "HITL_PENDING":
            q = path("logs") / "hitl_queue.json"; qd = json.loads(q.read_text()) if q.exists() else {}
            qd[out["thread_id"]] = {"thread_id": out["thread_id"], "reservation_id": sc["guest"].reservation_id, "reason": (out.get("interrupt") or {}).get("reason"),
                                    "draft": out["draft"], "status": "PENDING"}; q.write_text(json.dumps(qd, indent=2))
    out = st.session_state.get("last")
    if out:
        with c2:
            st.markdown(f"### Result: {tag(out['final_status'])} after {out.get('revision', 0)} revision(s)")
            rows = []
            for h in out["history"]:
                rows.append({"agent": h["agent"], "rev": h.get("revision"), "detail": (f"{h['sku']} @ ${h['price']} bundle={h.get('bundle')}" if h["agent"] == "strategist" else
                             f"{h.get('decision')} {h.get('violations')} tone={h.get('tone_score')}"), "ms": h.get("latency_ms"), "llm": h.get("llm_mode")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            d = out["draft"]; st.markdown(f"**Final draft** - {d['sku']} · ${d['offer_price']:.2f} ({d['pricing_unit']}) · {d['channel']}")
            st.text(f"Subject: {d['subject']}\n\n{d['body_copy']}"); st.caption(d["quant_rationale"])
            v = out["verdict"]
            if v["violations"]:
                st.markdown("**Latest auditor violations**"); st.table(pd.DataFrame([{"code": x["code"], "detail": x["detail"], "remediation": x["remediation"]} for x in v["violations"]]))
            if v.get("hitl_reason"):
                st.warning(f"HITL: {v['hitl_reason']}")
            st.graphviz_chart('digraph{rankdir=LR;node[shape=box,style=rounded];S[label="Agent 1\\nStrategist"];A[label="Agent 2\\nAuditor"];H[label="HITL"];F[label="Finalize"];S->A;A->S[label="REJECTED r<R_max"];A->F[label="APPROVED"];A->H[label="ESCALATE"];H->F}')

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
