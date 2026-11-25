"""Streamlit operations & flow dashboard: Live Agent Flow | Data Explorer | Guardrails | ROI & Validation."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings, detect_hardware, load_catalog
from agent.llm_client import endpoint

S = get_settings()
st.set_page_config(page_title="OTA-to-Direct Agent", layout="wide")
NODES = ["ingest_and_resolve", "score_guest", "select_strategy", "apply_guardrails", "compose_and_dispatch", "log_and_attribute"]


def rj(*p, default=None):
    f = S.out.joinpath(*p)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default


@st.cache_data
def bookings():
    from data.synthetic_generator import load
    return load()[0]


@st.cache_data
def resolved():
    p = Path(__file__).resolve().parents[1] / "data" / "generated" / "resolved_bookings.csv"
    return pd.read_csv(p) if p.exists() else None


def tail(name, n=200):
    p = S.out / "logs" / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()] if p.exists() else []


ep, hw = endpoint(), detect_hardware()
st.title("OTA-to-Direct Conversion Agent - Operations Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("LLM", "live" if ep["available"] else "offline fallback", ep["model"])
c2.metric("Compute", hw["device_name"] or "CPU", f"CUDA {hw['cuda_version']}" if hw["cuda"] else "no CUDA")
c3.metric("Models promoted", str((rj("models", "promotion.json", default={}) or {}).get("promoted")))
t0, t1, t2, t3, t4 = st.tabs(["End-to-End Flows", "Live Agent Flow", "Data Explorer", "Guardrails", "ROI & Validation"])

# ------------------------------------------------------------------ tab 0: curated end-to-end flows
with t0:
    from agent.scenarios import build_scenarios
    from agent.graph import run_agent
    from ui_streamlit.flow_view import render_run
    SC = build_scenarios(); runs = st.session_state.setdefault("flow_runs", {})
    st.markdown("Five curated flows, each taken from a real booking in the dataset and sent through all **six LangGraph stages** with their validation gates.")
    left, right = st.columns([1, 3])
    with left:
        pick = st.radio("Flow", [s["title"] for s in SC], key="flow_pick")
        cur = next(s for s in SC if s["title"] == pick)
        st.caption(cur["description"])
        if st.button("Run selected flow", type="primary", key="run_flow"):
            runs[cur["id"]] = run_agent(cur["webhook"])
        if st.button("Run all 5 flows", key="run_all"):
            with st.spinner("Running 5 flows..."):
                for s in SC: runs[s["id"]] = run_agent(s["webhook"])
    with right:
        st.subheader(cur["title"]); st.write(f"**What should happen:** {cur['expect']}")
        if cur["id"] in runs: render_run(runs[cur["id"]], cur["expect"])
        else: st.info("Click **Run selected flow** to execute the six stages.")
    if runs:
        st.markdown("#### Flow comparison")
        st.dataframe(pd.DataFrame([{"flow": s["title"], "identity": (runs[s["id"]].get("resolution") or {}).get("method"), "tau": round((runs[s["id"]].get("scores") or {}).get("tau", 0), 3),
                                    "final offer": runs[s["id"]].get("final_offer_id"), "dispatch": (runs[s["id"]].get("dispatch") or {}).get("status"), "guardrail retries": runs[s["id"]].get("guardrail_retries"),
                                    "holdout": runs[s["id"]].get("is_holdout")} for s in SC if s["id"] in runs]), use_container_width=True, hide_index=True)



# ------------------------------------------------------------------ tab 1
with t1:
    from web_flask.app import sample_webhook
    b = bookings()
    a, c = st.columns([1, 2])
    with a:
        idx = st.number_input("Booking row", 0, len(b) - 1, 0)
        window = st.selectbox("Window", ["pre_arrival", "on_property", "post_stay"])
        scen = st.selectbox("Scenario", ["As recorded", "OTA parity breach (direct 10% below OTA)", "Force ghost holdout", "Force treated"])
        wh = sample_webhook(int(idx), window)
        if scen.startswith("OTA parity"): wh["direct_public_rate"] = round(wh["ota_public_rate"] * 0.9, 2)
        if scen == "Force ghost holdout": wh["is_ghost_holdout"] = True
        if scen == "Force treated": wh["is_ghost_holdout"] = False
        st.json(wh, expanded=False)
        go = st.button("Run agent", type="primary")
    with c:
        if go:
            from agent.graph import run_agent
            r = run_agent(wh)
            st.session_state["run"] = r
        r = st.session_state.get("run")
        if r:
            by = {}
            for g in r["gates"]: by.setdefault(g["node"], []).append(g)
            cols = st.columns(6)
            for col, n in zip(cols, NODES):
                gs = by.get(n, []); ok = bool(gs) and all(g["passed"] for g in gs); tr = [t for t in r["trace"] if t["node"] == n]
                col.markdown(f"**{'✅' if ok else ('❌' if gs else '⏭️')} {n}**" + (f" ×{len(tr)}" if len(tr) > 1 else ""))
                col.caption(tr[-1].get("summary", "") if tr else "not executed")
            s = r.get("scores", {}); m = st.columns(5)
            m[0].metric("Uplift τ", f"{s.get('tau', 0):.3f}"); m[1].metric("P(cancel)", f"{s.get('p_cancel', 0):.3f}"); m[2].metric("Saved commission", f"${r.get('saved_commission_usd', 0)}")
            m[3].metric("Final offer", r.get("final_offer_id") or "-"); m[4].metric("Dispatch", (r.get("dispatch") or {}).get("status", "-"))
            if r.get("error"): st.error(r["error"])
            if (r.get("dispatch") or {}).get("copy"): st.info(r["dispatch"]["copy"] + "\n\n" + r["dispatch"]["link"])
            st.caption(f"Identity: {r['resolution'].get('method')} -> {r['resolved_profile']['profile_id']} | holdout={r.get('is_holdout')} | LLM={r.get('llm_mode')}")
            with st.expander("Validation gates for this run"):
                st.dataframe(pd.DataFrame([{"node": g["node"], "phase": g["phase"], "passed": g["passed"], "failed_checks": ", ".join(c["check"] for c in g["checks"] if not c["passed"])} for g in r["gates"]]), use_container_width=True)
            with st.expander("Guardrail verdict"):
                st.dataframe(pd.DataFrame([{**x, "detail": json.dumps(x["detail"])} for x in r["guardrail"].get("rule_results", [])]), use_container_width=True)
        else:
            st.caption("Run the agent to see the 6-node flow.")

# ------------------------------------------------------------------ tab 2
with t2:
    rb = resolved()
    st.subheader("PMS bookings (synthetic)")
    ota = st.multiselect("OTA", ["Expedia", "Booking.com", "Agoda"], default=["Expedia", "Booking.com", "Agoda"])
    d = (rb if rb is not None else bookings()); d = d[d.ota.isin(ota)]
    k = st.columns(4); k[0].metric("Bookings", f"{len(d):,}"); k[1].metric("Ghost holdout", f"{d.is_ghost_holdout.mean():.1%}"); k[2].metric("Avg commission", f"{d.ota_commission.mean():.1%}"); k[3].metric("Avg saved commission", f"${d.saved_commission_usd.mean():.0f}")
    st.bar_chart(d.groupby("ota").size())
    if rb is not None:
        st.subheader("Identity resolution"); st.dataframe(d.resolution_method.value_counts().rename("bookings"), use_container_width=True)
    st.dataframe(d.drop(columns=[c for c in ("true_tau",) if c in d.columns]).head(500), use_container_width=True)

# ------------------------------------------------------------------ tab 3
with t3:
    st.subheader("Policy rules")
    st.markdown("R1 OTA rate parity · R2 margin floor · R3 incentive ≤ 60% of saved commission · R4 positive expected net revenue delta · R5 fatigue cap & consent · R6 window fit")
    st.json(load_catalog()["constraints"], expanded=False)
    st.dataframe(pd.DataFrame(load_catalog()["offers"]), use_container_width=True)
    aud = tail("guardrail_audits.jsonl")
    if aud:
        rows = [{"ts": a["ts"][:19], "booking": a["booking_id"], "proposed": a["proposed_offer"], "approved": a["verdict"]["approved"], "blocking": ", ".join(a["verdict"]["blocking_rules"]),
                 "retry": a.get("retry"), "downgraded_to": a.get("downgraded_to")} for a in reversed(aud)]
        df = pd.DataFrame(rows); st.metric("Blocked proposals (recent)", int((~df.approved).sum()), f"of {len(df)}")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df[~df.approved].blocking.str.split(", ").explode().value_counts())
    else:
        st.caption("No guardrail audits yet.")

# ------------------------------------------------------------------ tab 4
with t4:
    ev = rj("evaluation", "offline_replay_roi.json", default={}); dv = rj("validation", "data_validation.json", default={})
    if ev:
        k = st.columns(5); k[0].metric("Lift (treated − holdout)", f"{ev['incremental_lift']:.1%}", f"CI {ev['lift_ci95'][0]:.1%}..{ev['lift_ci95'][1]:.1%}", delta_color="off")
        k[1].metric("Net revenue Δ", f"${ev['net_revenue_delta_usd']:,.0f}"); k[2].metric("ROI on incentives", f"{ev['roi_x']}x"); k[3].metric("Targeted share", f"{ev['target_rate']:.0%}")
        k[4].metric("Unconstrained blast (non-compliant)", f"${ev['blast_all_net_revenue_delta_usd']:,.0f}")
    gcols = st.columns(3)
    for col, g in zip(gcols, ("D1", "D2", "D3")):
        if g in dv: col.metric(dv[g]["gate"], "PASS" if dv[g]["passed"] else "FAIL")
    imgs = sorted((S.out / "charts").glob("*.png"))
    for i in range(0, len(imgs), 2):
        cc = st.columns(2)
        for col, p in zip(cc, imgs[i:i + 2]): col.image(str(p), caption=p.name)
    vl = rj("evaluation", "vlm_visual_audit.json")
    if vl: st.subheader("VLM visual audit"); st.dataframe(pd.DataFrame(vl["audits"])[["file", "mode", "quality_score", "issues", "summary"]], use_container_width=True)
