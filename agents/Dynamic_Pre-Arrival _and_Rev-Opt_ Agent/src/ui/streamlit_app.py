"""Streamlit Control Center: live trigger, state trace, HITL queue, data lineage, metrics & logs. Talks to the Flask API."""
from __future__ import annotations
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
API = os.environ.get("API_BASE", f"http://127.0.0.1:{os.environ.get('FLASK_PORT', '5055')}") + "/api/v1"

st.set_page_config(page_title="Revenue Agent Control Center", page_icon="🏨", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=600, **kw)
        return r.status_code, r.json()
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


st.title("🏨 Dynamic Pre-Arrival & Revenue Optimization Agent")
code, health = api("GET", "/health")
c1, c2, c3, c4 = st.columns(4)
c1.metric("API", "online" if code == 200 else "offline")
c2.metric("LLM", f"{health.get('llm_model', '-')}" if code == 200 else "-", "up" if health.get("llm_up") else ("mock" if health.get("llm_mode") == "mock" else "down"))
gpu = health.get("gpu", {})
c3.metric("GPU", gpu.get("name", "cpu"), f"{gpu.get('free_vram_gb', 0)} GB free" if gpu.get("cuda") else "CPU fallback")
c4.metric("Folio postings", health.get("pms", {}).get("folio", 0), f"{health.get('pms', {}).get('hitl_queue', 0)} HITL items")

tab1, tabuc, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Live Trigger", "🎬 Use Cases", "🧭 State Trace", "🧑‍💼 HITL Queue", "🗺️ Data Lineage", "📊 Metrics & Logs"])

# ------------------------------------------------------------------ tab 1
with tab1:
    code, rl = api("GET", "/reservations", params={"limit": 60})
    resv = rl.get("reservations", []) if code == 200 else []
    labels = {f"{r['reservation_id']} · {r['loyalty_tier']} · {r['segment']} · {r['channel_group']} · {r['check_in']}": r["reservation_id"] for r in resv}
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Trigger a guest journey")
        pick = st.selectbox("Upcoming reservation", list(labels) or ["(none)"], key="resv_pick")
        adv = st.selectbox("Adversarial test hook", ["none", "once", "persistent"],
                           help="'once' injects a draft with a fabricated price/SKU on the first attempt; 'persistent' on every attempt (forces HITL).")
        if st.button("Run pre-arrival workflow", type="primary", disabled=not labels):
            with st.spinner("Running LangGraph workflow..."):
                code, res = api("POST", "/trigger", json={"reservation_id": labels[pick], "adversarial": adv})
            st.session_state["last"] = res
            st.session_state["last_rid"] = labels[pick]
        st.divider()
        st.subheader("Two-way guest reply")
        txt = st.text_input("Guest reply", "Yes please, add it to my booking!")
        rid_reply = st.session_state.get("last_rid") or (labels[pick] if labels else "")
        if st.button("Send inbound reply", disabled=not rid_reply):
            with st.spinner("Processing reply..."):
                code, res = api("POST", "/reply", json={"reservation_id": rid_reply, "text": txt})
            st.session_state["last"] = res
    with right:
        res = st.session_state.get("last")
        if res:
            if res.get("error"):
                st.error(res)
            else:
                badge = {"SENT": "🟢", "BOOKED": "✅", "ESCALATED": "🟠", "NO_OFFER": "⚪", "FAIL_VALIDATION": "🔴"}.get(res["status"], "🔵")
                st.markdown(f"### {badge} Status: `{res['status']}`  ·  thread `{res['thread_id']}`")
                if res.get("hitl_reasons"):
                    st.warning("Escalated to human concierge: " + ", ".join(res["hitl_reasons"]))
                o = res.get("offer")
                if o:
                    st.markdown("**Selected offer (from RMS/XGBoost — the only source of prices)**")
                    st.dataframe(pd.DataFrame([{k: o[k] for k in ("sku", "name", "list_price", "discount_pct", "final_price", "margin_pct", "propensity", "expected_value", "available_units")}]), hide_index=True)
                m = res.get("message")
                if m:
                    st.markdown(f"**Message** — _{m.get('subject', '')}_ (source: `{m.get('source')}`)")
                    st.info(m.get("body") or m.get("body_template", ""))
                if res.get("guardrail"):
                    g = res["guardrail"]
                    st.markdown(f"**Guardrail:** `{g['status']}`")
                    st.dataframe(pd.DataFrame([{"check": k, "passed": v} for k, v in g["checks"].items()]), hide_index=True)
                    if g["violations"]:
                        st.error("Violations: " + ", ".join(g["violations"]))
                if res.get("booking"):
                    st.success(f"Folio posting: {res['booking']}")
        else:
            st.info("Pick a reservation and run the workflow.")


# ------------------------------------------------------------------ use cases
BADGE = {"SENT": "🟢", "BOOKED": "✅", "ESCALATED": "🟠", "NO_OFFER": "⚪", "FAIL_VALIDATION": "🔴", "REJECTED": "⛔", "SOLD_OUT": "🟡"}
with tabuc:
    code, sl = api("GET", "/scenarios")
    scs = sl.get("scenarios", []) if code == 200 else []
    st.subheader("Sample use cases - end-to-end flows")
    st.caption("Each case auto-selects a suitable upcoming reservation and runs the real workflow: GPU XGBoost + RMS rules, Qwen3-VL drafting, compliance gate, HITL, folio.")
    if scs:
        titles = {sc["title"]: sc for sc in scs}
        choice = st.radio("Choose a use case", list(titles), key="uc_choice")
        sc = titles[choice]
        st.markdown(f"**Persona:** {sc['persona']}")
        st.markdown(f"**What happens:** {sc['story']}")
        st.markdown(f"**Expected outcome:** `{sc['outcome']}` · **Steps:** {sc['n_steps']}")
        if st.button("▶ Run use case end-to-end", type="primary", key="uc_run"):
            with st.spinner("Running use case through the LangGraph workflow..."):
                code, out = api("POST", f"/scenarios/{sc['id']}/run")
            st.session_state["uc_result"] = out
        out = st.session_state.get("uc_result")
        if out and out.get("scenario", {}).get("id") == sc["id"]:
            if out.get("error"):
                st.error(out["error"])
            else:
                g, r = out["guest"], out["reservation"]
                st.success(f"Guest **{g['first_name']}** ({g['segment']}, {g['loyalty_tier']}, LTV ${g['ltv_usd']:,.0f}) · reservation `{r['reservation_id']}` "
                           f"· {r['channel_group']} · check-in {r['check_in']} · total {out['total_seconds']} s")
                for stp in out["steps"]:
                    rs = stp["response"]
                    with st.container(border=True):
                        st.markdown(f"#### Step {stp['n']} · {stp['title']}  {BADGE.get(rs['status'], '🔵')} `{rs['status']}`")
                        st.caption(f"{stp['request']['endpoint']}  {json.dumps(stp['request']['body'])}  ·  {stp['seconds']} s")
                        path = " → ".join(t["node"] for t in rs["trace"]) or "-"
                        st.markdown(f"**Graph path:** `{path}`")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if rs.get("intent"):
                                st.markdown(f"**Guest intent:** `{rs['intent'].get('intent')}` · sentiment `{rs['intent'].get('sentiment')}`")
                            o = rs.get("offer")
                            if o:
                                st.markdown(f"**Offer:** {o['name']} (`{o['sku']}`) · list ${o['list_price']:.2f} · discount {o['discount_pct']:g}% · **final ${o['final_price']:.2f}** · p={o['propensity']:.2f}")
                            if rs.get("violations_seen"):
                                st.error("Gate caught: " + ", ".join(rs["violations_seen"]) + f"  → retries: {rs['retries']}")
                            if rs.get("hitl_reasons"):
                                st.warning("Human-in-the-loop: " + ", ".join(rs["hitl_reasons"]))
                            if rs.get("guardrail"):
                                st.markdown(f"**Guardrail:** `{rs['guardrail']['status']}`")
                            if rs.get("booking"):
                                st.success(f"Folio posting: {rs['booking']}")
                        with cc2:
                            m = rs.get("message")
                            if m:
                                st.markdown(f"**Message** _{m.get('subject', '')}_ · source `{m.get('source')}`")
                                st.info(m.get("body") or m.get("body_template", ""))
                if out.get("folio"):
                    st.markdown("**Room folio after the flow**")
                    st.dataframe(pd.DataFrame(out["folio"])[["sku", "qty", "amount", "service_date"]], hide_index=True)
    else:
        st.warning("Scenario API unavailable.")

# ------------------------------------------------------------------ tab 2
with tab2:
    res = st.session_state.get("last")
    if not res or res.get("error"):
        st.info("Run a workflow in the first tab to inspect its state transitions.")
    else:
        tr = pd.DataFrame(res.get("trace", []))
        st.subheader("LangGraph node transitions")
        if not tr.empty:
            st.dataframe(tr[["node", "status", "pre_ok", "post_ok", "latency_ms", "note"] if "note" in tr else ["node", "status", "pre_ok", "post_ok", "latency_ms"]], hide_index=True, use_container_width=True)
            fig = px.bar(tr, x="latency_ms", y=tr.index.astype(str) + " · " + tr["node"], orientation="h", color="status", title="Node latency (ms) in execution order")
            fig.update_layout(yaxis_title="", height=360, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Repeated draft_message → compliance_check pairs indicate the FAIL_POLICY retry loop; hitl_gate = governance checkpoint.")
        st.json(res, expanded=False)

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Human-in-the-loop approval queue")
    code, pend = api("GET", "/hitl/pending")
    items = pend.get("pending", []) if code == 200 else []
    if st.button("Refresh queue"):
        st.rerun()
    if not items:
        st.success("No pending escalations. Trigger an adversarial 'persistent' run or send a complaint reply to create one.")
    for it in items:
        with st.container(border=True):
            st.markdown(f"**{it['thread_id']}** · guest `{it['guest_id']}` · reasons: " + ", ".join(f"`{r}`" for r in it["reasons"]))
            body = st.text_area("Draft (editable)", it["draft"].get("body", ""), key=f"b_{it['thread_id']}")
            a, b, _ = st.columns([1, 1, 4])
            if a.button("✅ Approve", key=f"a_{it['thread_id']}"):
                code, r = api("POST", "/hitl/approve", json={"thread_id": it["thread_id"], "decision": "approve", "reviewer": "ui_concierge",
                                                             "edited_body": body if body != it["draft"].get("body") else None})
                st.session_state["last"] = r
                st.rerun()
            if b.button("⛔ Reject", key=f"r_{it['thread_id']}"):
                api("POST", "/hitl/approve", json={"thread_id": it["thread_id"], "decision": "reject", "reviewer": "ui_concierge"})
                st.rerun()
    st.subheader("Recent outbox")
    code, ob = api("GET", "/outbox", params={"limit": 8})
    if code == 200 and ob.get("outbox"):
        st.dataframe(pd.DataFrame(ob["outbox"])[["ts", "reservation_id", "sku", "subject", "status"]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("Data lineage")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style=rounded];
      PMS[label="PMS reservations"]; CRM[label="CRM Guest360"]; RMS[label="RMS inventory & occupancy"]; WX[label="Weather & events"];
      KB[label="Knowledge base (md/json)"]; FEAT[label="Feature table (NBA)"]; XGB[label="XGBoost (GPU hist)"]; RAG[label="Chroma + BM25 + RRF (GPU)"];
      RULES[label="RMS rules (margin/inventory/discount)"]; LLM[label="Qwen3-VL drafting (placeholders)"]; GATE[label="Compliance gate"]; OUT[label="Guest message / folio"];
      PMS->FEAT; CRM->FEAT; RMS->FEAT; WX->FEAT; FEAT->XGB; XGB->RULES; RMS->RULES; KB->RAG; RAG->LLM; RULES->LLM; LLM->GATE; RULES->GATE; GATE->OUT; }""")
    raw = ROOT / "data" / "raw"
    pick = st.selectbox("Explore a table", ["guests", "reservations", "offer_history", "inventory", "occupancy", "weather"])
    if (raw / f"{pick}.csv").exists():
        df = pd.read_csv(raw / f"{pick}.csv", nrows=500)
        st.caption(f"{pick}: first 500 rows · {len(df.columns)} columns")
        st.dataframe(df, use_container_width=True, height=300)
    vr = ROOT / "validation" / "data_validation_report.json"
    if vr.exists():
        rep = json.loads(vr.read_text())
        st.metric("Data validation checks passed", f"{rep['n_passed']}/{rep['n_checks']}")
        st.dataframe(pd.DataFrame(rep["checks"]), hide_index=True, use_container_width=True, height=240)

# ------------------------------------------------------------------ tab 5
with tab5:
    code, m = api("GET", "/metrics")
    if code != 200:
        st.warning("No metrics yet — run `python run_all.py` first.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Hallucination rate", f"{m['hallucination_rate_pct']}%", f"{m['messages_judged']} msgs judged")
        b.metric("Adversarial blocked", f"{m['adversarial_caught']}/{m['adversarial_injected']}", f"{m['adversarial_sent_to_guest']} reached guest")
        c.metric("Model ROC-AUC / KS", f"{m['model']['roc_auc']} / {m['model']['ks']}", m["model"]["device"])
        d.metric("Ancillary RevPAR lift", f"{m['revenue']['anc_revpar_lift_pct']}%", f"${m['revenue']['anc_revpar_lift_usd']}/room-night")
        f = m["funnel"]
        st.plotly_chart(px.bar(x=list(f), y=list(f.values()), title="Guardrail funnel", labels={"x": "", "y": "journeys"}), use_container_width=True)
        lat = pd.DataFrame(m["node_latency"]).T.reset_index().rename(columns={"index": "node"})
        st.plotly_chart(px.bar(lat, x="node", y=["mean_ms", "p95_ms"], barmode="group", title="Node latency"), use_container_width=True)
        st.json(m["revenue"], expanded=False)
    ev = ROOT / "logs" / "events.jsonl"
    if ev.exists():
        lines = ev.read_text(encoding="utf-8").splitlines()[-40:]
        st.subheader("Recent structured logs")
        st.code("\n".join(lines), language="json")
