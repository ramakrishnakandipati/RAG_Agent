"""Stage-by-stage rendering of one agent run (used by the End-to-End Flows tab)."""
from __future__ import annotations
import json
import pandas as pd
import streamlit as st

NODES = ["ingest_and_resolve", "score_guest", "select_strategy", "apply_guardrails", "compose_and_dispatch", "log_and_attribute"]
TITLES = {"ingest_and_resolve": "Stage 1 - Ingest & resolve identity", "score_guest": "Stage 2 - Score guest (uplift + cancel risk)",
          "select_strategy": "Stage 3 - Select strategy (LLM)", "apply_guardrails": "Stage 4 - Policy guardrails",
          "compose_and_dispatch": "Stage 5 - Compose & dispatch", "log_and_attribute": "Stage 6 - Log & attribute"}


def _gate_table(gates):
    if not gates: return
    st.dataframe(pd.DataFrame([{"phase": g["phase"], "result": "PASS" if g["passed"] else "FAIL",
                                "checks": ", ".join(("+ " if c["passed"] else "x ") + c["check"] for c in g["checks"])} for g in gates]), use_container_width=True, hide_index=True)


def render_run(r: dict, expect: str | None = None):
    wh, sc = r.get("webhook") or {}, r.get("scores") or {}
    by = {}
    for g in r.get("gates", []): by.setdefault(g["node"], []).append(g)
    trace = {}
    for t in r.get("trace", []): trace.setdefault(t["node"], []).append(t)
    # ---- progress strip
    cols = st.columns(6)
    for c, n in zip(cols, NODES):
        gs = by.get(n, []); ok = bool(gs) and all(g["passed"] for g in gs); loops = len(trace.get(n, []))
        c.markdown(f"**{'✅' if ok else ('❌' if gs else '⏭️')} {n.replace('_', ' ')}**" + (f"  ×{loops}" if loops > 1 else ""))
        c.caption((trace.get(n) or [{}])[-1].get("summary", "not executed"))
    # ---- outcome banner
    d = r.get("dispatch") or {}; status = d.get("status", "-")
    msg = f"**Outcome:** {r.get('final_offer_id') or '-'} -> `{status}`  |  holdout={r.get('is_holdout')}  |  retries={r.get('guardrail_retries', 0)}  |  LLM={r.get('llm_mode')}"
    (st.success if status == "sent" else st.warning)(msg)
    if expect: st.caption("Expected: " + expect)
    if r.get("error"): st.error(r["error"])

    # ---- stage 1
    with st.expander(TITLES["ingest_and_resolve"], expanded=True):
        a, b = st.columns(2)
        a.markdown("**Incoming webhook**"); a.json({k: wh.get(k) for k in ("booking_id", "ota", "first_name", "last_name", "masked_email", "phone_last4", "adr", "los", "ota_commission", "lead_days", "window", "ota_public_rate", "direct_public_rate")}, expanded=False)
        res = r.get("resolution") or {}; prof = r.get("resolved_profile") or {}
        b.markdown("**CDP identity resolution**"); b.write(f"Method **{res.get('method')}** (confidence {res.get('confidence')}) -> `{prof.get('profile_id')}`")
        b.write("Profile: " + ("shadow profile (no CDP match, consent unknown)" if prof.get("shadow") else f"{prof.get('loyalty_tier')} tier, {prof.get('past_direct_stays')} past direct stays, opt-in={prof.get('email_opt_in')}, msgs_30d={prof.get('msgs_30d')}"))
        b.metric("Saved commission if converted", f"${r.get('saved_commission_usd')}")
        _gate_table(by.get("ingest_and_resolve"))
    # ---- stage 2
    with st.expander(TITLES["score_guest"], expanded=True):
        m = st.columns(4); m[0].metric("Uplift τ (CATE)", f"{sc.get('tau', 0):+.3f}"); m[1].metric("P(cancel)", f"{sc.get('p_cancel', 0):.3f}")
        m[2].metric("Device", sc.get("device", "-")); m[3].metric("Inference", f"{sc.get('latency_ms', 0)} ms")
        st.caption("τ > 0: the intervention is expected to increase P(direct rebook); τ < 0 would be a 'sleeping dog'.")
        with st.popover("Feature vector"): st.json(r.get("features") or {})
        _gate_table(by.get("score_guest"))
    # ---- stage 3
    with st.expander(TITLES["select_strategy"], expanded=True):
        s3 = r.get("strategy") or {}
        st.write(f"Proposed **{s3.get('offer_id')}** via **{s3.get('channel')}** (LLM mode `{r.get('llm_mode')}`, attempts {r.get('llm_attempts')}); proposals rejected by policy so far: `{r.get('rejected_offers')}`")
        if s3.get("rationale"): st.markdown(f"> {s3['rationale']}")
        if s3.get("personalized_copy"): st.info(s3["personalized_copy"])
        _gate_table(by.get("select_strategy")[-2:] if by.get("select_strategy") else None)
    # ---- stage 4
    with st.expander(TITLES["apply_guardrails"], expanded=True):
        g4 = r.get("guardrail") or {}
        k = st.columns(4); k[0].markdown("Final verdict"); k[0].markdown("### :green[APPROVED]" if g4.get("approved") else "### :red[BLOCKED]"); k[1].metric("Retries", r.get("guardrail_retries", 0))
        k[2].metric("Expected net rev. Δ", f"${g4.get('enrd', 0)}"); k[3].metric("Margin after incentive", f"{g4.get('margin_pct', 0):.0%}")
        if r.get("downgraded"): st.warning(f"Downgraded: {g4.get('downgrade_reason')}  (initially blocked by {g4.get('initial_blocking')})")
        rr = g4.get("rule_results") or []
        if rr: st.dataframe(pd.DataFrame([{"rule": x["rule"], "result": "PASS" if x["passed"] else "BLOCK", "detail": json.dumps(x["detail"])} for x in rr]), use_container_width=True, hide_index=True)
        _gate_table(by.get("apply_guardrails")[-2:] if by.get("apply_guardrails") else None)
    # ---- stage 5
    with st.expander(TITLES["compose_and_dispatch"], expanded=True):
        if d.get("copy"):
            st.info(d["copy"]); st.code(d.get("link", ""), language=None)
        st.write(f"Dispatch status: **{status}**  channel: {d.get('channel')}  CRS inventory: {d.get('inventory')}")
        s3o = (r.get("strategy") or {}).get("offer_id")
        if s3o and r.get("final_offer_id") and s3o != r["final_offer_id"]:
            st.warning(f"Final offer **{r['final_offer_id']}** differs from the LLM proposal **{s3o}** (policy downgrade or CRS inventory swap); the copy was re-rendered from a template.")
        _gate_table(by.get("compose_and_dispatch"))
    # ---- stage 6
    with st.expander(TITLES["log_and_attribute"], expanded=True):
        st.json(r.get("attribution") or {}, expanded=False); st.caption("Appended to outputs/logs/agent_decisions.jsonl; treated vs ghost-holdout attribution uses `treatment` / `ghost_holdout`.")
        _gate_table(by.get("log_and_attribute"))
