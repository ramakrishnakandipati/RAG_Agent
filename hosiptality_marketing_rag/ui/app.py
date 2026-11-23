"""Streamlit UI for the Hospitality Marketing RAG.
Run (API must be up):  streamlit run ui/app.py      or      python scripts/run_ui.py"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import api_client as api  # noqa: E402
from ui.components import CSS, chips_html, gauge_html, render_answer_html, score_bar_html  # noqa: E402

st.set_page_config(page_title="Hospitality Marketing RAG", page_icon=":hotel:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
S = st.session_state
S.setdefault("question", "")
S.setdefault("last", None)
S.setdefault("history", [])
S.setdefault("verify_open", False)
AUTO = "Auto (from question)"
STAGES = ["rewrite", "retrieve", "rerank", "generate", "verify"]


def pick_example(q: str) -> None:
    S.question = q


def open_verifier() -> None:
    S.verify_open = True


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Hospitality RAG")
    try:
        h = api.health()
        st.markdown(f'<span class="chip">API {h["status"].upper()}</span><span class="chip soft">{h["chunks"]} chunks</span>'
                    f'<span class="chip soft">LLM: {h["llm_mode"]}</span><span class="chip soft">'
                    f'{"GPU" if h["cuda"] else "CPU"}</span>', unsafe_allow_html=True)
        opts = api.filters()
        examples = api.examples()
        api_ok = h["index_ready"]
    except api.ApiError as e:
        st.error(str(e)); st.stop()
    st.subheader("Filters")
    st.caption("Explicit filters override anything extracted from the question.")
    f_prop = st.selectbox("Property", [AUTO] + opts["property"])
    f_per = st.selectbox("Period", [AUTO] + opts["period"])
    f_ch = st.selectbox("Channel", [AUTO] + opts["channel"])
    top_k = st.slider("Evidence chunks (top-k)", 1, 10, 5)
    st.subheader("Examples")
    for ex in examples:
        st.button(ex["label"], key=f"ex_{ex['id']}", on_click=pick_example, args=(ex["question"],), use_container_width=True)

overrides = {k: v for k, v in (("property", f_prop), ("period", f_per), ("channel", f_ch)) if v != AUTO}

# ------------------------------------------------------------------ header + question
st.title("Hospitality Marketing RAG")
st.caption("Hybrid retrieval (dense + BM25 + RRF) - cross-encoder rerank - grounded, cited answers - numeric verification")
st.text_area("Ask a question about hotel KPIs, campaigns or brand guidelines", key="question", height=80,
             placeholder="e.g. What was the ADR for the Direct channel at Miami Beach Resort in Q3 2025?")
ask = st.button("Ask", type="primary", disabled=not (S.question.strip() and api_ok))

if ask:
    with st.status("Running the 9-stage pipeline (first query loads GPU models)...", expanded=False) as status:
        try:
            res = api.query(S.question.strip(), overrides or None, top_k)
            S.last = res
            S.verify_open = False
            S.history.append({"question": res["question"], "faithfulness": res["verification"]["faithfulness_score"],
                              "refused": res["verification"]["is_refusal"], "filters": res["rewritten"]["filters"],
                              "total_s": round(res["latency_ms"]["total"] / 1000, 1)})
            status.update(label=f'Done in {res["latency_ms"]["total"] / 1000:.1f} s', state="complete")
        except api.ApiError as e:
            S.last = None
            status.update(label="Request failed", state="error")
            st.error(str(e))

tab_ask, tab_ev, tab_pipe, tab_met, tab_hist = st.tabs(["Ask", "Evidence", "Pipeline", "Metrics", "History"])
res = S.last

# ------------------------------------------------------------------ Ask
with tab_ask:
    if not res:
        st.info("Pick an example in the sidebar or type a question, then press Ask.")
    else:
        v = res["verification"]
        left, right = st.columns([3, 2], gap="large")
        with left:
            if v["is_refusal"]:
                st.markdown(f'<div class="refusal"><b>Out of scope - no answer given.</b><br>'
                            f'{res["answer"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer-card">{render_answer_html(res["answer"])}</div>', unsafe_allow_html=True)
            if res["rewritten"]["filters"]:
                st.markdown("**Filters applied:** " + chips_html(res["rewritten"]["filters"], tuple(res["rewritten"]["filters"])),
                            unsafe_allow_html=True)
        with right:
            st.markdown(gauge_html(v["faithfulness_score"]), unsafe_allow_html=True)
            st.caption("1 - (unsupported numbers + flagged sentences + citation errors) / (numbers + sentences checked)")
            if v["claims"]:
                df = pd.DataFrame([{"Claim value": c["value"], "Supported": "yes" if c["supported"] else "NO",
                                    "Evidence chunk": (c["evidence_chunk"] or "-").split("::")[0]} for c in v["claims"]])
                st.dataframe(df, hide_index=True, use_container_width=True)
            elif v["is_refusal"]:
                st.success("No numeric claims made - nothing to hallucinate.")
            for e in v["citation_errors"]:
                st.error(e)
            for j in v["judge_unsupported"]:
                st.warning("Judge flagged: " + j[:160])
        with st.expander("Try the verifier: inject a fabricated number", expanded=S.verify_open):
            base = res["answer"]
            fake = re.sub(r"\$\s?\d[\d,]*(?:\.\d+)?", "$299", base, count=1)
            txt = st.text_area("Answer to verify (edit freely)", value=fake, height=110, key=f"verify_text_{abs(hash(base))}")
            if st.button("Verify against the retrieved evidence", on_click=open_verifier):
                try:
                    vr = api.verify(txt, [c["chunk_id"] for c in res["contexts"]])
                    st.markdown(gauge_html(vr["faithfulness_score"], "Faithfulness of edited answer"), unsafe_allow_html=True)
                    for c in vr["unsupported_claims"]:
                        st.error(f'BLOCKED {c["value"]} - not found in the retrieved context')
                    for c in vr["claims"]:
                        if c["supported"]:
                            st.success(f'Verified {c["value"]}')
                except api.ApiError as e:
                    st.error(str(e))

# ------------------------------------------------------------------ Evidence
with tab_ev:
    if not res:
        st.info("Ask a question to see the retrieved evidence.")
    else:
        rq = res["rewritten"]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Rewritten search queries** (max 3)")
            for i, q in enumerate(rq["search_queries"], 1):
                st.markdown(f"{i}. {q}")
        with c2:
            st.markdown("**Acronym expansion**")
            if rq["expansions"]:
                st.table(pd.DataFrame(rq["expansions"].items(), columns=["Acronym", "Expanded to"]).set_index("Acronym"))
            else:
                st.caption("No acronyms in the question.")
        st.markdown(f"**Top {len(res['contexts'])} chunks after reranking**")
        for i, c in enumerate(res["contexts"], 1):
            m = c["metadata"]
            with st.expander(f"{i}. {m['source']} - page {m['page']} - {m['period']}   (rerank {c['score']:.3f})", expanded=i == 1):
                st.markdown(score_bar_html(c["score"]) + chips_html(m), unsafe_allow_html=True)
                if c["kind"] == "table":
                    try:
                        full = api.chunk(c["chunk_id"])
                        st.markdown("**Source table (kept whole)**")
                        st.markdown(full["table_md"].replace("$", "\\$"))
                    except api.ApiError as e:
                        st.caption(str(e))
                st.text(c["text"].split("Row summaries:")[-1].strip() if c["kind"] == "table" else c["text"])

# ------------------------------------------------------------------ Pipeline
with tab_pipe:
    if not res:
        st.info("Ask a question to see per-stage latency.")
    else:
        lat = res["latency_ms"]
        df = pd.DataFrame({"stage": STAGES, "ms": [lat.get(s, 0) for s in STAGES]})
        df["label"] = df["ms"].map(lambda x: f"{x / 1000:.1f} s" if x >= 1000 else f"{x:.0f} ms")
        base = alt.Chart(df).encode(y=alt.Y("stage:N", sort=STAGES, title=None), x=alt.X("ms:Q", title="milliseconds"))
        st.altair_chart((base.mark_bar(color="#0F766E") + base.mark_text(align="left", dx=4).encode(text="label:N")).properties(height=240),
                        use_container_width=True)
        llm_share = (lat["rewrite"] + lat["generate"] + lat["verify"]) / max(lat["total"], 1) * 100
        k1, k2, k3 = st.columns(3)
        k1.metric("Total", f'{lat["total"] / 1000:.1f} s')
        k2.metric("LLM calls share", f"{llm_share:.0f}%")
        k3.metric("Retrieve + rerank", f'{lat["retrieve"] + lat["rerank"]:.0f} ms')

# ------------------------------------------------------------------ Metrics
with tab_met:
    try:
        mt = api.metrics()
    except api.ApiError as e:
        st.error(str(e)); mt = {}
    if mt.get("retrieval"):
        rows = [{"system": k, "metric": m.split("@")[0].upper(), "value": v} for k, d in mt["retrieval"].items() for m, v in d.items()]
        ch = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
            x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="system:N", y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.05]), title="score"),
            color=alt.Color("system:N", scale=alt.Scale(range=["#94A3B8", "#5B8E7D", "#0F766E", "#C8793B"])),
            tooltip=["system", "metric", alt.Tooltip("value:Q", format=".3f")]).properties(height=300)
        st.subheader("Retrieval quality (12 golden questions)")
        st.altair_chart(ch, use_container_width=True)
        t = mt.get("tests", {})
        k1, k2, k3 = st.columns(3)
        k1.metric("Mean faithfulness", f'{mt.get("mean_faithfulness") or 0:.2f}')
        k2.metric("Tests passed", f'{t.get("passed", "-")} / {t.get("total", "-")}')
        k3.metric("Indexed chunks", mt.get("ingest", {}).get("chunks", "-"))
        if mt.get("latency"):
            st.subheader("Latency per benchmark query (ms)")
            st.dataframe(pd.DataFrame(mt["latency"]).round(0), hide_index=True, use_container_width=True)
    else:
        st.info("No metrics yet - run `python scripts/run_end_to_end.py`.")

# ------------------------------------------------------------------ History
with tab_hist:
    if not S.history:
        st.info("Questions you ask in this session appear here.")
    else:
        st.dataframe(pd.DataFrame(S.history[::-1]), hide_index=True, use_container_width=True)
