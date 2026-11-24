"""Static PNG (Matplotlib/Seaborn) + interactive HTML (Plotly) charts -> outputs/charts/. All data read from generated files."""
from __future__ import annotations
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src.config import get_settings

PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#6B6B6B"]
sns.set_theme(style="whitegrid", font_scale=0.95)
NODE_ORDER = ["enrich_context", "retrieve_knowledge_and_offers", "draft_message", "compliance_check", "hitl_gate", "dispatch_and_log"]


def _save(name: str, mfig, pfig) -> None:
    s = get_settings()
    s.charts_dir.mkdir(parents=True, exist_ok=True)
    mfig.tight_layout()
    mfig.savefig(s.charts_dir / f"{name}.png", dpi=160)
    plt.close(mfig)
    pfig.update_layout(template="plotly_white", font=dict(size=13), margin=dict(l=60, r=30, t=60, b=50), colorway=PAL)
    pfig.write_html(s.charts_dir / f"{name}.html", include_plotlyjs="cdn")


def bar(name, title, x, y, xlabel="", ylabel="", horizontal=False, series: dict | None = None):
    """Single series (x,y) or grouped bars (series={label: values}, x = categories)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    pf = go.Figure()
    if series is None:
        series = {"": list(y)}
    k = len(series)
    idx = np.arange(len(x))
    w = 0.8 / k
    for i, (lab, vals) in enumerate(series.items()):
        if horizontal:
            ax.barh(idx + i * w, vals, height=w, color=PAL[i], label=lab or None)
            pf.add_trace(go.Bar(y=list(x), x=list(vals), orientation="h", name=lab, marker_color=PAL[i]))
        else:
            ax.bar(idx + i * w, vals, width=w, color=PAL[i], label=lab or None)
            pf.add_trace(go.Bar(x=list(x), y=list(vals), name=lab, marker_color=PAL[i]))
    if horizontal:
        ax.set_yticks(idx + w * (k - 1) / 2, x); ax.invert_yaxis(); pf.update_yaxes(autorange="reversed")
    else:
        ax.set_xticks(idx + w * (k - 1) / 2, x, rotation=20 if max(len(str(v)) for v in x) > 8 else 0)
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if k > 1 or series.get(""):
        if k > 1: ax.legend(frameon=False)
    pf.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, barmode="group")
    _save(name, fig, pf)


def lines(name, title, series: dict, xlabel="", ylabel="", diagonal=False):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    pf = go.Figure()
    for i, (lab, (xs, ys)) in enumerate(series.items()):
        ax.plot(xs, ys, color=PAL[i], lw=2, marker="o" if len(xs) < 20 else None, label=lab)
        pf.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers" if len(xs) < 20 else "lines", name=lab, line=dict(color=PAL[i])))
    if diagonal:
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="reference")
        pf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="reference", line=dict(dash="dash", color="#999")))
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(frameon=False)
    pf.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    _save(name, fig, pf)


def heat(name, title, mat, xl, yl, fmt="d"):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", xticklabels=xl, yticklabels=yl, ax=ax, cbar=False)
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel("to"); ax.set_ylabel("from")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    pf = go.Figure(go.Heatmap(z=mat, x=xl, y=yl, colorscale="Blues", text=mat, texttemplate="%{text}"))
    pf.update_layout(title=title, xaxis_title="to", yaxis_title="from", yaxis_autorange="reversed")
    _save(name, fig, pf)


def generate_all_charts() -> list[str]:
    s = get_settings()
    raw = lambda n: pd.read_csv(s.raw_dir / f"{n}.csv", keep_default_na=False)  # noqa: E731
    guests, res, oh = raw("guests"), raw("reservations"), raw("offer_history")
    # ---------------- data side
    seg = guests.segment.value_counts()
    bar("01_guest_segments", "Guest segment distribution", seg.index, seg.values, ylabel="guests")
    ch = res.groupby(["channel_group", "status"]).size().unstack(fill_value=0)
    bar("02_booking_channel_split", "Booking channel split (OTA vs Direct)", list(ch.index), None, ylabel="reservations",
        series={c: ch[c].tolist() for c in ch.columns})
    spend = oh[oh.accepted == 1].groupby("reservation_id").final_price.sum().rename("spend")
    d = res.merge(spend, left_on="reservation_id", right_index=True, how="left").fillna({"spend": 0.0})
    d = d[d.status == "completed"]
    d["adr_bin"] = pd.qcut(d.adr, 8, duplicates="drop")
    g = d.groupby("adr_bin", observed=True).agg(adr=("adr", "mean"), spend=("spend", "mean"))
    bar("03_adr_vs_ancillary_spend", "Mean ancillary spend by ADR band (historical stays)", [f"${a:.0f}" for a in g.adr], g.spend.round(2).tolist(),
        xlabel="average daily rate", ylabel="mean ancillary spend per stay ($)")
    oh["precip_bin"] = pd.cut(oh.precip_prob, [0, .2, .35, .5, .65, 1.0], include_lowest=True)
    oh["indoor"] = np.where(oh.category.isin(["spa", "dining"]), "Spa & dining", "Other offers")
    w = oh.groupby(["precip_bin", "indoor"], observed=True).accepted.mean().unstack()
    xs = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in w.index]
    lines("04_weather_vs_conversion", "Offer acceptance vs precipitation probability",
          {c: (xs, w[c].round(4).tolist()) for c in w.columns}, xlabel="precip probability band", ylabel="acceptance rate")
    # ---------------- model side
    mm = json.loads((s.eval_dir / "model_metrics.json").read_text())
    c = mm["curves"]
    lines("05_roc_curve", f"ROC curve (AUC {mm['roc_auc']}, KS {mm['ks']})", {"XGBoost NBA": (c["roc"]["fpr"], c["roc"]["tpr"])},
          "false positive rate", "true positive rate", diagonal=True)
    lines("06_pr_curve", f"Precision-Recall (PR-AUC {mm['pr_auc']}, base rate {mm['base_rate']})",
          {"XGBoost NBA": (c["pr"]["recall"], c["pr"]["precision"])}, "recall", "precision")
    lines("07_calibration", f"Calibration (Brier {mm['brier']} vs {mm['brier_baseline']} baseline)",
          {"XGBoost NBA": (c["calibration"]["mean_pred"], c["calibration"]["frac_pos"])}, "mean predicted probability", "observed acceptance", diagonal=True)
    fi = mm["feature_importance"][:10]
    bar("08_feature_importance", "Top XGBoost feature importances (gain)", [f["feature"] for f in fi], [f["gain"] for f in fi], xlabel="gain", horizontal=True)
    mp = s.metrics_dir / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        nl = m["node_latency"]
        nodes = [n for n in NODE_ORDER if n in nl]
        bar("09_node_latency", "Node-by-node latency (ms)", nodes, None, ylabel="milliseconds",
            series={"mean": [nl[n]["mean_ms"] for n in nodes], "p95": [nl[n]["p95_ms"] for n in nodes]})
        f = m["funnel"]
        bar("10_guardrail_funnel", "Guardrail outcome funnel", list(f), list(f.values()), xlabel="journeys", horizontal=True)
        r = m["revenue"]
        bar("11_revpar_lift", "Incremental ancillary value per room-night: agent vs static baseline", ["Revenue", "Contribution margin"], None,
            ylabel="$ per room-night (expected)",
            series={"Static baseline": [r["baseline_anc_revpar"], round(r["baseline_exp_margin_usd"] / max(r["room_nights"], 1), 3)],
                    "Agent": [r["agent_anc_revpar"], round(r["agent_exp_margin_usd"] / max(r["room_nights"], 1), 3)]})
        # state transitions from telemetry
        tp = s.logs_dir / "telemetry.jsonl"
        if tp.exists():
            seqs = defaultdict(list)
            for line in tp.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                seqs[rec["thread_id"]].append(rec["node"])
            tr = Counter()
            for seq in seqs.values():
                for a, b in zip(seq, seq[1:]):
                    tr[(a, b)] += 1
            mat = np.array([[tr.get((a, b), 0) for b in NODE_ORDER] for a in NODE_ORDER])
            heat("12_state_transitions", "LangGraph state transitions (observed counts)", mat, NODE_ORDER, NODE_ORDER)
    return sorted(p.name for p in s.charts_dir.glob("*.png"))


if __name__ == "__main__":
    print(generate_all_charts())
