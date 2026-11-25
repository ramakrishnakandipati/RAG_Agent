"""Programmatic, publication-grade diagrams (matplotlib patches): architecture, data lineage, LangGraph state machine."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings

INK, BLUE, GREEN, AMBER, RED, GRAY = "#1b2333", "#2a6fdb", "#1f9d6b", "#e3a008", "#d1495b", "#8a94a6"
FILL = {"blue": "#e8f0fd", "green": "#e5f6ef", "amber": "#fdf3d8", "red": "#fbe7ea", "gray": "#eef0f4"}
EDGE = {"blue": BLUE, "green": GREEN, "amber": AMBER, "red": RED, "gray": GRAY}
WHITE = dict(fc="white", ec="none", pad=1.5)


def canvas(w=16, h=9, title=""):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(0.3, h - 0.4, title, fontsize=19, fontweight="bold", color=INK, va="center"); return fig, ax


def box(ax, x, y, w, h, text, color="blue", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=FILL[color], ec=EDGE[color], lw=1.6))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, color=GRAY, style="-|>", rad=0.0, ls="-", lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def gate(ax, x, y, label, color=AMBER):
    ax.add_patch(Polygon([(x, y + .32), (x + .32, y), (x, y - .32), (x - .32, y)], fc=FILL["amber"], ec=color, lw=1.6))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)


def band(ax, y, h, label, w=15.4):
    ax.add_patch(FancyBboxPatch((0.3, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec="#d6dbe5", lw=1.2))
    ax.text(0.5, y + h - 0.2, label, fontsize=10, color=GRAY, fontweight="bold", va="center")


def architecture():
    fig, ax = canvas(16, 10, "OTA-to-Direct Conversion Agent - System Architecture")
    band(ax, 8.0, 1.2, "1  INGESTION"); box(ax, 0.7, 8.15, 2.6, .7, "Expedia webhook", "gray"); box(ax, 3.5, 8.15, 2.6, .7, "Booking.com webhook", "gray"); box(ax, 6.3, 8.15, 2.6, .7, "Agoda webhook", "gray")
    box(ax, 10.2, 8.15, 2.8, .7, "BookingWebhookSchema\n+ synthetic PMS generator", "blue"); gate(ax, 14.0, 8.5, "D1")
    band(ax, 6.2, 1.5, "2  IDENTITY RESOLUTION & FEATURE STORE (GPU)")
    box(ax, 0.7, 6.35, 3.4, .8, "Deterministic keys\n(real email)", "blue"); box(ax, 4.4, 6.35, 3.6, .8, "FAISS blocking + CUDA cosine\n+ Jaro-Winkler rerank", "green", bold=True)
    box(ax, 8.3, 6.35, 3.2, .8, "CDP profiles\n(tier, LTV, consent, fatigue)", "gray"); box(ax, 11.8, 6.35, 3.0, .8, "Feature vector (19)", "blue"); gate(ax, 15.3, 6.75, "D2")
    band(ax, 4.4, 1.5, "3  GPU ML SCORING")
    box(ax, 0.7, 4.55, 4.3, .8, "XGBoost X-/T-Learner uplift\ntau = CATE (device=cuda)", "green", bold=True); box(ax, 5.3, 4.55, 4.0, .8, "XGBoost cancel risk\n+ isotonic calibration", "green", bold=True)
    box(ax, 9.6, 4.55, 3.6, .8, "Promotion gate\nQini>0, AUC>=0.72", "amber"); gate(ax, 14.0, 4.95, "D3")
    band(ax, 2.0, 2.2, "4  LANGGRAPH ORCHESTRATOR")
    names = ["1 ingest_\nand_resolve", "2 score_\nguest", "3 select_\nstrategy", "4 apply_\nguardrails", "5 compose_\nand_dispatch", "6 log_and_\nattribute"]
    for i, n in enumerate(names):
        box(ax, 0.6 + i * 2.55, 2.25, 2.1, .85, n, "blue", fs=9.5)
        if i < 5: arrow(ax, (2.7 + i * 2.55, 2.67), (3.15 + i * 2.55, 2.67))
    box(ax, 5.1, 3.3, 3.3, .65, "Local qwen3-vl-30b-a3b-instruct\n(strict JSON schema)", "amber", fs=8.8); arrow(ax, (6.75, 3.3), (6.75, 3.1), AMBER)
    box(ax, 8.7, 3.3, 2.9, .65, "Policy engine (deterministic)\nparity | margin | fatigue", "red", fs=8.8, bold=True); arrow(ax, (9.3, 3.3), (9.3, 3.1), RED)
    ax.text(15.3, 3.65, "validation gate per node", fontsize=9, color=GRAY, ha="right")
    band(ax, 0.2, 1.6, "5  INTERFACES & EVALUATION")
    box(ax, 0.7, .35, 3.4, .8, "Flask REST + Web UI", "blue"); box(ax, 4.4, .35, 3.4, .8, "Streamlit Ops Dashboard", "blue"); box(ax, 8.1, .35, 3.6, .8, "Offline replay vs 15% holdout", "green")
    box(ax, 12.0, .35, 3.4, .8, "VLM chart auditor\n(image_url -> qwen3-vl)", "amber")
    for x in (5.0, 8.5, 11.0): arrow(ax, (x, 8.0), (x, 7.7))
    fig.savefig(get_settings().path("charts", "system_architecture_diagram.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)


def data_flow():
    s = get_settings(); dv = {}
    p = s.path("validation", "data_validation.json")
    if p.exists(): dv = json.loads(p.read_text())
    rows = dv.get("D1", {}).get("rows", 0); m = dv.get("D2", {}).get("metrics", {})
    ho = int(round(rows * s.HOLDOUT_FRAC)); fig, ax = canvas(16, 8.6, "Data Pipeline - Lineage, Record Counts & Holdout Split")
    W, H = 3.2, 1.3
    box(ax, 0.5, 6.0, W, H, f"PMS bookings\n{rows:,} rows\nExpedia / Booking.com / Agoda", "gray", bold=True, fs=10)
    box(ax, 4.6, 6.0, W, H, "Gate D1\nschema + integrity\ncommission .10-.30\nADR>0, LOS>=1", "amber", bold=True, fs=9)
    box(ax, 8.7, 6.0, W, H, f"Identity resolution (Gate D2)\nmatch rate {m.get('match_rate', 0):.1%}\nrecall {m.get('recall', 0):.1%} | FP merge {m.get('false_positive_merge_rate', 0):.2%}", "green", bold=True, fs=9.5)
    box(ax, 12.8, 6.0, 2.8, H, "Feature store\n19 features\nzero-NaN contract", "blue", bold=True, fs=10)
    arrow(ax, (3.7, 6.65), (4.6, 6.65)); arrow(ax, (7.8, 6.65), (8.7, 6.65)); arrow(ax, (11.9, 6.65), (12.8, 6.65))
    box(ax, 0.5, 3.4, W, H, f"Treated (85%)\n{rows - ho:,} bookings\nreceive agent actions", "green", bold=True, fs=10)
    box(ax, 4.6, 3.4, W, H, f"Ghost holdout (15% +/-1%)\n{ho:,} bookings\nstratified by OTA\nnever messaged", "red", bold=True, fs=9)
    box(ax, 8.7, 3.4, W, H, "CDP profiles\n30% of N: tier, LTV, consent,\nfatigue counters", "gray", bold=True, fs=9.5)
    box(ax, 12.8, 3.4, 2.8, H, "Train 56% | Val 14%\nTest 30%\nstratified by treatment", "blue", bold=True, fs=10)
    arrow(ax, (2.1, 6.0), (2.1, 4.7)); arrow(ax, (6.2, 6.0), (6.2, 4.7), RED, ls="--"); arrow(ax, (10.3, 4.7), (10.3, 6.0)); arrow(ax, (14.2, 6.0), (14.2, 4.7))
    box(ax, 12.8, 0.8, 2.8, H, "Uplift X-learner\n+ cancel model\nGate D3", "green", bold=True, fs=10)
    box(ax, 8.7, 0.8, W, H, "LangGraph agent\ndecisions + gate logs\n(JSONL)", "blue", bold=True, fs=10)
    box(ax, 3.1, 0.8, 4.0, H, "Offline replay\ntreated vs holdout\nlift, ROI, Qini", "amber", bold=True, fs=10)
    arrow(ax, (14.2, 3.4), (14.2, 2.1)); arrow(ax, (12.8, 1.45), (11.9, 1.45)); arrow(ax, (8.7, 1.45), (7.1, 1.45))
    arrow(ax, (2.1, 3.4), (3.9, 2.1)); arrow(ax, (6.2, 3.4), (5.9, 2.1), RED, ls="--")
    ax.text(6.35, 2.75, "control arm", fontsize=9, color=RED)
    fig.savefig(s.path("charts", "data_pipeline_erd_flow.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)


def state_machine():
    fig, ax = canvas(16, 8.2, "LangGraph State Machine - 6 Nodes, Validation Gates & Retry Loop")
    names = ["ingest_and_\nresolve", "score_\nguest", "select_\nstrategy", "apply_\nguardrails", "compose_and_\ndispatch", "log_and_\nattribute"]
    sub = ["PMS payload -> CDP ID\nsaved_commission > 0", "features, tau, P(cancel)\nCUDA/CPU latency", "qwen3-vl strict JSON\n<=2 LLM retries", "parity | margin | fatigue\nENRD > 0", "CRS inventory, gated link\nholdout suppressed", "decisions + attribution\nJSONL logs"]
    xs = [0.6 + i * 2.55 for i in range(6)]
    for i, (n, sb, x) in enumerate(zip(names, sub, xs)):
        col = "red" if i == 3 else ("amber" if i == 2 else "blue")
        box(ax, x, 4.0, 2.2, 1.9, "", col); ax.text(x + 1.1, 5.35, f"{i + 1}. {n}", ha="center", va="center", fontsize=10, fontweight="bold", color=INK, linespacing=1.2)
        ax.text(x + 1.1, 4.5, sb, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.3)
        gate(ax, x + 1.1, 6.55, f"G{i + 1}"); arrow(ax, (x + 1.1, 6.23), (x + 1.1, 5.92), AMBER, lw=1.2)
        if i < 5: arrow(ax, (x + 2.2, 4.95), (x + 2.55, 4.95), BLUE)
    ax.text(0.6, 7.05, "G = pre-condition + post-condition validation gate per node (logged to outputs/validation/step_gates.json)", fontsize=9.5, color=GRAY)
    arrow(ax, (xs[3] + 1.1, 3.98), (xs[2] + 1.1, 3.98), RED, rad=-0.9, ls="--", lw=2)
    ybus = 1.75
    for i in range(5): ax.plot([xs[i] + 1.1] * 2, [3.98, ybus], color=GRAY, ls=":", lw=1.1)
    ax.plot([xs[0] + 1.1, xs[5] + 1.1], [ybus] * 2, color=GRAY, ls=":", lw=1.1); arrow(ax, (xs[5] + 1.1, ybus), (xs[5] + 1.1, 3.98), GRAY, ls=":", lw=1.1)
    ax.text((xs[2] + xs[3]) / 2 + 1.1, 2.75, "policy violation -> retry select_strategy (max 2),\nthen deterministic downgrade to best compliant offer / NO_OFFER",
            fontsize=9, color=RED, ha="center", linespacing=1.4, bbox=WHITE, zorder=5)
    ax.text(8.0, 1.3, "dotted bus: any failed gate short-circuits to log_and_attribute (error recorded, nothing dispatched)", ha="center", fontsize=9.5, color=GRAY)
    box(ax, 0.5, 0.2, 1.4, .55, "START", "gray", bold=True); arrow(ax, (1.0, .78), (1.0, 4.0), GRAY)
    box(ax, 14.5, 0.2, 1.4, .55, "END", "gray", bold=True); arrow(ax, (15.2, 4.0), (15.2, .78), GRAY)
    fig.savefig(get_settings().path("charts", "langgraph_state_machine.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)


def run():
    architecture(); data_flow(); state_machine(); print("diagrams written to", get_settings().path("charts", "x").parent)


if __name__ == "__main__":
    run()
