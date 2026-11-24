"""Compile README.md from the files the pipeline actually produced (metrics/, evaluation/, validation/, outputs/). No hardcoded numbers."""
from __future__ import annotations
import json
from datetime import datetime

from src.config import get_settings

MERMAID_STATE = """```mermaid
flowchart TD
    START([Trigger: pre-arrival / inbound reply]) --> N1
    N1["1 enrich_context<br/>PMS + CRM + weather/events"] -->|pre/post validators| N2
    N1 -.->|FAIL_VALIDATION| END1([END: error])
    N2["2 retrieve_knowledge_and_offers<br/>Chroma+BM25 RRF (GPU) + XGBoost (GPU) + RMS rules"] -->|offers found| N3
    N2 -.->|NO_OFFER| END2([END: no send])
    N3["3 draft_message<br/>Qwen3-VL copy with placeholders"] --> N4
    N4{"4 compliance_check<br/>price / discount / SKU / tone / caps"}
    N4 -->|PASS| N5
    N4 -->|"FAIL_POLICY (retry <= MAX_RETRIES)"| N3
    N4 -->|"FAIL_POLICY (retries exhausted)"| N5
    N4 -.->|"FAIL_VALIDATION: opt-out, cap, quiet hours, no inventory"| END3([END: blocked])
    N5{"5 hitl_gate<br/>VIP custom / negative sentiment / repeated failure"}
    N5 -->|no risk| N6
    N5 -->|"interrupt(): ESCALATE_HITL"| Q[(Human concierge queue)]
    Q -->|approve / edit| N6
    Q -.->|reject| END4([END: rejected])
    N6["6 dispatch_and_log<br/>final token re-scan, send, folio booking, audit"] --> END5([SENT / BOOKED])
```"""

MERMAID_DATA = """```mermaid
flowchart LR
    subgraph Sources
      PMS[(PMS reservations)]; CRM[(CRM Guest360)]; RMS[(RMS inventory & occupancy)]; WX[(Weather & events)]; KB[/Knowledge base md+json/]
    end
    PMS & CRM & WX & RMS --> VAL{{data_validator<br/>schema, nulls, ranges, FK, PSI}}
    VAL --> FEAT[Feature table: offer_history]
    FEAT --> XGB[XGBoost NBA<br/>tree_method=hist, device=cuda]
    KB --> CH[(ChromaDB dense<br/>bge-large fp16 on CUDA)]
    KB --> BM[BM25 sparse]
    CH & BM --> RRF[RRF fusion + cross-encoder rerank]
    XGB --> OFF[Ranked offers]
    RMS --> RULES[RMS rules: margin floor, discount cap, inventory]
    RULES --> OFF
    OFF --> LLM[Qwen3-VL: copy with placeholders]
    RRF --> LLM
    LLM --> GATE[Deterministic compliance gate]
    OFF --> GATE
    GATE --> OUT[Guest message / folio / audit]
```"""

MERMAID_SEQ = """```mermaid
sequenceDiagram
    participant G as Guest
    participant API as Flask API
    participant WF as LangGraph
    participant PMS as PMS / Inventory
    participant H as Human concierge
    G->>API: POST /api/v1/reply (text)
    API->>WF: run(inbound_reply)
    WF->>PMS: Guest360, stay, last offer
    WF->>WF: classify intent + sentiment (Qwen3-VL + rules)
    alt accept and inventory free
        WF->>WF: draft confirmation, compliance gate PASS
        WF->>PMS: atomic inventory check + folio posting
        PMS-->>WF: booked
        WF-->>G: confirmation message
    else complaint or Platinum custom request or repeated policy failure
        WF->>H: interrupt() into HITL queue
        H->>API: POST /api/v1/hitl/approve
        API->>WF: resume(decision)
        WF-->>G: approved message (final re-scan) or no send if rejected
    else sold out
        WF-->>G: apology, nothing charged
    end
```"""


def _load(p):
    if not p.exists():
        raise FileNotFoundError(f"readme_updater needs {p} - run the pipeline first (python run_all.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def _table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _dictionary() -> str:
    from src.validation.data_validator import ENUMS, RANGES, SCHEMAS
    desc = {"guests": "CRM Guest360 profile", "reservations": "PMS reservation", "weather": "Daily forecast", "occupancy": "RMS daily occupancy/ADR",
            "inventory": "Ancillary capacity per SKU/day"}
    parts = []
    for t, cols in SCHEMAS.items():
        rows = []
        for c, k in cols.items():
            rule = []
            if c in RANGES.get(t, {}): rule.append(f"range {RANGES[t][c]}")
            if c in ENUMS.get(t, {}): rule.append("in " + "/".join(ENUMS[t][c]))
            if c.endswith("_id") and c in ("guest_id", "reservation_id") and t in ("guests", "reservations") and c == {"guests": "guest_id", "reservations": "reservation_id"}[t]:
                rule.append("primary key, unique")
            if t == "reservations" and c == "guest_id": rule.append("FK -> guests")
            rows.append([c, {"int": "integer", "float": "float", "bool": "boolean", "object": "string"}[k], "not null", "; ".join(rule) or "-"])
        parts.append(f"**`{t}`** - {desc[t]}\n\n" + _table(rows, ["column", "type", "constraint", "validation rule"]))
    parts.append("**`offer_history`** - historical offers with outcome (XGBoost training table)\n\n" + _table(
        [["category, segment, loyalty_tier, channel_group, room_type", "categorical", "not null", "enum, consistent with generator levels"],
         ["adr, nights, party_size, lead_days, ltv_usd, total_stays, ota_ratio", "numeric", "not null", "ranges as above"],
         ["precip_prob, temp_c, has_event, occupancy_pct", "numeric", "not null", "precip_prob in [0,1]"],
         ["price, discount_pct, final_price", "float", "not null", "final_price = price x (1 - discount_pct/100)"],
         ["accepted", "int", "not null", "label in {0,1}"]], ["column(s)", "type", "constraint", "validation rule"]))
    return "\n\n".join(parts)


def update_readme() -> str:
    s = get_settings()
    m = _load(s.metrics_dir / "metrics.json")
    vr = _load(s.validation_dir / "data_validation_report.json")
    ti = _load(s.eval_dir / "training_info.json")
    mm = m["model"]
    rev = m["revenue"]
    charts = sorted(s.charts_dir.glob("*.png"))
    shots = sorted(s.screens_dir.glob("*.png"))
    gpu = m.get("gpu", {})
    llm_src = ", ".join(f"{k}: {v}" for k, v in m["llm_sources"].items())
    funnel = _table([[k, v] for k, v in m["funnel"].items()], ["outcome", "journeys"])
    lat = _table([[n, v["mean_ms"], v["p50_ms"], v["p95_ms"], v["n"]] for n, v in m["node_latency"].items()],
                 ["node", "mean ms", "p50 ms", "p95 ms", "calls"])
    hitl = ", ".join(f"`{k}` x{v}" for k, v in m["hitl_reasons"].items()) or "none in this run"
    fi = _table([[f["feature"], f["gain"]] for f in mm["feature_importance"][:8]], ["feature", "gain"])

    def img(p, w=520):
        return f'<img src="outputs/charts/{p.name}" width="{w}" alt="{p.stem}">'

    chart_html = "\n".join(f"<p><b>{p.stem}</b> &middot; <a href=\"outputs/charts/{p.stem}.html\">interactive</a><br>{img(p, 640)}</p>" for p in charts)
    shot_html = "\n".join(f"<p><b>{p.stem}</b><br><img src=\"outputs/ui_screens/{p.name}\" width=\"900\" alt=\"{p.stem}\"></p>" for p in shots) \
        or "_No screenshots yet - run `python -m src.visualization.capture_ui_screens`._"
    ok_hall = m["hallucination_rate_pct"] == 0.0
    md = f"""# GPU-Accelerated Dynamic Pre-Arrival & Revenue Optimization Agent

_Auto-generated by `src/utils/readme_updater.py` on {datetime.now():%Y-%m-%d %H:%M} from the files in `metrics/`, `evaluation/`, `validation/` and `outputs/`. Do not edit by hand._

A bounded **LangGraph** state-machine agent for a (synthetic) resort, **Azure Bay Resort & Spa**. Probabilistic reasoning (Qwen3-VL copywriting and intent classification) is strictly separated from deterministic business logic (GPU XGBoost propensity + RMS inventory/margin/discount rules). Every number a guest sees comes from the RMS payload and is verified by a deterministic gate before sending.

## Verified results (latest run, seed {m['seed']})

| Metric | Value |
|---|---|
| Guest journeys replayed | **{m['n_journeys']}** ({m['n_crashed']} crashed) in {m['wall_seconds']} s |
| Messages sent / judged | {m['messages_sent']} / {m['messages_judged']} |
| **Price / discount / SKU hallucination rate** | **{m['hallucination_rate_pct']}%** ({m['hallucinated_messages']} of {m['messages_judged']}) {'OK' if ok_hall else 'FAILED'} |
| Adversarial drafts injected / caught / reached a guest | {m['adversarial_injected']} / {m['adversarial_caught']} / {m['adversarial_sent_to_guest']} |
| Inbound replies / accepted / booked to folio / sold-out | {m['inbound_replies']} / {m['inbound_accepts']} / {m['inbound_booked']} / {m['inbound_sold_out']} |
| Avg brand-tone score (1-5, {', '.join(f'{k}: {v}' for k, v in m['tone_source'].items())}) | {m['avg_tone_score']} |
| XGBoost ROC-AUC / PR-AUC / KS / Brier | {mm['roc_auc']} / {mm['pr_auc']} / {mm['ks']} / {mm['brier']} (baseline {mm['brier_baseline']}) |
| XGBoost compute | `tree_method={mm['tree_method']}`, `device={mm['device']}`, best iteration {ti['best_iteration']}, top-decile lift {mm['top_decile_lift']}x |
| Expected ancillary revenue per room-night: agent vs static baseline | ${rev['agent_anc_revpar']} vs ${rev['baseline_anc_revpar']} ({rev['anc_revpar_lift_pct']:+}%) |
| Expected contribution margin: agent vs baseline | ${rev['agent_exp_margin_usd']} vs ${rev['baseline_exp_margin_usd']} ({rev['margin_lift_pct']:+}%) |
| Realized ancillary revenue (sampled acceptances): agent vs baseline | ${rev['agent_realized_usd']} vs ${rev['baseline_realized_usd']} ({rev['realized_lift_pct']}%) |
| Est. OTA commission avoided via loyalty conversions | ${rev['est_ota_commission_avoided_usd']} ({m['loyalty_offers_sent']} loyalty offers sent) |
| Data validation | {vr['n_passed']}/{vr['n_checks']} checks passed |
| Hardware | {gpu.get('name', 'CPU')} ({gpu.get('total_vram_gb', '-')} GB), torch {gpu.get('torch', '-')} |
| Draft sources | {llm_src} |

> **How to read the revenue numbers.** They are computed on *synthetic* data against a hidden guest-behaviour model the XGBoost model never sees. Basis: {rev['basis']}. They demonstrate the mechanics and the guardrails, not a real-world uplift claim. The agent also withholds offers (opt-out, frequency cap, quiet hours, sold-out), which lowers raw volume versus a blast-everyone baseline - see the funnel below.

### Guardrail funnel
{funnel}

HITL escalation reasons observed: {hitl}.

### Node latency
{lat}

### XGBoost top features
{fi}

## Architecture

### (a) LangGraph state machine with validation gates and retry loops
{MERMAID_STATE}

### (b) Data pipeline and feature lineage
{MERMAID_DATA}

### (c) Two-way guest reply and HITL approval
{MERMAID_SEQ}

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DATA_FLOW_AND_SCHEMAS.md`](docs/DATA_FLOW_AND_SCHEMAS.md), [`docs/EVALUATION_METHODOLOGY.md`](docs/EVALUATION_METHODOLOGY.md).

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && playwright install chromium
cp .env.example .env          # LLM_BASE_URL defaults to LM Studio at http://localhost:1234/v1, model qwen3-vl-30b-a3b-instruct
python run_all.py             # data -> validation -> GPU train -> replay -> charts -> UI screenshots -> README
python run_all.py --mock-llm  # no LLM server needed (deterministic templates)
pytest -q                     # tests (mock LLM); add -m live_llm for the local-Qwen smoke test
```

Services: `python -m src.api.flask_app` (port {s.flask_port}) and `streamlit run src/ui/streamlit_app.py --server.port {s.streamlit_port}`.

## UI walkthrough (Streamlit Control Center)
1. **Live Trigger** - choose an upcoming reservation, press *Run pre-arrival workflow*. The offer table shows the RMS price; the message and the guardrail check table appear below.
2. Set *Adversarial test hook* to `once` to inject a fabricated-price draft: the gate rejects it and the retry loop produces a clean one. `persistent` forces escalation.
3. Type a guest reply (e.g. "Yes please, add it to my booking!") and press *Send inbound reply*: inventory is checked live and the item is posted to the folio. Try "This is terrible, I want a refund" for a negative-sentiment escalation.
4. **State Trace** - node-by-node statuses and latencies; repeated draft/compliance pairs are the retry loop.
5. **HITL Queue** - review, edit, approve or reject escalations; approval resumes the paused graph from its SQLite checkpoint.
6. **Data Lineage** - lineage graph, raw table explorer and validation results.
7. **Metrics & Logs** - live metrics from `/api/v1/metrics` plus structured JSONL logs.

### Screenshots
{shot_html}

## Charts
{chart_html}

## REST API
| Endpoint | Purpose |
|---|---|
| `POST /api/v1/trigger` | `{{reservation_id, trigger_type, adversarial}}` run the workflow |
| `POST /api/v1/reply` | `{{reservation_id, text}}` inbound guest reply |
| `GET /api/v1/hitl/pending` | pending human-review items |
| `POST /api/v1/hitl/approve` | `{{thread_id, decision: approve|reject, reviewer, edited_body?}}` resume paused graph |
| `GET /api/v1/metrics` | latest `metrics/metrics.json` |
| `GET /api/v1/health` | LLM, GPU and PMS status |

Invalid payloads return HTTP 422 with Pydantic error details.

## Data dictionary
Validated by `src/validation/data_validator.py` (report: `validation/data_validation_report.json`; PSI drift: {vr['psi']}).

{_dictionary()}

## Repository layout
`data/` generator + datasets + knowledge base - `src/tools` PMS/CRM, RAG, XGBoost+RMS, compliance - `src/graph` nodes/edges/workflow - `src/validation` data + step validators - `src/evaluation` model eval, judge, replay - `src/visualization` charts + Playwright capture - `src/api`, `src/ui` - `tests/`.
"""
    (s.root / "README.md").write_text(md, encoding="utf-8")
    return str(s.root / "README.md")


if __name__ == "__main__":
    print(update_readme())
