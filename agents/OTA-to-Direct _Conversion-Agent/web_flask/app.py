"""Flask REST API + interactive web UI: booking simulator, step flow, guardrail inspector, evaluation viewer."""
from __future__ import annotations
import json, sys
from functools import lru_cache
from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings, detect_hardware, load_catalog
from agent.llm_client import endpoint

app = Flask(__name__)
S = get_settings()


def _clean(o):
    """Strict JSON: NaN/Inf -> null (browsers reject the NaN token)."""
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))): return None
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    return o


def J(obj, code=200):
    return Response(json.dumps(_clean(obj), default=str, allow_nan=False), status=code, mimetype="application/json")


def _read(*parts, default=None):
    p = S.out.joinpath(*parts)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tail_jsonl(name, n=50):
    p = S.out / "logs" / name
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()]


@lru_cache
def _bookings():
    from data.synthetic_generator import load
    return load()[0]


def sample_webhook(i: int, window: str = "pre_arrival") -> dict:
    r = _bookings().iloc[int(i) % len(_bookings())]
    l4 = r.phone_last4 if isinstance(r.phone_last4, str) else None
    return {"booking_id": r.booking_id, "ota": r.ota, "first_name": r.first_name, "last_name": r.last_name, "masked_email": r.masked_email, "phone_last4": l4,
            "country": r.country, "checkin_date": r.checkin_date, "los": int(r.los), "adr": float(r.adr), "ota_commission": float(r.ota_commission),
            "lead_days": int(r.lead_days), "trip_purpose": r.trip_purpose, "free_cancellation": bool(r.free_cancellation), "ota_public_rate": float(r.adr),
            "direct_public_rate": float(r.adr), "window": window, "is_ghost_holdout": bool(r.is_ghost_holdout)}


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return render_template("index.html", catalog=load_catalog()["offers"])


@app.get("/evaluation")
def evaluation():
    return render_template("evaluation.html")


# ------------------------------------------------------------------ API
@app.get("/api/health")
def health():
    ep = endpoint(); return J({"status": "ok", "llm": ep, "hardware": detect_hardware(), "models_promoted": (_read("models", "promotion.json") or {}).get("promoted")})


@app.get("/api/sample")
def api_sample():
    import random
    i = request.args.get("i", type=int); i = random.randrange(len(_bookings())) if i is None else i
    return J(sample_webhook(i, request.args.get("window", "pre_arrival")))


@app.get("/api/scenarios")
def api_scenarios():
    from agent.scenarios import build_scenarios
    return J(build_scenarios())


@app.post("/api/simulate")
def api_simulate():
    from agent.graph import run_agent
    wh = request.get_json(force=True, silent=True) or {}
    try:
        st = run_agent(wh)
    except Exception as e:
        return J({"error": f"{type(e).__name__}: {e}"}, 400)
    return J({k: st.get(k) for k in ("run_id", "error", "resolved_profile", "resolution", "saved_commission_usd", "is_holdout", "scores", "strategy", "llm_mode",
                                      "llm_attempts", "guardrail", "rejected_offers", "guardrail_retries", "final_offer_id", "downgraded", "dispatch", "attribution", "gates", "trace")})


@app.get("/api/audit/decisions")
def api_decisions(): return J(_tail_jsonl("agent_decisions.jsonl", request.args.get("n", 50, type=int)))


@app.get("/api/audit/guardrails")
def api_guardrails(): return J(_tail_jsonl("guardrail_audits.jsonl", request.args.get("n", 50, type=int)))


@app.get("/api/validation")
def api_validation(): return J({"data": _read("validation", "data_validation.json", default={}), "steps": _read("validation", "step_gates.json", default={})})


@app.get("/api/metrics")
def api_metrics():
    import pandas as pd
    p = S.out / "metrics" / "model_metrics.csv"
    return J({"model_metrics": pd.read_csv(p).to_dict("records") if p.exists() else [], "latency": _read("metrics", "latency_gpu.json", default={}),
              "hardware": _read("logs", "hardware_telemetry.json", default={})})


@app.get("/api/evaluation")
def api_evaluation(): return J({"replay": _read("evaluation", "offline_replay_roi.json", default={}), "vlm": _read("evaluation", "vlm_visual_audit.json", default={}),
                                "e2e": _read("evaluation", "e2e_report.json", default={})})


@app.get("/api/charts")
def api_charts(): return J(sorted(p.name for p in (S.out / "charts").glob("*.png")))


@app.get("/charts/<path:name>")
def charts(name): return send_from_directory(S.out / "charts", name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=S.FLASK_PORT, debug=False, threaded=True)
