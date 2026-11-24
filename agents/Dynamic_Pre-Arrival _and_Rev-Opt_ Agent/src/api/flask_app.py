"""Flask REST API with Pydantic request/response validation."""
from __future__ import annotations
import json
import threading

from flask import Flask, jsonify, request
from pydantic import ValidationError

from src.config import get_settings
from src.graph.workflow import resume_workflow, run_workflow
from src.llm.client import get_llm
from src.scenarios import BY_ID, SCENARIOS, run_scenario, to_response as _resp
from src.schemas.domain import ApproveRequest, ReplyRequest, TriggerRequest
from src.tools import pms_crm_tools as pms
from src.utils import device as dev

_lock = threading.Lock()   # serialise graph runs (single SQLite checkpointer connection)


def create_app() -> Flask:
    app = Flask(__name__)
    pms.init_pms()

    @app.errorhandler(ValidationError)
    def _val(e):
        return jsonify({"error": "validation_error", "details": json.loads(e.json())}), 422

    @app.get("/api/v1/health")
    def health():
        llm = get_llm()
        with pms.connect() as con:
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("guests", "reservations", "folio", "outbox", "hitl_queue")}
        return jsonify({"status": "ok", "llm_mode": llm.mode, "llm_model": get_settings().llm_model, "llm_up": llm.is_up(),
                        "gpu": dev.gpu_info(), "pms": counts})

    @app.post("/api/v1/trigger")
    def trigger():
        req = TriggerRequest.model_validate(request.get_json(force=True, silent=True) or {})
        if not pms.get_stay(req.reservation_id):
            return jsonify({"error": "reservation_not_found", "reservation_id": req.reservation_id}), 404
        with _lock:
            res = run_workflow(req.reservation_id, trigger_type=req.trigger_type, adversarial=req.adversarial)
        return jsonify(_resp(res))

    @app.post("/api/v1/reply")
    def reply():
        req = ReplyRequest.model_validate(request.get_json(force=True, silent=True) or {})
        if not pms.get_stay(req.reservation_id):
            return jsonify({"error": "reservation_not_found", "reservation_id": req.reservation_id}), 404
        with _lock:
            res = run_workflow(req.reservation_id, trigger_type="inbound_reply", inbound_text=req.text)
        return jsonify(_resp(res))

    @app.get("/api/v1/hitl/pending")
    def hitl_pending():
        return jsonify({"pending": pms.list_hitl("PENDING")})

    @app.post("/api/v1/hitl/approve")
    def hitl_approve():
        req = ApproveRequest.model_validate(request.get_json(force=True, silent=True) or {})
        pend = {p["thread_id"] for p in pms.list_hitl("PENDING")}
        if req.thread_id not in pend:
            return jsonify({"error": "thread_not_pending", "thread_id": req.thread_id}), 404
        with _lock:
            res = resume_workflow(req.thread_id, {"decision": req.decision, "reviewer": req.reviewer, "edited_body": req.edited_body})
        return jsonify(_resp(res))

    @app.get("/api/v1/metrics")
    def metrics():
        s = get_settings()
        p = s.metrics_dir / "metrics.json"
        if not p.exists():
            return jsonify({"error": "metrics_not_generated", "hint": "run python run_all.py"}), 404
        return jsonify(json.loads(p.read_text()))

    @app.get("/api/v1/scenarios")
    def scenarios():
        return jsonify({"scenarios": [{k: v for k, v in sc.items() if k != "steps"} | {"n_steps": len(sc["steps"])} for sc in SCENARIOS]})

    @app.post("/api/v1/scenarios/<sid>/run")
    def scenario_run(sid):
        if sid not in BY_ID:
            return jsonify({"error": "unknown_scenario", "scenario": sid}), 404
        with _lock:
            return jsonify(run_scenario(sid))

    @app.get("/api/v1/reservations")
    def reservations():
        return jsonify({"reservations": pms.list_upcoming_reservations(int(request.args.get("limit", 100)))})

    @app.get("/api/v1/folio/<rid>")
    def folio(rid):
        return jsonify({"folio": pms.get_folio(rid)})

    @app.get("/api/v1/outbox")
    def outbox():
        with pms.connect() as con:
            rows = [dict(r) for r in con.execute("SELECT * FROM outbox ORDER BY id DESC LIMIT ?", (int(request.args.get("limit", 20)),))]
        return jsonify({"outbox": rows})

    return app


if __name__ == "__main__":
    s = get_settings()
    create_app().run(host=s.flask_host, port=s.flask_port, threaded=True)
