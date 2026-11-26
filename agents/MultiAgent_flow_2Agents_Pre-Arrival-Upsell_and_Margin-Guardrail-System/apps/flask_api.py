"""Flask REST API, PMS webhook receiver, HITL approval queue and operational audit-log UI.

Endpoints
  GET  /health                         service + GPU + LLM status
  GET  /api/v1/scenarios               pre-loaded scenarios
  POST /api/v1/offers/generate         {"scenario": name} | {"reservation_id": "RES-00042"} (+ optional "pms_override", "injections")
  GET  /api/v1/pms                     live PMS inventory + dynamic safety buffers
  POST /webhooks/pms                   {"key": "JUNIOR_OCEAN_VIEW", "available": 1}  (inventory push from the PMS)
  GET  /api/v1/hitl/queue              pending human reviews
  POST /api/v1/hitl/<thread_id>/decision   {"decision": "approve"|"reject", "reviewer": "...", "note": "..."}
  GET  /api/v1/audit/log               recent runs
  GET  /audit                          HTML operational audit UI
"""
from __future__ import annotations
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, jsonify, request, Response  # noqa: E402

from agents.graph import resume_hitl, run_offer  # noqa: E402
from agents.llm_client import get_llm  # noqa: E402
from agents.tools import PMS, SCENARIOS, crm_get_guest  # noqa: E402
from models.utils import gpu_snapshot, load_config, path  # noqa: E402

app = Flask(__name__)
_lock = threading.Lock()
QUEUE_FILE = lambda: path("logs") / "hitl_queue.json"  # noqa: E731


def _load_queue() -> dict:
    p = QUEUE_FILE()
    return json.loads(p.read_text()) if p.exists() else {}


def _save_queue(q: dict) -> None:
    QUEUE_FILE().write_text(json.dumps(q, indent=2, default=str))


def _summary(out: dict) -> dict:
    return {"thread_id": out["thread_id"], "final_status": out.get("final_status"), "revisions": out.get("revision", 0), "draft": out.get("draft"),
            "verdict": out.get("verdict"), "history": out.get("history"), "interrupt": out.get("interrupt")}


@app.get("/health")
def health():
    llm = get_llm()
    return jsonify(status="ok", llm_mode=llm.mode, llm_endpoint=load_config()["llm"]["base_url"], gpu=gpu_snapshot(), max_revisions=load_config()["agents"]["max_revisions"])


@app.get("/api/v1/scenarios")
def scenarios():
    return jsonify({k: {"title": v["title"], "description": v["description"], "reservation_id": v["guest"].reservation_id} for k, v in SCENARIOS.items()})


@app.post("/api/v1/offers/generate")
def generate():
    body = request.get_json(force=True, silent=True) or {}
    try:
        if "scenario" in body:
            sc = SCENARIOS[body["scenario"]]; guest, inj, pms = sc["guest"].model_dump(), sc["injections"], sc["pms_override"]
        elif "reservation_id" in body:
            guest, inj, pms = crm_get_guest(body["reservation_id"]).model_dump(), {}, {}
        else:
            return jsonify(error="provide 'scenario' or 'reservation_id'"), 400
    except KeyError as e:
        return jsonify(error=f"unknown {e}"), 404
    inj = {**inj, **body.get("injections", {})}; pms = {**pms, **body.get("pms_override", {})}
    out = run_offer(guest, inj, pms)
    if out.get("final_status") == "HITL_PENDING":
        with _lock:
            q = _load_queue(); q[out["thread_id"]] = {"thread_id": out["thread_id"], "reservation_id": guest["reservation_id"], "reason": (out.get("interrupt") or {}).get("reason"),
                                                      "draft": out.get("draft"), "status": "PENDING"}; _save_queue(q)
    return jsonify(_summary(out))


@app.get("/api/v1/pms")
def pms_get():
    return jsonify(PMS.snapshot())


@app.post("/webhooks/pms")
def pms_hook():
    b = request.get_json(force=True, silent=True) or {}
    try:
        PMS.update(b["key"], int(b["available"]))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"invalid payload: {e}"), 400
    return jsonify(ok=True, snapshot=PMS.snapshot()[b["key"]])


@app.get("/api/v1/hitl/queue")
def hitl_queue():
    return jsonify([v for v in _load_queue().values() if v.get("status") == "PENDING"])


@app.post("/api/v1/hitl/<thread_id>/decision")
def hitl_decide(thread_id):
    b = request.get_json(force=True, silent=True) or {}
    with _lock:
        q = _load_queue()
        if thread_id not in q or q[thread_id]["status"] != "PENDING":
            return jsonify(error="no such pending review"), 404
        out = resume_hitl(thread_id, b.get("decision", "reject"), b.get("reviewer", "ops"), b.get("note", ""))
        q[thread_id].update(status="RESOLVED", final_status=out.get("final_status")); _save_queue(q)
    return jsonify(final_status=out.get("final_status"), hitl=out.get("hitl"))


def _read_jsonl(name: str, n: int = 200) -> list:
    p = path("logs") / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()]


@app.get("/api/v1/audit/log")
def audit_log():
    return jsonify(runs=_read_jsonl("offer_runs.jsonl"), trace=_read_jsonl("agent_trace.jsonl", 300))


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Upsell Guardrail - Operational Audit</title>
<style>body{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#f6f4ef;color:#1f2430}header{background:#1f2430;color:#f1e6c8;padding:18px 28px}
h1{margin:0;font-size:20px;font-weight:600}main{padding:22px 28px;display:grid;gap:20px}section{background:#fff;border-radius:10px;padding:16px 20px;box-shadow:0 1px 3px #0002}
h2{margin:0 0 10px;font-size:15px;color:#6b5b2e}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
.tag{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}.APPROVED,.HITL_APPROVED{background:#d6f0dc;color:#1b6b2f}.REJECTED,.HITL_REJECTED{background:#f8d9d6;color:#9b2a1f}
.ESCALATE_HITL,.HITL_PENDING{background:#fbe9c5;color:#8a5a00}button{padding:5px 12px;border:0;border-radius:6px;cursor:pointer;margin-right:4px}.ok{background:#2e7d4f;color:#fff}.no{background:#b3402f;color:#fff}
select,input{padding:5px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}#health{font-size:12px;opacity:.85}</style></head><body>
<header><h1>Pre-Arrival Upsell &amp; Margin Guardrail - Operational Audit</h1><div id="health">loading...</div></header><main>
<section><h2>Run an offer through the Strategist / Auditor loop</h2><div class="row"><select id="sc"></select><button class="ok" id="run">Generate</button><span id="runmsg"></span></div></section>
<section><h2>HITL approval queue</h2><div id="queue">...</div></section>
<section><h2>Live PMS inventory (webhook-fed)</h2><div class="row"><select id="pk"></select><input id="pv" type="number" value="1" min="0" style="width:70px"><button class="ok" id="push">POST /webhooks/pms</button></div><div id="pms"></div></section>
<section><h2>Recent runs</h2><div id="runs">...</div></section><section><h2>Agent trace (latest)</h2><div id="trace">...</div></section></main>
<script>
const $=id=>document.getElementById(id);const tag=s=>`<span class="tag ${s}">${s}</span>`;
async function j(u,o){return (await fetch(u,o)).json()}
async function refresh(){
 const h=await j('/health');$('health').textContent=`LLM: ${h.llm_mode} | GPU: ${h.gpu.device} | R_max=${h.max_revisions}`;
 const q=await j('/api/v1/hitl/queue');$('queue').innerHTML=q.length?'<table><tr><th>Thread</th><th>Reservation</th><th>Reason</th><th>Offer</th><th></th></tr>'+q.map(x=>`<tr><td>${x.thread_id}</td><td>${x.reservation_id}</td><td>${x.reason||''}</td><td>${x.draft.sku} $${x.draft.offer_price}</td><td><button class="ok" onclick="dec('${x.thread_id}','approve')">Approve</button><button class="no" onclick="dec('${x.thread_id}','reject')">Reject</button></td></tr>`).join('')+'</table>':'<i>No pending reviews</i>';
 const p=await j('/api/v1/pms');$('pms').innerHTML='<table><tr><th>Key</th><th>Capacity</th><th>Available</th><th>Safety buffer</th></tr>'+Object.entries(p).map(([k,v])=>`<tr><td>${k}</td><td>${v.capacity}</td><td>${v.available}</td><td>${v.buffer}</td></tr>`).join('')+'</table>';
 if(!$('pk').options.length)Object.keys(p).forEach(k=>$('pk').add(new Option(k,k)));
 const a=await j('/api/v1/audit/log');$('runs').innerHTML='<table><tr><th>Time</th><th>Thread</th><th>Reservation</th><th>Status</th><th>Revisions</th><th>SKU</th><th>Price</th></tr>'+a.runs.slice().reverse().slice(0,15).map(r=>`<tr><td>${(r.ts||'').slice(11,19)}</td><td>${r.thread_id}</td><td>${r.reservation_id||''}</td><td>${tag(r.status)}</td><td>${r.revisions??''}</td><td>${r.sku||''}</td><td>${r.price??''}</td></tr>`).join('')+'</table>';
 $('trace').innerHTML='<table><tr><th>Thread</th><th>Agent</th><th>Rev</th><th>Detail</th></tr>'+a.trace.slice().reverse().slice(0,20).map(t=>`<tr><td>${t.thread_id||''}</td><td>${t.agent}</td><td>${t.revision??''}</td><td>${t.decision?tag(t.decision)+' '+(t.violations||[]).join(', '):(t.sku?t.sku+' $'+t.price:(t.final_status||''))}</td></tr>`).join('')+'</table>';}
async function dec(t,d){await j(`/api/v1/hitl/${t}/decision`,{method:'POST',body:JSON.stringify({decision:d,reviewer:'ops-ui'})});refresh()}
$('run').onclick=async()=>{$('runmsg').textContent='running...';const r=await j('/api/v1/offers/generate',{method:'POST',body:JSON.stringify({scenario:$('sc').value})});$('runmsg').innerHTML=tag(r.final_status)+` after ${r.revisions} revision(s)`;refresh()};
$('push').onclick=async()=>{await j('/webhooks/pms',{method:'POST',body:JSON.stringify({key:$('pk').value,available:+$('pv').value})});refresh()};
j('/api/v1/scenarios').then(s=>{Object.entries(s).forEach(([k,v])=>$('sc').add(new Option(v.title,k)));});refresh();setInterval(refresh,8000);
</script></body></html>"""


@app.get("/")
@app.get("/audit")
def audit_ui():
    return Response(_PAGE, mimetype="text/html")


def main():
    c = load_config()["flask"]; app.run(host=c["host"], port=c["port"], debug=False, threaded=True)


if __name__ == "__main__":
    main()
