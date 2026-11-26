"""Automated E2E verification: boots Flask + Streamlit, exercises the REST/webhook/HITL API, and captures Playwright screenshots.

Run:  python scripts/verify_and_capture.py          (exit code 1 if any check fails)
Outputs: outputs/screenshots/*.png, outputs/metrics/verification_report.json
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.utils import load_config, path  # noqa: E402

CFG = load_config(); FLASK = f"http://{CFG['flask']['host']}:{CFG['flask']['port']}"; STREAMLIT = f"http://localhost:{CFG['streamlit']['port']}"
SHOTS = ROOT / "outputs" / "screenshots"; SHOTS.mkdir(parents=True, exist_ok=True)
results: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}"); return bool(ok)


def wait(url: str, timeout=120) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def api_checks():
    r = requests.get(f"{FLASK}/health").json(); check("GET /health", r["status"] == "ok", f"llm={r['llm_mode']} gpu={r['gpu'].get('device')}")
    sc = requests.get(f"{FLASK}/api/v1/scenarios").json(); check("GET /api/v1/scenarios", len(sc) >= 7, f"{len(sc)} scenarios")
    out = requests.post(f"{FLASK}/api/v1/offers/generate", json={"scenario": "clean_approval"}).json()
    check("POST generate clean_approval -> APPROVED", out["final_status"] == "APPROVED" and out["revisions"] == 0)
    out = requests.post(f"{FLASK}/api/v1/offers/generate", json={"scenario": "below_floor_price"}).json()
    check("below_floor_price rejected then approved", out["final_status"] == "APPROVED" and out["revisions"] == 1, f"revisions={out['revisions']}")
    wh = requests.post(f"{FLASK}/webhooks/pms", json={"key": "JUNIOR_OCEAN_VIEW", "available": 1}).json(); check("POST /webhooks/pms", wh.get("ok") is True)
    out = requests.post(f"{FLASK}/api/v1/offers/generate", json={"scenario": "last_room_upgrade", "pms_override": {}}).json()
    check("webhook-driven inventory protection", out["history"][1]["decision"] == "REJECTED" and "INVENTORY_BUFFER" in out["history"][1]["violations"])
    requests.post(f"{FLASK}/webhooks/pms", json={"key": "JUNIOR_OCEAN_VIEW", "available": 12})
    out = requests.post(f"{FLASK}/api/v1/offers/generate", json={"scenario": "vip_duplicate_perk"}).json(); check("VIP -> HITL_PENDING", out["final_status"] == "HITL_PENDING")
    q = requests.get(f"{FLASK}/api/v1/hitl/queue").json(); check("HITL queue lists pending item", any(x["thread_id"] == out["thread_id"] for x in q))
    d = requests.post(f"{FLASK}/api/v1/hitl/{out['thread_id']}/decision", json={"decision": "approve", "reviewer": "verify-script"}).json(); check("HITL approve resumes graph", d.get("final_status") == "HITL_APPROVED")
    esc = requests.post(f"{FLASK}/api/v1/offers/generate", json={"scenario": "unresolvable_escalation"}).json()      # leave one pending for the screenshot
    check("unresolvable -> HITL after R_max", esc["final_status"] == "HITL_PENDING" and esc["revisions"] == CFG["agents"]["max_revisions"])
    bad = requests.post(f"{FLASK}/webhooks/pms", json={"key": "NOPE", "available": 1}); check("webhook rejects bad payload", bad.status_code == 400)


def shots():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1500, "height": 1000})
        pg.goto(f"{FLASK}/audit"); pg.wait_for_timeout(2500); pg.screenshot(path=str(SHOTS / "flask_audit_ui.png"), full_page=True); check("screenshot flask_audit_ui", True)
        pg.select_option("#sc", "spammy_copy"); pg.click("#run"); pg.wait_for_timeout(4000); pg.screenshot(path=str(SHOTS / "flask_audit_after_run.png"), full_page=True)
        for name, url in (("api_health", "/health"), ("api_scenarios", "/api/v1/scenarios"), ("api_pms", "/api/v1/pms")):
            pg.goto(FLASK + url); pg.screenshot(path=str(SHOTS / f"{name}.png")); check(f"screenshot {name}", True)
        pg.goto(STREAMLIT); pg.wait_for_selector("text=Pre-Arrival Upsell", timeout=90000); pg.wait_for_timeout(4000)
        pg.screenshot(path=str(SHOTS / "streamlit_executive.png"), full_page=True); check("screenshot streamlit_executive", True)
        pg.set_viewport_size({"width": 1500, "height": 3400})   # streamlit scrolls an inner container, so make the viewport tall instead of full_page
        from agents.flows import FLOWS
        from agents.tools import SCENARIOS
        pg.get_by_role("tab", name="End-to-End Flow").click(); pg.wait_for_timeout(1500)
        for fi, (fkey, fl) in enumerate(FLOWS.items()):
            for ex in fl["scenarios"]:
                pg.get_by_role("radio").nth(fi).check(force=True); pg.wait_for_timeout(1200)
                pg.get_by_test_id("stSelectbox").first.click(); pg.wait_for_timeout(500)
                pg.get_by_role("option", name=SCENARIOS[ex]["title"], exact=True).click(); pg.wait_for_timeout(800)
                pg.get_by_role("button", name="Run end-to-end flow").click(); pg.wait_for_selector("text=Result:", timeout=120000); pg.wait_for_timeout(2500)
                pg.screenshot(path=str(SHOTS / f"streamlit_flow_{ex}.png"), full_page=True); check(f"screenshot streamlit_flow_{ex}", True, f"[{fkey}]")
        pg.get_by_role("tab", name="Flow Metrics").click(); pg.wait_for_timeout(3000)
        pg.screenshot(path=str(SHOTS / "streamlit_flow_metrics.png"), full_page=True); check("screenshot streamlit_flow_metrics", True)
        for tab, fn in (("Quant Models", "streamlit_quant"), ("Offline Policy Evaluation", "streamlit_ope"), ("HITL Queue", "streamlit_hitl")):
            pg.get_by_role("tab", name=tab).click(); pg.wait_for_timeout(2500); pg.screenshot(path=str(SHOTS / f"{fn}.png"), full_page=True); check(f"screenshot {fn}", True)
        b.close()


def main() -> int:
    env_py = sys.executable; procs = []; log = open(path("logs") / "verify_servers.log", "w")
    try:
        procs.append(subprocess.Popen([env_py, str(ROOT / "apps" / "flask_api.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        procs.append(subprocess.Popen([env_py, "-m", "streamlit", "run", str(ROOT / "apps" / "streamlit_app.py"), "--server.port", str(CFG["streamlit"]["port"]), "--server.headless", "true",
                                       "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        check("flask boots", wait(FLASK + "/health", 120)); check("streamlit boots", wait(STREAMLIT + "/_stcore/health", 120))
        api_checks(); shots()
    except Exception as e:  # noqa: BLE001
        check("verification run", False, f"{type(e).__name__}: {e}")
    finally:
        for p in procs:
            p.terminate()
    ok = all(r["ok"] for r in results)
    (path("metrics") / "verification_report.json").write_text(json.dumps({"passed": ok, "checks": results}, indent=2))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} ({sum(r['ok'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
