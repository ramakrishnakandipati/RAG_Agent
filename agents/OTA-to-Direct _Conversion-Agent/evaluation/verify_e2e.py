"""Master end-to-end runner: data -> train (gates D1-D3) -> replay -> diagrams -> VLM audit -> pytest -> UIs -> Playwright screenshots.

Usage:  python evaluation/verify_e2e.py [--skip-tests] [--skip-ui] [--n 20000]
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import get_settings


def step(report, name, fn):
    t0 = time.perf_counter()
    try:
        out = fn(); report["steps"].append({"step": name, "ok": True, "seconds": round(time.perf_counter() - t0, 1), **({"detail": out} if isinstance(out, dict) else {})}); print(f"[ok]   {name}")
        return out
    except Exception as e:
        report["steps"].append({"step": name, "ok": False, "seconds": round(time.perf_counter() - t0, 1), "error": f"{type(e).__name__}: {e}"}); print(f"[FAIL] {name}: {e}")
        return None


def wait(url, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200: return True
        except Exception:
            time.sleep(1)
    return False


def screenshots(s, report):
    from playwright.sync_api import sync_playwright
    shots = s.out / "screenshots"; shots.mkdir(parents=True, exist_ok=True); fl = f"http://127.0.0.1:{s.FLASK_PORT}"; stl = f"http://127.0.0.1:{s.STREAMLIT_PORT}"; made = []
    with sync_playwright() as p:
        br = p.chromium.launch(); pg = br.new_page(viewport={"width": 1500, "height": 1100})
        pg.goto(fl + "/"); pg.click("#btnSample"); pg.wait_for_timeout(500); pg.click("#run"); pg.wait_for_selector("#flow .node.pass, #flow .node.fail", timeout=60000); pg.wait_for_timeout(800)
        pg.screenshot(path=str(shots / "flask_simulator.png"), full_page=True); made.append("flask_simulator.png")
        pg.click("button[data-p=parity]"); pg.click("#run"); pg.wait_for_timeout(4000); pg.screenshot(path=str(shots / "flask_parity_breach.png"), full_page=True); made.append("flask_parity_breach.png")
        pg.goto(fl + "/evaluation"); pg.wait_for_selector("#charts img", timeout=30000); pg.wait_for_timeout(1500); pg.screenshot(path=str(shots / "flask_evaluation.png"), full_page=True); made.append("flask_evaluation.png")
        pg.goto(stl); pg.wait_for_selector("text=Operations Dashboard", timeout=90000); pg.wait_for_timeout(1500); pg.get_by_role("tab", name="Live Agent Flow").click(); pg.wait_for_timeout(800); pg.get_by_role("button", name="Run agent").click(); pg.wait_for_selector("text=Validation gates for this run", timeout=120000); pg.wait_for_timeout(1500); pg.screenshot(path=str(shots / "streamlit_agent_flow.png"), full_page=True); made.append("streamlit_agent_flow.png")
        for tab, fn in (("Guardrails", "streamlit_guardrails.png"), ("ROI & Validation", "streamlit_roi.png")):
            pg.get_by_role("tab", name=tab).click(); pg.wait_for_timeout(2500); pg.screenshot(path=str(shots / fn), full_page=True); made.append(fn)
        # ---- five curated end-to-end flows: Streamlit (stage-by-stage) + Flask (step flow), saved under screenshots/e2e_flows/
        from agent.scenarios import build_scenarios
        fdir = shots / "e2e_flows"; fdir.mkdir(exist_ok=True); flows = build_scenarios(); summary = []
        pg.set_viewport_size({"width": 1500, "height": 4300})   # Streamlit scrolls inside its own container; a tall viewport captures all six stages
        pg.goto(stl); pg.wait_for_selector("text=Operations Dashboard", timeout=90000); pg.wait_for_timeout(1500)
        for i, sc in enumerate(flows, 1):
            pg.get_by_text(sc["title"], exact=True).first.click(); pg.wait_for_timeout(600)
            pg.get_by_role("button", name="Run selected flow").click(); pg.wait_for_selector("text=Stage 6 - Log & attribute", timeout=120000); pg.wait_for_timeout(1500)
            fn = f"streamlit_flow_{i}_{sc['id']}.png"; pg.screenshot(path=str(fdir / fn), full_page=True); made.append("e2e_flows/" + fn)
        pg.get_by_role("button", name="Run all 5 flows").click(); pg.wait_for_selector("text=Running 5 flows", timeout=30000); pg.wait_for_selector("text=Running 5 flows", state="detached", timeout=180000); pg.wait_for_selector("text=Flow comparison", timeout=30000); pg.wait_for_timeout(1500); pg.screenshot(path=str(fdir / "streamlit_flows_summary.png"), full_page=True); made.append("e2e_flows/streamlit_flows_summary.png")
        pg.set_viewport_size({"width": 1500, "height": 1100}); pg.goto(fl + "/"); pg.wait_for_selector(".flowbtn", timeout=30000)
        for i, sc in enumerate(flows, 1):
            pg.click(f".flowbtn[data-i='{i - 1}']"); pg.wait_for_selector("#runmeta:not(:empty)", timeout=120000); pg.wait_for_timeout(1200)
            fn = f"flask_flow_{i}_{sc['id']}.png"; pg.screenshot(path=str(fdir / fn), full_page=True); made.append("e2e_flows/" + fn)
            pg.evaluate("document.querySelector('#runmeta').textContent=''")
        for name in ("system_architecture_diagram", "langgraph_state_machine", "data_pipeline_erd_flow"):
            pg.goto(f"{fl}/charts/{name}.png"); pg.wait_for_timeout(300); pg.screenshot(path=str(shots / f"diagram_{name}.png")); made.append(f"diagram_{name}.png")
        br.close()
    return {"screenshots": made}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-tests", action="store_true"); ap.add_argument("--skip-ui", action="store_true"); ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(); s = get_settings(); report = {"ts": datetime.now(timezone.utc).isoformat(), "steps": []}

    from models.train_and_evaluate import run as train
    from evaluation import offline_replay, diagram_generator, vlm_chart_auditor
    tr = step(report, "data + gates D1/D2/D3 + training", lambda: train(a.n))
    step(report, "offline replay + charts", offline_replay.run)
    step(report, "diagrams", diagram_generator.run)
    step(report, "VLM chart audit", lambda: {k: v for k, v in vlm_chart_auditor.run().items() if k != "audits"})
    if not a.skip_tests:
        def tests():
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-800:]
            return {"summary": r.stdout.strip().splitlines()[-1]}
        step(report, "pytest", tests)
    if not a.skip_ui:
        procs = []
        try:
            env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
            procs.append(subprocess.Popen([sys.executable, "-m", "web_flask.app"], cwd=ROOT, env=env, stdout=open(s.path("logs", "flask.out"), "w"), stderr=subprocess.STDOUT))
            procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui_streamlit/dashboard.py", "--server.port", str(s.STREAMLIT_PORT), "--server.headless", "true",
                                           "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, stdout=open(s.path("logs", "streamlit.out"), "w"), stderr=subprocess.STDOUT))
            step(report, "flask up", lambda: {"healthy": wait(f"http://127.0.0.1:{s.FLASK_PORT}/api/health")} if wait(f"http://127.0.0.1:{s.FLASK_PORT}/api/health") else (_ for _ in ()).throw(RuntimeError("flask not healthy")))
            step(report, "streamlit up", lambda: {"healthy": True} if wait(f"http://127.0.0.1:{s.STREAMLIT_PORT}/_stcore/health") else (_ for _ in ()).throw(RuntimeError("streamlit not healthy")))
            step(report, "playwright screenshots", lambda: screenshots(s, report))
        finally:
            for p in procs: p.terminate()
    report["all_ok"] = all(x["ok"] for x in report["steps"])
    s.path("evaluation", "e2e_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nE2E", "PASSED" if report["all_ok"] else "FAILED", "->", s.path("evaluation", "e2e_report.json"))
    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
