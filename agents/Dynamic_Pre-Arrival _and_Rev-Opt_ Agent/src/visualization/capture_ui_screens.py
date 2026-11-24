"""Launch Flask + Streamlit, drive the UI with Playwright, and save a PNG of every tab to outputs/ui_screens/."""
from __future__ import annotations
import os
import subprocess
import sys
import time

import requests

from src.config import ROOT, get_settings


def _wait(url: str, timeout: int = 120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code < 500:
                return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise TimeoutError(url)


def capture() -> list[str]:
    from playwright.sync_api import sync_playwright
    s = get_settings()
    s.screens_dir.mkdir(parents=True, exist_ok=True)
    for p in s.screens_dir.glob("*.png"):
        p.unlink()
    env = {**os.environ, "FLASK_PORT": str(s.flask_port), "API_BASE": f"http://127.0.0.1:{s.flask_port}", "PYTHONPATH": str(ROOT)}
    procs = [
        subprocess.Popen([sys.executable, "-m", "src.api.flask_app"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py", "--server.port", str(s.streamlit_port),
                          "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    saved: list[str] = []
    try:
        base_api = f"http://127.0.0.1:{s.flask_port}/api/v1"
        _wait(f"{base_api}/health")
        _wait(f"http://127.0.0.1:{s.streamlit_port}/_stcore/health")
        # prime state so the HITL queue and outbox are populated
        rl = requests.get(f"{base_api}/reservations", params={"limit": 40}, timeout=60).json()["reservations"]
        requests.post(f"{base_api}/trigger", json={"reservation_id": rl[5]["reservation_id"], "adversarial": "persistent"}, timeout=600)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1500, "height": 1000})
            page.goto(f"http://127.0.0.1:{s.streamlit_port}", wait_until="networkidle")
            page.wait_for_selector("text=Revenue Optimization Agent", timeout=60000)

            def shot(name: str):
                page.wait_for_timeout(1800)
                path = s.screens_dir / f"{name}.png"
                page.screenshot(path=str(path), full_page=True)
                saved.append(path.name)

            def tab(label: str):
                page.get_by_role("tab", name=label).click()
                page.wait_for_timeout(1200)

            shot("01_control_center_home")
            page.get_by_role("button", name="Run pre-arrival workflow").click()
            page.wait_for_selector("text=Status:", timeout=300000)
            page.wait_for_selector("text=Running LangGraph workflow", state="detached", timeout=300000)
            shot("02_live_trigger_result")
            page.get_by_role("button", name="Send inbound reply").click()
            page.wait_for_selector("text=Processing reply", timeout=30000)
            page.wait_for_selector("text=Processing reply", state="detached", timeout=300000)
            page.wait_for_selector("text=Folio posting", timeout=60000)
            shot("03_two_way_reply_booking")
            tab("State Trace"); shot("04_state_trace")
            tab("HITL Queue"); shot("05_hitl_queue")
            tab("Data Lineage"); shot("06_data_lineage")
            tab("Metrics"); shot("07_metrics_and_logs")
            b.close()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                p.kill()
    return saved


if __name__ == "__main__":
    print(capture())
