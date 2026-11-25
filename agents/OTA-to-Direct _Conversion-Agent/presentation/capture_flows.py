"""Deck-specific captures of the Streamlit End-to-End Flows tab: narrow viewport + 2x pixel density so text stays legible
when a single stage panel is placed on a slide. Output: presentation/assets/raw/flow_<n>.png and flows_summary.png."""
import subprocess, sys, time
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "presentation" / "assets" / "raw"
PORT = 8512
sys.path.insert(0, str(ROOT))


def main():
    from agent.scenarios import build_scenarios
    RAW.mkdir(parents=True, exist_ok=True)
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui_streamlit/dashboard.py", "--server.port", str(PORT), "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 90:
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=2).status_code == 200: break
            except Exception:
                time.sleep(1)
        flows = build_scenarios()
        with sync_playwright() as p:
            br = p.chromium.launch()
            ctx = br.new_context(viewport={"width": 1250, "height": 4600}, device_scale_factor=2); pg = ctx.new_page()
            pg.goto(f"http://127.0.0.1:{PORT}"); pg.wait_for_selector("text=Operations Dashboard", timeout=90000); pg.wait_for_timeout(1500)
            for i, sc in enumerate(flows, 1):
                pg.get_by_text(sc["title"], exact=True).first.click(); pg.wait_for_timeout(600)
                pg.get_by_role("button", name="Run selected flow").click(); pg.wait_for_selector("text=Stage 6 - Log & attribute", timeout=120000); pg.wait_for_timeout(1800)
                pg.screenshot(path=str(RAW / f"flow_{i}.png")); print("captured flow", i)
            pg.get_by_role("button", name="Run all 5 flows").click(); pg.wait_for_selector("text=Running 5 flows", timeout=30000)
            pg.wait_for_selector("text=Running 5 flows", state="detached", timeout=240000); pg.wait_for_selector("text=Flow comparison", timeout=30000); pg.wait_for_timeout(1800)
            pg.screenshot(path=str(RAW / "flows_summary.png")); print("captured summary")
            br.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
