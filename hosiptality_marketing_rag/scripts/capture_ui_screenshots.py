"""Drive the real UI and Swagger with Playwright (system Chrome) and save screenshots to outputs/screenshots/.

    python scripts/capture_ui_screenshots.py [--mock]
UI shots are named ui_*.png, new-API shots api_*.png.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from playwright.sync_api import Page, sync_playwright  # noqa: E402

from run_ui import API_URL, UI_URL, Stack, pick_mode  # noqa: E402

OUT = ROOT / "outputs" / "screenshots"
T = 300_000  # ms; first live answers take ~30 s

EX_MIAMI = "Miami Direct ADR (grounded KPI)"
EX_FILTER = "RevPAR by channel, only Q3 2025 (filter)"
EX_TOKYO = "Tokyo ADR (out of scope, refused)"
EX_CRM = "CRM and LTV strategy (narrative)"


def shot(page: Page, name: str) -> None:
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / name), full_page=True)
    print("saved", name)


def open_ui(page: Page) -> None:
    page.goto(UI_URL)
    page.get_by_text("API OK").first.wait_for(timeout=60_000)


def ask(page: Page, example: str | None = None) -> None:
    if example:
        page.get_by_role("button", name=example).click()
        page.wait_for_timeout(600)
    page.get_by_role("button", name="Ask", exact=True).click()
    page.get_by_text("Faithfulness", exact=True).first.wait_for(timeout=T)
    page.wait_for_timeout(500)


def tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(700)


def ui_shots(page: Page, only_metrics: bool = False) -> None:
    if only_metrics:
        return metrics_shot(page)
    # 01 home
    open_ui(page); shot(page, "ui_01_home_health.png")
    # scenario 1: grounded KPI (+ evidence, pipeline, verifier demo)
    open_ui(page); ask(page, EX_MIAMI); shot(page, "ui_02_ask_grounded_answer.png")
    tab(page, "Evidence"); shot(page, "ui_03_evidence_tab.png")
    tab(page, "Pipeline"); shot(page, "ui_04_pipeline_latency.png")
    tab(page, "Ask")
    page.get_by_text("Try the verifier").click()
    page.get_by_role("button", name="Verify against the retrieved evidence").click()
    page.get_by_text("BLOCKED").first.wait_for(timeout=120_000)
    shot(page, "ui_08_verifier_demo.png")
    # scenario 2: explicit filters (Property = Orlando) + "only Q3 2025"
    open_ui(page)
    page.get_by_label("Property").click()
    page.get_by_role("option", name="Orlando Grand Hotel").click()
    ask(page, EX_FILTER); tab(page, "Evidence"); shot(page, "ui_05_filter_isolation.png")
    # scenario 3: out of scope
    open_ui(page); ask(page, EX_TOKYO); shot(page, "ui_06_out_of_scope_refusal.png")
    # scenario 4: narrative
    open_ui(page); ask(page, EX_CRM); shot(page, "ui_09_narrative_crm_ltv.png")
    metrics_shot(page)


def metrics_shot(page: Page) -> None:
    open_ui(page); tab(page, "Metrics"); page.get_by_text("Retrieval quality").first.wait_for(timeout=30_000)
    shot(page, "ui_07_metrics_dashboard.png")


def swagger_op(page: Page, path: str, name: str, fill: dict | None = None, body: dict | None = None) -> None:
    """Expand a Swagger operation, 'Try it out', Execute, and screenshot just that operation block."""
    op = page.locator(".opblock", has_text=path).first
    op.locator(".opblock-summary").click()
    op.get_by_role("button", name="Try it out").click()
    for ph, val in (fill or {}).items():
        op.locator(f'input[placeholder="{ph}"]').fill(val)
    if body is not None:
        ta = op.locator("textarea.body-param__text")
        ta.fill(json.dumps(body, indent=2))
    op.get_by_role("button", name="Execute").click()
    op.locator(".live-responses-table").wait_for(timeout=T)
    page.wait_for_timeout(800)
    op.screenshot(path=str(OUT / name))
    print("saved", name)


def api_shots(page: Page) -> None:
    page.goto(f"{API_URL}/docs")
    page.locator(".opblock").first.wait_for(timeout=60_000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / "api_05_swagger_all_routes.png"), full_page=True); print("saved api_05_swagger_all_routes.png")
    swagger_op(page, "/meta/filters", "api_06_meta_filters.png")
    swagger_op(page, "/meta/examples", "api_07_meta_examples.png")
    swagger_op(page, "/meta/metrics", "api_08_meta_metrics.png")
    swagger_op(page, "/query", "api_09_query_with_filter_override.png",
               body={"question": "Show RevPAR by channel", "filters": {"period": "Q3 2025", "property": "Orlando Grand Hotel"}, "top_k": 3})
    swagger_op(page, "/chunks/{chunk_id}", "api_10_chunk_by_id.png",
               fill={"chunk_id": "case_study_miami_beach_resort.md::p1::t31"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", choices=["ui", "api", "metrics"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mode = pick_mode(a.mock)
    print(f"LLM mode: {mode}")
    with Stack(mode), sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1700}, device_scale_factor=1.5)
        page = ctx.new_page()
        if a.only != "api":
            ui_shots(page, only_metrics=a.only == "metrics")
        if a.only in (None, "api"):
            api_shots(page)
        browser.close()


if __name__ == "__main__":
    main()
