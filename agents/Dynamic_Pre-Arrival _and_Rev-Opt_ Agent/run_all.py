"""Master orchestrator: data gen -> validation -> GPU train -> offline replay -> charts -> UI screenshots -> README."""
from __future__ import annotations
import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-journeys", type=int, default=120)
    ap.add_argument("--mock-llm", action="store_true", help="use deterministic templates instead of the local Qwen server")
    ap.add_argument("--skip-ui", action="store_true", help="skip Playwright screenshots")
    ap.add_argument("--skip-data", action="store_true", help="reuse existing data/raw")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mock_llm:
        os.environ["LLM_MODE"] = "mock"

    from src.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    from src.utils.device import log_device
    from src.utils.logger import get_logger
    log = get_logger("run_all")
    t0 = time.time()

    def step(n, msg):
        print(f"\n[{n}/7] {msg}", flush=True)

    step(1, "Hardware")
    print(log_device())
    from src.llm.client import get_llm
    llm = get_llm()
    print(f"LLM mode={llm.mode} model={s.llm_model} up={llm.is_up()}")
    if llm.mode == "live" and not llm.is_up():
        print("ERROR: live LLM not reachable/model not loaded. Start the server or pass --mock-llm.", file=sys.stderr)
        return 2

    step(2, "Synthetic data")
    if not (a.skip_data and (s.raw_dir / "guests.csv").exists()):
        from data.synthetic_generator import generate_all
        d = generate_all(seed=a.seed)
        print({k: len(v) for k, v in d.items()})

    step(3, "Data validation")
    from src.validation.data_validator import validate_all
    rep = validate_all()
    print(f"{rep['n_passed']}/{rep['n_checks']} checks passed")
    if not rep["all_passed"]:
        print("ERROR: data validation failed:", [c for c in rep["checks"] if not c["passed"]], file=sys.stderr)
        return 3

    step(4, "PMS load + GPU XGBoost training")
    from src.tools import nba_rms_engine as nba
    from src.tools import pms_crm_tools as pms
    pms.init_pms(force=True)
    print(nba.train_model(seed=a.seed))

    step(5, f"Offline replay ({a.n_journeys} journeys)")
    from src.evaluation.offline_replay import run_replay
    m = run_replay(a.n_journeys, a.seed)
    print({k: m[k] for k in ("n_journeys", "hallucination_rate_pct", "funnel", "adversarial_catch_rate_pct")})

    step(6, "Charts")
    from src.visualization.generate_charts import generate_all_charts
    print(generate_all_charts())

    step(7, "UI screenshots + README")
    if not a.skip_ui:
        try:
            from src.visualization.capture_ui_screens import capture
            print(capture())
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: UI capture failed: {type(e).__name__}: {e}", file=sys.stderr)
    from src.utils.readme_updater import update_readme
    print("README ->", update_readme())
    log.info("run_all_done", seconds=round(time.time() - t0, 1))
    print(f"\nDone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
