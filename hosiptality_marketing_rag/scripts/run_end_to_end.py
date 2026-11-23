"""One command: generate data -> index -> run pipeline -> metrics/charts -> pytest -> render all screenshots.

    python scripts/run_end_to_end.py            # live Qwen if reachable, else mock automatically
    python scripts/run_end_to_end.py --mock     # force the deterministic mock LLM
"""
from __future__ import annotations
import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from loguru import logger  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation import screenshot_renderer as R  # noqa: E402
from src.evaluation import visualizer as V  # noqa: E402
from src.generation.verifier import Verifier  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

QUERIES = [
    "What was the ADR for the Direct channel at Miami Beach Resort in Q3 2025?",
    "Show RevPAR and occupancy by channel at Orlando Grand Hotel, only Q3 2025",
    "How did the OTA channels compare with Direct on ADR at Fort Lauderdale Marina Inn in Q1 2025?",
    "What ROI did the summer direct booking campaign at Miami Beach Resort deliver?",
    "How does our CRM team use LTV to target loyalty members?",
    "What was Q3 2025 ADR in Tokyo?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = resolve(cfg, "outputs")
    for d in ("logs", "stage_artifacts", "metrics", "visualizations", "screenshots"):
        (out / d).mkdir(parents=True, exist_ok=True)

    if args.mock:
        cfg["llm"]["mode"] = "mock"
    elif not LLMClient(cfg).is_up():
        logger.warning("Qwen endpoint unreachable -> falling back to mock LLM")
        cfg["llm"]["mode"] = "mock"
    live = cfg["llm"]["mode"] == "live"

    pipe = RAGPipeline(cfg)
    logger.info(f"=== 1/6 data + index (llm mode: {cfg['llm']['mode']}) ===")
    info = pipe.ingest(regenerate=True)
    chunks = pipe.index.chunks

    logger.info("=== 2/6 pipeline queries ===")
    responses = [pipe.query(q) for q in QUERIES]
    M.write_latency_csv([{"question": r.question, **r.latency_ms} for r in responses],
                        out / "metrics" / "latency_breakdown.csv")

    logger.info("=== 3/6 retrieval metrics + charts ===")
    qa = M.build_golden(chunks)
    M.write_json(qa, resolve(cfg, "golden"))
    comp = M.compare_systems(pipe, qa)
    summary = {"retrieval": comp, "ingest": info, "llm_mode": cfg["llm"]["mode"],
               "mean_faithfulness": sum(r.verification.faithfulness_score for r in responses) / len(responses),
               "answers": [{"q": r.question, "faithfulness": r.verification.faithfulness_score,
                            "refusal": r.verification.is_refusal, "filters": r.rewritten.filters} for r in responses]}
    M.write_json(summary, out / "metrics" / "retrieval_metrics.json")
    viz = out / "visualizations"
    V.latency_waterfall([r.latency_ms for r in responses], viz / "latency_waterfall.png")
    V.retrieval_comparison(comp, viz / "retrieval_comparison.png")
    V.embedding_projection(pipe.index, viz / "embedding_umap.png")

    logger.info("=== 4/6 API payload ===")
    import api.main as api_main
    api_main.state["pipeline"] = pipe
    api_q = QUERIES[0] + " only Q3 2025"
    with TestClient(api_main.app) as client:
        api_payload = client.post("/query", json={"question": api_q}).json()

    # hallucination-guard example built from real retrieved context
    good = responses[0]
    good_ans = good.answer
    bad_ans = good_ans.replace("$182", "$299")
    v_ok = Verifier(None).verify(good_ans, good.contexts)
    v_bad = Verifier(None).verify(bad_ans, good.contexts)
    table_chunk = next(c for c in chunks if c.kind == "table" and "ADR (Average Daily Rate) $182" in c.text)

    # free GPU before the pytest subprocess loads its own models next to Qwen
    del pipe, api_main.state["pipeline"]
    gc.collect(); torch.cuda.empty_cache()

    rc = 0
    if not args.skip_tests:
        logger.info("=== 5/6 pytest ===")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--cov-report=json:outputs/metrics/coverage.json"] + (["--live-llm"] if live else [])
        with open(out / "logs" / "pytest_execution.log", "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
        logger.info(f"pytest exit code {rc}")

    logger.info("=== 6/6 screenshots ===")
    shots = out / "screenshots"
    rep = out / "metrics" / "test_report.json"
    if rep.exists():
        r = json.loads(rep.read_text(encoding="utf-8"))
        cov = out / "metrics" / "coverage.json"
        if cov.exists():
            r["_coverage"] = json.loads(cov.read_text())["totals"]["percent_covered"]
        tmp = out / "metrics" / "_report_for_render.json"
        tmp.write_text(json.dumps(r), encoding="utf-8")
        R.pytest_card(tmp, shots / "01_pytest_suite_terminal.png", live=live)
        tmp.unlink()
    R.table_linearization_card(table_chunk, shots / "02_table_linearization_inspect.png")
    R.query_rewrite_card([r.rewritten for r in responses[:4]], shots / "03_query_rewrite_audit.png")
    R.verification_card(good_ans, bad_ans, v_ok, v_bad, shots / "04_verification_hallucination_guard.png")
    R.api_response_card(f'POST /query  {{"question": "{api_q}"}}', api_payload,
                        shots / "05_fastapi_endpoint_response.png")
    logger.info(f"Done. Artifacts in {out}")
    print(json.dumps({"mode": cfg["llm"]["mode"], "pytest_exit": rc, "retrieval": comp,
                      "mean_faithfulness": summary["mean_faithfulness"]}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
