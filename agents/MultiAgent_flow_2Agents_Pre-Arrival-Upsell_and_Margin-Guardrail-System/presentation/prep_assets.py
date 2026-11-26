"""Prepare deck assets: per-example cross-model comparison JSON + cropped Streamlit screenshots (top / bottom halves)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = ROOT / "presentation" / "assets"; SH = ROOT / "outputs" / "screenshots"

def crops():
    for ex in ["clean_approval", "corporate_late_checkout", "below_floor_price", "spammy_copy", "last_room_upgrade", "vip_duplicate_perk", "unresolvable_escalation"]:
        im = Image.open(SH / f"streamlit_flow_{ex}.png").convert("RGB"); a = np.asarray(im)
        nonwhite = np.where((a < 250).any(axis=(1, 2)))[0]; end = int(nonwhite.max()) + 20
        start = 470                                                      # below tabs/title, at the flow-type selector
        mid = (start + end) // 2; best = mid
        for d in range(0, 260):                                          # snap split to a fully white gap between panels
            for y in (mid + d, mid - d):
                if (a[y] >= 250).all(): best = y; break
            else: continue
            break
        im.crop((60, start, 1440, best)).save(OUT / f"{ex}_top.png"); im.crop((60, best, 1440, end)).save(OUT / f"{ex}_bot.png")
        print(ex, start, best, end)

def compare():
    from agents.flows import run_scenario
    from agents.tools import SCENARIOS, get_engine
    from agents.graph import build_graph
    get_engine(); g = build_graph(); rows = {}
    for k in SCENARIOS:
        r = run_scenario(k, g); c0 = r["out"]["history"][0]["candidates"]; live = [c for c in c0 if not c["excluded"]]
        top = next((c for c in live if c["unit_margin"] > c["hurdle"]), live[0])
        rows[k] = {"top": top, "all": c0, "draft_sku": r["out"]["draft"]["sku"]}
    (OUT / "model_compare.json").write_text(json.dumps(rows, indent=1)); print("compare ok")

def extra():
    for src, dst, top in (("streamlit_flow_metrics.png", "flow_metrics_crop.png", 180), ("flask_audit_after_run.png", "flask_top.png", 0)):
        im = Image.open(SH / src).convert("RGB"); a = np.asarray(im)
        end = int(np.where((a < 250).any(axis=(1, 2)))[0].max()) + 20
        if src.startswith("flask"): end = min(end, 1250)
        im.crop((0, top, im.width, end)).save(OUT / dst)

crops(); extra(); compare()
