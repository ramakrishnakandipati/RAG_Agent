"""Crop the 2x Streamlit captures (capture_flows.py) into per-stage panels for the deck. Borders are detected, not hard-coded.
Also copies the existing charts/screenshots the deck uses and writes assets/sizes.json."""
import json, shutil
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "presentation" / "assets" / "raw"
OUT = ROOT / "presentation" / "assets"
KEY = {1: 4, 2: 4, 3: 5, 4: 1, 5: 3}   # flow -> most telling stage
TRIM = 255                              # px (2x) of the trailing gate table
NAMES = ["happy_path", "parity_breach", "ghost_holdout", "new_guest_wifi", "post_stay_winback"]


def borders(a):
    cnt = (a < 240).sum(axis=0)
    w = a.shape[1]
    left = [x for x in range(int(w * 0.2), int(w * 0.5)) if cnt[x] > 3000]
    right = [x for x in range(int(w * 0.7), w) if cnt[x] > 3000]
    return left[0], right[-1]


def panels(a, x0):
    col = a[:, x0 - 1: x0 + 3].min(axis=1) < 240
    runs, start = [], None
    for y, v in enumerate(col):
        if v and start is None: start = y
        if not v and start is not None:
            if y - start > 120: runs.append((start, y))
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 28: merged[-1] = (merged[-1][0], r[1])
        else: merged.append(r)
    return merged


def main():
    for i, name in enumerate(NAMES, 1):
        img = Image.open(RAW / f"flow_{i}.png").convert("RGB"); a = np.asarray(img.convert("L")).astype(int)
        x0, x1 = borders(a); ps = [p for p in panels(a, x0) if p[0] > 1000]
        assert len(ps) >= 6, (name, ps)
        s = ps[0][0]
        rgb = np.asarray(img).astype(int); g = (rgb[:, x0:x1, 1] > 150) & (rgb[:, x0:x1, 0] < 90) & (rgb[:, x0:x1, 2] < 130)   # green status icons
        ys = [y for y in range(800, s) if g[y].sum() > 10]; y_icons = ys[0]
        img.crop((x0 - 4, y_icons - 24, x1 + 4, s - 125)).save(OUT / f"{name}_top.png")
        k = KEY[i]; ky0, ky1 = ps[k - 1]
        img.crop((x0 - 4, ky0 - 4, x1 + 4, ky1 - TRIM)).save(OUT / f"{name}_key.png")   # key stage panel without its gate table
        for k, (y0, y1) in enumerate(ps[:6], 1):
            img.crop((x0 - 4, y0 - 4, x1 + 4, y1 + 4)).save(OUT / f"{name}_stage{k}.png")
        print(name, "panels", len(ps), "pane x", x0, x1)
        if i == 5: last_end = ps[5][1]
    s = Image.open(RAW / "flows_summary.png").convert("RGB"); a = np.asarray(s.convert("L")).astype(int)
    x0, x1 = borders(a); ps = [p for p in panels(a, x0) if p[0] > 1000]
    rows = np.where(a[:, 150:2350].min(axis=1) < 235)[0]
    s.crop((150, ps[5][1] + 20, 2350, int(rows.max()) + 30)).save(OUT / "summary_table.png")
    # existing artifacts used by the deck
    for src in ["outputs/screenshots/streamlit_roi.png", "outputs/screenshots/streamlit_guardrails.png", "outputs/charts/system_architecture_diagram.png",
                "outputs/charts/data_pipeline_erd_flow.png", "outputs/charts/langgraph_state_machine.png"]:
        shutil.copy(ROOT / src, OUT / Path(src).name)
    for src, dst in (("streamlit_guardrails.png", "gov_guardrails.png"), ("streamlit_roi.png", "gov_roi.png")):
        im = Image.open(ROOT / "outputs" / "screenshots" / src).convert("RGB"); im.crop((60, 335, 1440, im.height)).save(OUT / dst)
    sizes = {p.name: list(Image.open(p).size) for p in OUT.glob("*.png")}
    (OUT / "sizes.json").write_text(json.dumps(sizes, indent=1)); print(len(sizes), "assets")


if __name__ == "__main__":
    main()
