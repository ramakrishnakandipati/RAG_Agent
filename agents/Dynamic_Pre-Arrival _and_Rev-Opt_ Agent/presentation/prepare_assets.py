"""Crop the Playwright screenshots so the deck can show them larger (writes presentation/assets/)."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "ui_screens"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

# use-case screens: drop the header, tabs and use-case picker; keep the guest banner and all steps
for p in sorted(SRC.glob("uc*.png")):
    im = Image.open(p)
    im.crop((0, 812, im.width, im.height)).save(OUT / f"{p.stem}_crop.png")
# metrics tab is very tall: keep KPIs and the first charts
im = Image.open(SRC / "07_metrics_and_logs.png")
im.crop((0, 0, im.width, min(im.height, 1500))).save(OUT / "07_metrics_top.png")
print(sorted(x.name for x in OUT.iterdir()))
