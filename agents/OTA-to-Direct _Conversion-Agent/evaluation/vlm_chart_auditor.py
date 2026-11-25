"""VLM visual QA: send each generated chart to qwen3-vl (image_url base64) and store a structured audit.
Offline fallback: a deterministic pixel heuristic (labelled mode='offline_heuristic' - it is NOT a visual-language judgement)."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings
from agent.llm_client import vision, endpoint


class ChartAudit(BaseModel):
    chart_type: str
    title_present: bool
    axes_or_labels_legible: bool
    overlapping_or_clipped_text: bool
    issues: list[str] = Field(default_factory=list)
    quality_score: int = Field(ge=1, le=10)
    summary: str


PROMPT = ("You are a visual QA reviewer for analytics charts and system diagrams. Inspect the image: identify the chart/diagram type, whether a title is present, "
          "whether axis labels/legend/node labels are legible, whether text overlaps or is clipped, and list concrete issues. Score 1-10.")


EXAMPLE = {"chart_type": "line chart", "title_present": True, "axes_or_labels_legible": True, "overlapping_or_clipped_text": False, "issues": ["legend overlaps curve"], "quality_score": 7, "summary": "one sentence"}


def _heuristic(path: Path) -> ChartAudit:
    import matplotlib.image as mpimg
    im = mpimg.imread(path); h, w = im.shape[:2]; g = im[..., :3].mean(-1)
    ink = float((g < 0.9).mean()); issues = []
    if w < 1200 or h < 600: issues.append(f"low resolution {w}x{h}")
    if ink < 0.02: issues.append("almost blank image")
    edge_ink = float(np.mean([(g[:3] < .9).mean(), (g[-3:] < .9).mean(), (g[:, :3] < .9).mean(), (g[:, -3:] < .9).mean()]))
    if edge_ink > 0.05: issues.append("content touches image border (possible clipping)")
    return ChartAudit(chart_type=path.stem.replace("_", " "), title_present=ink > 0.02, axes_or_labels_legible=w >= 1200, overlapping_or_clipped_text=edge_ink > 0.05,
                      issues=issues, quality_score=max(1, 9 - 2 * len(issues)), summary=f"Heuristic only: {w}x{h}px, ink coverage {ink:.1%}. Not a semantic visual review.")


def run() -> dict:
    s = get_settings(); charts = sorted(s.path("charts", "x").parent.glob("*.png")); ep = endpoint(refresh=True); results = []
    for p in charts:
        obj, meta = vision(PROMPT, p, ChartAudit, lambda p=p: _heuristic(p), example=EXAMPLE)
        results.append({"file": p.name, "mode": meta["mode"], "errors": meta.get("errors", []), **obj.model_dump()})
    rep = {"ts": datetime.now(timezone.utc).isoformat(), "model": ep["model"], "endpoint": ep["base_url"], "vlm_live": ep["available"], "n_charts": len(results),
           "mean_score": round(float(np.mean([r["quality_score"] for r in results])), 2) if results else None, "audits": results}
    s.path("evaluation", "vlm_visual_audit.json").write_text(json.dumps(rep, indent=2))
    print(f"audited {len(results)} charts, live VLM={ep['available']}, mean score={rep['mean_score']}"); return rep


if __name__ == "__main__":
    run()
