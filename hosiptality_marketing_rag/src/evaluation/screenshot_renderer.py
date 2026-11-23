"""Render terminal logs, pytest results, API payloads and verification cards to PNG screenshots (Pillow)."""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM = (22, 27, 34), (201, 209, 217), (125, 133, 144)
GREEN, RED, AMBER, BLUE, TEAL = (63, 185, 80), (248, 81, 73), (210, 153, 34), (88, 166, 255), (45, 212, 191)

_FONT_CANDIDATES = ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _color_for(line: str):
    low = line.lower()
    if any(t in low for t in ("passed", "[pass]", "verified", " ok", "supported")) and "unsupported" not in low:
        return GREEN
    if any(t in low for t in ("failed", "error", "blocked", "unsupported", "hallucinat", "[fail]")):
        return RED
    if low.startswith(("==", "---", "#")):
        return BLUE
    return FG


def render_card(lines: list[str], path: Path, title: str = "", width_chars: int = 110, size: int = 16) -> Path:
    """Render lines as a dark terminal-style card. Each line may be 'text' or (text, (r,g,b))."""
    f, fb = _font(size), _font(size + 2)
    wrapped: list[tuple[str, tuple]] = []
    for ln in lines:
        text, col = ln if isinstance(ln, tuple) else (ln, _color_for(ln))
        parts = textwrap.wrap(text, width_chars, subsequent_indent="    ", drop_whitespace=False) or [""]
        wrapped += [(p, col) for p in parts]
    lh = size + 6
    pad, top = 24, 56
    w = int(f.getlength("M") * width_chars) + 2 * pad
    h = top + lh * len(wrapped) + pad
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 40], fill=(13, 17, 23))
    for i, c in enumerate((RED, AMBER, GREEN)):
        d.ellipse([16 + i * 22, 14, 28 + i * 22, 26], fill=c)
    d.text((w // 2 - 6 * len(title), 10), title, font=fb, fill=DIM)
    y = top
    for text, col in wrapped:
        d.text((pad, y), text, font=f, fill=col)
        y += lh
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------- specific cards
def pytest_card(report_json: Path, path: Path, live: bool = False) -> Path:
    r = json.loads(Path(report_json).read_text(encoding="utf-8"))
    s = r["summary"]
    lines = [f"$ pytest --cov=src{' --live-llm' if live else ''}", "=" * 40 + " test session starts " + "=" * 40, ""]
    for t in r["tests"]:
        mark = "PASSED" if t["outcome"] == "passed" else t["outcome"].upper()
        lines.append((f"{t['nodeid'].replace('tests/', ''):<88} {mark}",
                      GREEN if t["outcome"] == "passed" else (AMBER if t["outcome"] == "skipped" else RED)))
    lines += ["", "=" * 32 + f" {s.get('passed', 0)} passed, {s.get('failed', 0)} failed, "
              f"{s.get('skipped', 0)} skipped in {r.get('duration', 0):.1f}s " + "=" * 32]
    cov = r.get("_coverage")
    if cov is not None:
        lines.append((f"Total coverage (src/): {cov:.0f}%", AMBER))
    return render_card(lines, path, "pytest - Hospitality RAG test suite")


def table_linearization_card(chunk, path: Path) -> Path:
    md = chunk.table_md or ""
    sentences = chunk.text.split("Row summaries:\n", 1)[-1].splitlines()
    lines = [("# RAW KPI TABLE (kept intact as one chunk)", BLUE)] + md.splitlines() + [""]
    lines += [("# LINEARIZED ROW SENTENCES (appended for dense/sparse matching)", BLUE)] + sentences + [""]
    lines += [("# METADATA (Pydantic-validated)", BLUE)]
    lines += [f"  {k}: {v}" for k, v in chunk.metadata.model_dump().items()]
    return render_card(lines, path, "Table linearization inspector", width_chars=140, size=15)


def query_rewrite_card(rewrites: list, path: Path) -> Path:
    lines = []
    for rq in rewrites:
        lines += [(f"USER   : {rq.original}", AMBER)]
        for i, q in enumerate(rq.search_queries, 1):
            lines.append((f"QUERY {i}: {q}", TEAL))
        lines.append(f"FILTERS: {json.dumps(rq.filters)}")
        lines.append("ACRONYM MAP: " + (", ".join(f"{k} -> {v}" for k, v in rq.expansions.items()) or "-"))
        lines.append("")
    return render_card(lines, path, "Query rewrite audit (<=3 queries, acronym expansion, filters)", width_chars=130)


def verification_card(verified, blocked, v_ok, v_bad, path: Path) -> Path:
    """Side-by-side: grounded answer (verified) vs injected hallucination (blocked)."""
    half = 62
    left = [("VERIFIED ANSWER", GREEN), ""] + textwrap.wrap(verified, half) + [
        "", f"faithfulness_score = {v_ok.faithfulness_score:.2f}", f"claims supported: {len(v_ok.claims) - len(v_ok.unsupported_claims)}/{len(v_ok.claims)}"]
    left += [f"  OK {c.value}  <- {c.evidence_chunk}" for c in v_ok.claims]
    right = [("BLOCKED HALLUCINATION", RED), ""] + textwrap.wrap(blocked, half) + [
        "", f"faithfulness_score = {v_bad.faithfulness_score:.2f}",
        f"unsupported claims: {len(v_bad.unsupported_claims)}"]
    right += [f"  BLOCKED {c.value}  (not in retrieved context)" for c in v_bad.unsupported_claims]
    right += [f"  OK {c.value}" for c in v_bad.claims if c.supported]
    n = max(len(left), len(right))
    left += [""] * (n - len(left)); right += [""] * (n - len(right))
    f = _font(15)
    lh, pad = 21, 24
    col_w = int(f.getlength("M") * (half + 4))
    W, H = 2 * col_w + 3 * pad, 60 + lh * n + pad
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill=(13, 17, 23)); d.text((W // 2 - 140, 10), "Verification / hallucination guard", font=_font(18), fill=DIM)
    d.line([W // 2, 50, W // 2, H - 10], fill=DIM)
    for x0, col_lines in ((pad, left), (W // 2 + pad, right)):
        y = 56
        for ln in col_lines:
            text, col = ln if isinstance(ln, tuple) else (ln, _color_for(ln))
            d.text((x0, y), text, font=f, fill=col); y += lh
    path.parent.mkdir(parents=True, exist_ok=True); img.save(path)
    return path


def api_response_card(request: str, payload: dict, path: Path) -> Path:
    slim = {k: payload[k] for k in ("question", "answer", "citations") if k in payload}
    slim["filters"] = payload.get("rewritten", {}).get("filters")
    slim["search_queries"] = payload.get("rewritten", {}).get("search_queries")
    v = payload.get("verification", {})
    slim["verification"] = {"faithfulness_score": v.get("faithfulness_score"),
                            "unsupported_claims": len(v.get("unsupported_claims", [])),
                            "citation_errors": v.get("citation_errors")}
    slim["latency_ms"] = payload.get("latency_ms")
    lines = [(request, AMBER), ("HTTP/1.1 200 OK", GREEN), ("content-type: application/json", DIM), ""]
    lines += json.dumps(slim, indent=2, ensure_ascii=False).splitlines()
    return render_card(lines, path, "FastAPI  POST /query", width_chars=120)
