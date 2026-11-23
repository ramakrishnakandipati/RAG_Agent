"""Small presentation helpers for the Streamlit app."""
from __future__ import annotations
import html
import re

TEAL, COPPER, SLATE, RED = "#0F766E", "#C8793B", "#5B6B7A", "#B42318"
CITE_RE = re.compile(r"\[([^\[\]|]+?)\s*\|\s*(\d+)\s*\|\s*([^\[\]]+?)\]")


def faith_color(score: float) -> str:
    return TEAL if score >= 0.95 else (COPPER if score >= 0.75 else RED)


def render_answer_html(answer: str) -> str:
    """Escape, bold **headings**, bullet lines, and turn [source | page | period] into citation chips.
    Returns HTML (not markdown), so '$' never triggers LaTeX rendering."""
    out = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        bullet = line.startswith(("\u2022", "- ", "* "))
        line = re.sub(r"^(\u2022|- |\* )\s*", "", line)
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
        esc = CITE_RE.sub(lambda m: (f'<span class="chip">{html.escape(m.group(1))} &middot; p{m.group(2)} '
                                     f'&middot; {html.escape(m.group(3))}</span>'), esc)
        out.append(f'<li>{esc}</li>' if bullet else f'<p class="lead">{esc}</p>')
    body = "".join(out)
    if "<li>" in body:
        body = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", body, flags=re.S)
    return body


def gauge_html(score: float, label: str = "Faithfulness") -> str:
    pct = int(round(score * 100))
    c = faith_color(score)
    return (f'<div class="gauge"><div class="gauge-top"><span>{label}</span><b style="color:{c}">{score:.2f}</b></div>'
            f'<div class="gauge-bar"><div style="width:{pct}%;background:{c}"></div></div></div>')


def score_bar_html(score: float) -> str:
    pct = max(2, int(round(score * 100)))
    return f'<div class="gauge-bar thin"><div style="width:{pct}%;background:{TEAL}"></div></div>'


def chips_html(meta: dict, keys=("property", "period", "channel", "segment", "doc_type")) -> str:
    return "".join(f'<span class="chip soft">{html.escape(k)}: {html.escape(str(meta.get(k, "")))}</span>' for k in keys)


CSS = f"""
<style>
.chip {{display:inline-block;background:#E3F1EF;color:{TEAL};border-radius:10px;padding:1px 9px;font-size:0.78rem;
        font-weight:600;margin:0 2px;white-space:nowrap}}
.chip.soft {{background:#EEF3F2;color:{SLATE};font-weight:500;margin:2px 4px 2px 0}}
.answer-card {{background:#F7FAF9;border:1px solid #D5DEDC;border-radius:12px;padding:16px 20px}}
.answer-card p.lead {{font-size:1.08rem;font-weight:600;margin:0 0 8px 0}}
.answer-card ul {{margin:0 0 0 18px;padding:0}} .answer-card li {{margin:6px 0;line-height:1.5}}
.refusal {{background:#FFF4E5;border:1px solid {COPPER};border-radius:12px;padding:14px 18px}}
.gauge {{margin:6px 0}} .gauge-top {{display:flex;justify-content:space-between;font-size:0.9rem}}
.gauge-bar {{background:#E6ECEA;border-radius:6px;height:12px;overflow:hidden}} .gauge-bar.thin {{height:7px}}
.gauge-bar div {{height:100%}}
.kpi {{background:#EEF3F2;border-radius:12px;padding:10px 14px}}
</style>
"""
