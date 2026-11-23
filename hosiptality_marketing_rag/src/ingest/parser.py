"""Document & table extractor: PDF (pymupdf text + pdfplumber tables), Markdown, CSV/XLSX."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ParsedElement:
    kind: str                       # "text" | "table"
    source: str
    page: int
    section: str
    text: str = ""                  # paragraph text, or markdown for tables
    table: Optional[pd.DataFrame] = None
    doc_title: str = ""


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def _clean_table(rows: list[list]) -> Optional[pd.DataFrame]:
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r] for r in rows if r]
    if len(rows) < 2:
        return None
    return pd.DataFrame(rows[1:], columns=rows[0])


# ---------------------------------------------------------------- PDF
def parse_pdf(path: Path) -> list[ParsedElement]:
    import pymupdf
    import pdfplumber

    els: list[ParsedElement] = []
    doc = pymupdf.open(str(path))
    title = ""
    section = "Introduction"
    with pdfplumber.open(str(path)) as plumb:
        for pno, page in enumerate(doc, start=1):
            tables = plumb.pages[pno - 1].find_tables()
            table_bboxes = [t.bbox for t in tables]

            def in_table(b) -> bool:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                return any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in table_bboxes)

            blocks = page.get_text("dict")["blocks"]
            # body font size = most common
            sizes: dict[float, int] = {}
            for b in blocks:
                for l in b.get("lines", []):
                    for s in l["spans"]:
                        sizes[round(s["size"])] = sizes.get(round(s["size"]), 0) + len(s["text"])
            body = max(sizes, key=sizes.get) if sizes else 10
            items = []  # (y, kind, payload)
            for b in blocks:
                if b.get("type") != 0 or in_table(b["bbox"]):
                    continue
                txt = " ".join(s["text"] for l in b["lines"] for s in l["spans"]).strip()
                if not txt:
                    continue
                size = max(s["size"] for l in b["lines"] for s in l["spans"])
                items.append((b["bbox"][1], "text", (txt, size)))
            for t in tables:
                df = _clean_table(t.extract())
                if df is not None:
                    items.append((t.bbox[1], "table", df))
            items.sort(key=lambda x: x[0])
            buf: list[str] = []

            def flush():
                if buf:
                    els.append(ParsedElement("text", path.name, pno, section, " ".join(buf), doc_title=title))
                    buf.clear()

            for _, kind, payload in items:
                if kind == "table":
                    flush()
                    els.append(ParsedElement("table", path.name, pno, section, df_to_markdown(payload),
                                             table=payload, doc_title=title))
                else:
                    txt, size = payload
                    if size >= body + 8 and not title:
                        title = txt
                    elif size >= body + 3:
                        flush()
                        section = txt
                    else:
                        buf.append(txt)
            flush()
    return els


# ---------------------------------------------------------------- Markdown
_HDR = re.compile(r"^(#{1,6})\s+(.*)$")


def _md_table_to_df(lines: list[str]) -> Optional[pd.DataFrame]:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for i, r in enumerate(rows) if not (i == 1 and all(re.fullmatch(r":?-{3,}:?", c) for c in r))]
    return _clean_table(rows)


def parse_markdown(path: Path) -> list[ParsedElement]:
    els: list[ParsedElement] = []
    title, section = "", "Introduction"
    para: list[str] = []
    tbl: list[str] = []

    def flush_para():
        if para:
            els.append(ParsedElement("text", path.name, 1, section, " ".join(para), doc_title=title))
            para.clear()

    def flush_tbl():
        if tbl:
            df = _md_table_to_df(tbl)
            if df is not None:
                els.append(ParsedElement("table", path.name, 1, section, "\n".join(tbl), table=df, doc_title=title))
            tbl.clear()

    for ln in path.read_text(encoding="utf-8").splitlines():
        m = _HDR.match(ln)
        if m:
            flush_para(); flush_tbl()
            if len(m.group(1)) == 1 and not title:
                title = m.group(2).strip()
            section = m.group(2).strip()
        elif ln.strip().startswith("|"):
            flush_para(); tbl.append(ln)
        elif not ln.strip():
            flush_para(); flush_tbl()
        else:
            flush_tbl(); para.append(ln.strip())
    flush_para(); flush_tbl()
    return els


# ---------------------------------------------------------------- CSV / Excel
def parse_tabular(path: Path) -> list[ParsedElement]:
    if path.suffix.lower() == ".csv":
        sheets = {"Sheet1": pd.read_csv(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)
    els = []
    for name, df in sheets.items():
        df = df.copy()
        if "Property" in df.columns:
            groups = [(f"{p}", g) for p, g in df.groupby("Property", sort=False)]
        else:
            groups = [(name, df)]
        for label, g in groups:
            for per, gg in (g.groupby("Period", sort=False) if "Period" in g.columns else [("all", g)]):
                gg = gg.reset_index(drop=True)
                for c in ("ADR", "RevPAR"):
                    if c in gg.columns:
                        gg[c] = gg[c].map(lambda v: f"${v:.2f}" if c == "RevPAR" else f"${int(v)}")
                if "Occupancy" in gg.columns:
                    gg["Occupancy"] = gg["Occupancy"].map(lambda v: f"{int(v)}%")
                els.append(ParsedElement("table", path.name, 1, f"{label} KPI {per}", df_to_markdown(gg),
                                         table=gg, doc_title=str(label)))
    return els


def parse_file(path: Path) -> list[ParsedElement]:
    s = path.suffix.lower()
    if s == ".pdf":
        return parse_pdf(path)
    if s in (".md", ".markdown"):
        return parse_markdown(path)
    if s in (".csv", ".xlsx", ".xls"):
        return parse_tabular(path)
    return []


def parse_dir(raw_dir: Path) -> list[ParsedElement]:
    out: list[ParsedElement] = []
    for p in sorted(raw_dir.iterdir()):
        out += parse_file(p)
    return out
