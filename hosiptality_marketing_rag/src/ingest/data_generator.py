"""Deterministic synthetic hospitality corpus: case studies (MD+PDF), campaign reports (PDF),
brand guidelines (MD), KPI sheets (CSV/XLSX). No Tokyo property exists on purpose."""
from __future__ import annotations
import json
import random
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve

PROPERTIES = ["Miami Beach Resort", "Orlando Grand Hotel", "Fort Lauderdale Marina Inn"]
QUARTERS = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
CHANNELS = ["Direct", "Booking.com OTA", "Expedia OTA", "Corporate", "Wholesale"]
SEGMENTS = ["Leisure", "Business", "Group"]

BASE_ADR = {"Miami Beach Resort": 205, "Orlando Grand Hotel": 168, "Fort Lauderdale Marina Inn": 150}
Q_MULT = {"Q1 2025": 1.10, "Q2 2025": 0.92, "Q3 2025": 0.88, "Q4 2025": 1.02}
CH_MULT = {"Direct": 1.0, "Booking.com OTA": 0.93, "Expedia OTA": 0.91, "Corporate": 0.86, "Wholesale": 0.78}


def kpi_rows(rng: random.Random) -> pd.DataFrame:
    rows = []
    for p in PROPERTIES:
        for q in QUARTERS:
            for c in CHANNELS:
                adr = round(BASE_ADR[p] * Q_MULT[q] * CH_MULT[c] + rng.uniform(-3, 3))
                occ = round(min(96, max(52, 78 * (1.04 if q in ("Q1 2025", "Q4 2025") else 0.95)
                                        + rng.uniform(-6, 6))))
                rows.append({"Property": p, "Period": q, "Channel": c, "ADR": adr, "Occupancy": occ})
    df = pd.DataFrame(rows)
    # Anchor row used by the tests: Q3 2025, Direct, ADR $182, Occupancy 74% -> RevPAR $134.68
    m = (df.Property == "Miami Beach Resort") & (df.Period == "Q3 2025") & (df.Channel == "Direct")
    df.loc[m, ["ADR", "Occupancy"]] = [182, 74]
    df["RevPAR"] = (df.ADR * df.Occupancy / 100).round(2)
    return df


def fmt_table(df: pd.DataFrame) -> str:
    d = df.copy()
    d["ADR"] = d["ADR"].map(lambda v: f"${v}")
    d["Occupancy"] = d["Occupancy"].map(lambda v: f"{v}%")
    d["RevPAR"] = d["RevPAR"].map(lambda v: f"${v:.2f}")
    cols = list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def case_study_md(prop: str, df: pd.DataFrame) -> str:
    sub = df[df.Property == prop].drop(columns=["Property"])
    parts = [f"# {prop} Case Study 2025", "",
             f"## Executive Summary", "",
             f"{prop} pursued a direct-booking-first strategy in 2025, reducing dependence on online travel agencies "
             f"while growing loyalty enrolment. Direct channel mix rose each quarter, protecting average daily rate.", ""]
    for q in QUARTERS:
        qdf = sub[sub.Period == q].drop(columns=["Period"])
        parts += [f"## {q} Channel Performance", "",
                  f"The following KPI table summarises {prop} channel performance in {q}.", "",
                  fmt_table(qdf.assign(Period=q)[["Period", "Channel", "ADR", "Occupancy", "RevPAR"]]), "",
                  f"In {q}, direct bookings carried the highest ADR at {prop}, while wholesale carried the "
                  f"lowest. Management focused the CRM programme on converting OTA guests to loyalty members.", ""]
    parts += ["## Loyalty and CRM Strategy", "",
              f"The {prop} CRM team segmented guests by lifetime value (LTV) and sent tailored offers. "
              f"Members with high LTV received early access to suites and late checkout. "
              f"Email campaigns drove a measurable lift in repeat stays.", ""]
    return "\n".join(parts)


def campaign_report_lines(prop: str, rng: random.Random) -> list[tuple[str, list[str]]]:
    spend = rng.randrange(40, 90) * 1000
    roi = round(rng.uniform(2.1, 4.8), 1)
    sections = [
        (f"{prop} Summer Direct Booking Campaign", [
            f"The Q3 2025 summer campaign promoted direct booking benefits for {prop}.",
            f"Total spend was ${spend:,} across paid search, email and social.",
            f"Campaign ROI reached {roi}x versus a 2.0x target.",
            "Online travel agency commission savings were reinvested into the loyalty programme."]),
        (f"{prop} Loyalty Email Campaign", [
            f"A customer relationship management email series targeted high lifetime value members in Q4 2025.",
            f"Open rate was {rng.randrange(28, 46)}% and conversion {rng.randrange(3, 9)}%."]),
    ]
    return sections


BRAND_GUIDELINES = """# Brand Guidelines

## Voice and Tone

Our brand voice is warm, confident and effortlessly welcoming. We never use pressure tactics
or fake urgency in guest communications.

## Rate Parity Policy

All properties must maintain rate parity across channels. Direct bookings receive added value
(late checkout, welcome drink) rather than lower rates.

## Visual Identity

Use the primary palette of deep teal and warm sand. Photography must feature real guests and
real property spaces.
"""


def _pdf(path: Path, title: str, blocks: list[tuple[str, list[str], pd.DataFrame | None]]) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    ss = getSampleStyleSheet()
    story = [Paragraph(title, ss["Title"]), Spacer(1, 12)]
    for i, (head, paras, table) in enumerate(blocks):
        if i and i % 2 == 0:
            story.append(PageBreak())
        story.append(Paragraph(head, ss["Heading1"]))
        for t in paras:
            story += [Paragraph(t, ss["BodyText"]), Spacer(1, 6)]
        if table is not None:
            data = [list(table.columns)] + [[str(v) for v in r] for r in table.values.tolist()]
            tb = Table(data, repeatRows=1)
            tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
            story += [tb, Spacer(1, 10)]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)


def generate(out_dir: Path | None = None, seed: int = 42) -> Path:
    cfg = load_config()
    out = Path(out_dir) if out_dir else resolve(cfg, "raw")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    df = kpi_rows(rng)

    # Case studies: Miami as Markdown, Orlando as PDF (with table), Fort Lauderdale as Markdown
    (out / "case_study_miami_beach_resort.md").write_text(case_study_md(PROPERTIES[0], df), encoding="utf-8")
    (out / "case_study_fort_lauderdale_marina_inn.md").write_text(case_study_md(PROPERTIES[2], df), encoding="utf-8")
    orl = df[df.Property == PROPERTIES[1]]
    blocks = []
    blocks.append(("Executive Summary", [
        "Orlando Grand Hotel is a family-oriented property that leaned into group and leisure segments in 2025.",
        "Direct channel growth reduced online travel agency dependence."], None))
    for q in QUARTERS:
        t = orl[orl.Period == q][["Period", "Channel", "ADR", "Occupancy", "RevPAR"]].copy()
        t["ADR"] = t["ADR"].map(lambda v: f"${v}")
        t["Occupancy"] = t["Occupancy"].map(lambda v: f"{v}%")
        t["RevPAR"] = t["RevPAR"].map(lambda v: f"${v:.2f}")
        blocks.append((f"{q} Channel Performance",
                       [f"Orlando Grand Hotel channel KPI summary for {q}."], t))
    _pdf(out / "case_study_orlando_grand_hotel.pdf", "Orlando Grand Hotel Case Study 2025", blocks)

    # Campaign reports (PDF)
    for prop in PROPERTIES:
        blks = [(h, ps, None) for h, ps in campaign_report_lines(prop, rng)]
        _pdf(out / f"campaign_report_{prop.lower().replace(' ', '_')}.pdf", f"{prop} Campaign Report 2025", blks)

    (out / "brand_guidelines.md").write_text(BRAND_GUIDELINES, encoding="utf-8")

    # KPI sheets (CSV for Miami and Orlando, XLSX for Fort Lauderdale)
    for prop in PROPERTIES[:2]:
        df[df.Property == prop].to_csv(out / f"kpi_{prop.lower().replace(' ', '_')}.csv", index=False)
    df[df.Property == PROPERTIES[2]].to_excel(out / "kpi_fort_lauderdale_marina_inn.xlsx", index=False)
    return out


if __name__ == "__main__":
    print(generate())
