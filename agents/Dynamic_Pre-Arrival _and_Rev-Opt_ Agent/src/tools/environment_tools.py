"""Environmental signals: weather forecast, local events, property occupancy (from RMS)."""
from __future__ import annotations
from functools import lru_cache

import pandas as pd

from src.config import get_settings


@lru_cache
def _tables():
    s = get_settings()
    w = pd.read_csv(s.raw_dir / "weather.csv").set_index("date")
    e = pd.read_csv(s.raw_dir / "events.csv")
    o = pd.read_csv(s.raw_dir / "occupancy.csv").set_index("date")
    return w, e, o


def get_environmental_signals(check_in: str, check_out: str) -> dict:
    w, e, o = _tables()
    days = [d for d in w.index if check_in <= d < check_out]
    if not days:
        days = [check_in] if check_in in w.index else []
    if not days:
        return {"precip_prob": 0.3, "precip_prob_max": 0.3, "temp_c": 22.0, "condition": "unknown",
                "events": [], "has_event": 0, "occupancy_pct": 60.0, "rms_adr": 230.0, "source": "default"}
    sub = w.loc[days]
    ev = e[e.date.isin(days)]["event_name"].tolist()
    ci = check_in if check_in in o.index else days[0]
    return {
        "precip_prob": round(float(sub.precip_prob.mean()), 3), "precip_prob_max": round(float(sub.precip_prob.max()), 3),
        "temp_c": round(float(sub.temp_c.mean()), 1), "condition": str(sub.condition.mode().iloc[0]),
        "events": ev, "has_event": int(bool(ev)),
        "occupancy_pct": float(o.loc[ci, "occupancy_pct"]), "rms_adr": float(o.loc[ci, "rms_adr"]), "source": "forecast",
    }
