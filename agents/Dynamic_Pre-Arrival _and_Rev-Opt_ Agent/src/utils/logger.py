"""Structured JSONL + rotating-file logger writing to logs/."""
from __future__ import annotations
import json, logging, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGS = Path(__file__).resolve().parents[2] / "logs"
_LOGS.mkdir(parents=True, exist_ok=True)
_configured: dict[str, "StructLogger"] = {}


class StructLogger:
    def __init__(self, name: str):
        self.name = name
        self._l = logging.getLogger(f"hra.{name}")
        if not self._l.handlers:
            self._l.setLevel(logging.INFO)
            self._l.propagate = False
            fh = RotatingFileHandler(_LOGS / "agent.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%H:%M:%S"))
            self._l.addHandler(fh)
            self._l.addHandler(sh)

    def _emit(self, level: int, event: str, **kw):
        self._l.log(level, "%s %s", event, json.dumps(kw, default=str) if kw else "")
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "logger": self.name,
               "level": logging.getLevelName(level), "event": event, **kw}
        with open(_LOGS / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def info(self, event, **kw): self._emit(logging.INFO, event, **kw)
    def warning(self, event, **kw): self._emit(logging.WARNING, event, **kw)
    def error(self, event, **kw): self._emit(logging.ERROR, event, **kw)


def get_logger(name: str) -> StructLogger:
    if name not in _configured:
        _configured[name] = StructLogger(name)
    return _configured[name]


def telemetry(rec: dict) -> None:
    """Append a node/step telemetry record to logs/telemetry.jsonl."""
    with open(_LOGS / "telemetry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
