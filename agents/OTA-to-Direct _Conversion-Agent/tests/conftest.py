import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Isolate test side-effects (JSONL logs, step_gates.json): work on a temp copy of the trained artifacts, never the deliverable outputs/.
import os, shutil, tempfile
_TMP = Path(tempfile.mkdtemp(prefix="ota_tests_"))
for sub in ("models", "validation"):
    if (ROOT / "outputs" / sub).exists():
        shutil.copytree(ROOT / "outputs" / sub, _TMP / sub, ignore=shutil.ignore_patterns("step_gates.json"))
os.environ["OUTPUT_DIR"] = str(_TMP)


@pytest.fixture(scope="session", autouse=True)
def ensure_models():
    """Tests need promoted models + generated data; train a small set if absent."""
    from config.settings import get_settings
    p = get_settings().out / "models" / "promotion.json"
    if not (p.exists() and json.loads(p.read_text()).get("promoted")):
        from models.train_and_evaluate import run
        assert run(n_bookings=12000)["promoted"]


@pytest.fixture(scope="session")
def bookings():
    from data.synthetic_generator import load
    return load()[0]


def make_webhook(row, **over):
    wh = {"booking_id": row.booking_id, "ota": row.ota, "first_name": row.first_name, "last_name": row.last_name, "masked_email": row.masked_email,
          "phone_last4": row.phone_last4 if isinstance(row.phone_last4, str) else None, "country": row.country, "checkin_date": row.checkin_date,
          "los": int(row.los), "adr": float(row.adr), "ota_commission": float(row.ota_commission), "lead_days": int(row.lead_days),
          "trip_purpose": row.trip_purpose, "free_cancellation": bool(row.free_cancellation), "ota_public_rate": float(row.adr), "direct_public_rate": float(row.adr),
          "is_ghost_holdout": bool(row.is_ghost_holdout)}
    wh.update(over); return wh
