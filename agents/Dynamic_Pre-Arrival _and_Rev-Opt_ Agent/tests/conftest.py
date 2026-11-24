import os
import sys
from pathlib import Path

os.environ["LLM_MODE"] = "mock"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "live_llm: needs the local Qwen server (deselect with -m 'not live_llm')")


@pytest.fixture(scope="session", autouse=True)
def _bootstrap():
    from src.config import get_settings
    s = get_settings()
    if not (s.raw_dir / "guests.csv").exists():
        from data.synthetic_generator import generate_all
        generate_all()
    from src.tools import pms_crm_tools as pms
    pms.init_pms()
    if not (s.models_dir / "nba_xgb.joblib").exists():
        from src.tools import nba_rms_engine as nba
        nba.train_model()
    yield


@pytest.fixture()
def clean_db():
    from src.tools import pms_crm_tools as pms
    pms.reset_runtime_tables()
    yield pms


@pytest.fixture()
def upcoming(clean_db):
    return clean_db.list_upcoming_reservations(60)
