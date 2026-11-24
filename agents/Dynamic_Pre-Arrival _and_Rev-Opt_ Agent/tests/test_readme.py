import json

import pytest

from src.config import get_settings


def test_readme_updater_uses_real_files_and_fails_loudly(tmp_path, monkeypatch):
    from src.utils import readme_updater as ru
    s = get_settings()
    if not (s.metrics_dir / "metrics.json").exists():
        with pytest.raises(FileNotFoundError):
            ru.update_readme()
        return
    path = ru.update_readme()
    text = open(path, encoding="utf-8").read()
    m = json.loads((s.metrics_dir / "metrics.json").read_text())
    assert str(m["hallucination_rate_pct"]) in text and str(m["model"]["roc_auc"]) in text
    assert text.count("```mermaid") == 3
    assert "Data dictionary" in text
