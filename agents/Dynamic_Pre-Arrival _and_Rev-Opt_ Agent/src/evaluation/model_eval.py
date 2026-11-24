"""XGBoost NBA model evaluation: ROC-AUC, PR-AUC, KS, Brier, calibration, feature importance -> evaluation/."""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from src.config import get_settings
from src.tools import nba_rms_engine as nba


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate() -> dict:
    s = get_settings()
    ho = pd.read_parquet(s.eval_dir / "holdout_predictions.parquet")
    y, p = ho["accepted"].to_numpy(), ho["p"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    model = nba.get_model()
    gain = model.get_booster().get_score(importance_type="gain")
    imp = sorted(({"feature": k, "gain": round(float(v), 3)} for k, v in gain.items()), key=lambda d: -d["gain"])
    base = float(y.mean())
    # lift @ top decile
    order = np.argsort(-p)
    top = y[order][: max(len(y) // 10, 1)].mean()
    info = json.loads((s.eval_dir / "training_info.json").read_text())
    out = {
        "device": info["device"], "tree_method": info["tree_method"], "n_test": int(len(y)), "base_rate": round(base, 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
        "ks": round(ks_statistic(y, p), 4), "brier": round(float(brier_score_loss(y, p)), 4),
        "brier_baseline": round(float(brier_score_loss(y, np.full_like(p, base))), 4),
        "top_decile_lift": round(float(top / base), 3), "feature_importance": imp[:12],
        "curves": {"roc": {"fpr": fpr[:: max(len(fpr) // 200, 1)].round(4).tolist(), "tpr": tpr[:: max(len(tpr) // 200, 1)].round(4).tolist()},
                   "pr": {"precision": prec[:: max(len(prec) // 200, 1)].round(4).tolist(), "recall": rec[:: max(len(rec) // 200, 1)].round(4).tolist()},
                   "calibration": {"mean_pred": mean_pred.round(4).tolist(), "frac_pos": frac_pos.round(4).tolist()}},
    }
    (s.eval_dir / "model_metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    m = evaluate()
    print({k: v for k, v in m.items() if k not in ("curves", "feature_importance")})
