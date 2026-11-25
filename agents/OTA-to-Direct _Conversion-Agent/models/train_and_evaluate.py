"""Train uplift + cancellation models, run Gates D1-D3, export artifacts, metrics and hardware telemetry."""
from __future__ import annotations
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings, detect_hardware, ROOT
from data.synthetic_generator import generate, load, GEN_DIR
from data.schema_validator import gate_d1, update_report
from models import identity_resolution as ir
from models.uplift_pipeline import (UpliftModel, FEATURES, CANCEL_FEATURES, build_features, train_cancel, qini, calibration_bins)


def run(n_bookings: int | None = None, regenerate: bool = True) -> dict:
    s = get_settings(); hw = detect_hardware(); dev = "cuda" if hw["cuda"] else "cpu"
    outm = s.path("models", "_").parent
    b, cdp = generate(n_bookings) if (regenerate or not (GEN_DIR / "pms_bookings.csv").exists()) else load()

    d1 = gate_d1(b); update_report("D1", d1)
    resolved, tel_gpu = ir.resolve(b, cdp)
    _, tel_cpu = ir.resolve(b.head(5000), cdp, force_device="cpu")
    d2 = gate_d2 = ir.gate_d2(resolved); d2["telemetry"] = tel_gpu; update_report("D2", d2)
    resolved.to_csv(GEN_DIR / "resolved_bookings.csv", index=False)
    import faiss
    e = ir._embed(cdp.first_name, cdp.last_name); idx = faiss.IndexFlatIP(ir.DIM); idx.add(e); faiss.write_index(idx, str(outm / "cdp_faiss.index"))

    X = build_features(resolved, cdp)
    y = resolved.converted_direct.astype(int).values; t = resolved.treated.astype(int).values; yc = resolved.cancelled.astype(int).values
    tau_true = resolved.true_tau.values
    idx_all = np.arange(len(X))
    tr, te = train_test_split(idx_all, test_size=0.3, random_state=s.SEED, stratify=t)
    tr, va = train_test_split(tr, test_size=0.2, random_state=s.SEED, stratify=t[tr])

    rows, models, timings = [], {}, {}
    for kind in ("t", "x"):
        t0 = time.perf_counter(); m = UpliftModel(dev, kind).fit(X.iloc[tr], y[tr], t[tr]); timings[f"uplift_{kind}_train_{dev}_s"] = round(time.perf_counter() - t0, 3)
        tau = m.predict(X.iloc[te]); _, _, _, q = qini(y[te], t[te], tau)
        top = tau >= np.quantile(tau, 0.7)
        rows.append({"model": f"{kind.upper()}-learner", "device": dev, "qini_coef": round(q, 5), "corr_true_tau": round(float(np.corrcoef(tau, tau_true[te])[0, 1]), 4),
                     "mean_tau": round(float(tau.mean()), 4), "top30_true_tau": round(float(tau_true[te][top].mean()), 4), "all_true_tau": round(float(tau_true[te].mean()), 4)})
        models[kind] = (m, q)
    # CPU comparison timing for the best-practice X-learner
    t0 = time.perf_counter(); UpliftModel("cpu", "x").fit(X.iloc[tr], y[tr], t[tr]); timings["uplift_x_train_cpu_s"] = round(time.perf_counter() - t0, 3)
    best_kind = max(models, key=lambda k: models[k][1]); best, best_q = models[best_kind]

    # cancellation + isotonic calibration on validation split
    t0 = time.perf_counter(); cm = train_cancel(X.iloc[tr], yc[tr], dev); timings[f"cancel_train_{dev}_s"] = round(time.perf_counter() - t0, 3)
    pv = cm.predict_proba(X.iloc[va][CANCEL_FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yc[va])
    pt_raw = cm.predict_proba(X.iloc[te][CANCEL_FEATURES])[:, 1]; pt = iso.predict(pt_raw)
    auc = float(roc_auc_score(yc[te], pt)); _, ece_raw = calibration_bins(yc[te], pt_raw); cal, ece = calibration_bins(yc[te], pt + 1e-9 * np.arange(len(pt)))
    cancel_row = {"model": "Cancel-XGB", "device": dev, "roc_auc": round(auc, 4), "brier": round(float(brier_score_loss(yc[te], pt)), 4),
                  "ece_raw": round(ece_raw, 4), "ece_isotonic": round(ece, 4), "base_rate": round(float(yc[te].mean()), 4)}

    # Gate D3
    d3 = {"gate": "D3_model_quality", "chosen_uplift": f"{best_kind.upper()}-learner", "qini": round(best_q, 5), "cancel_auc": round(auc, 4),
          "checks": [{"check": "qini_gt_0", "passed": best_q > 0, "detail": best_q}, {"check": "cancel_auc_ge_0.72", "passed": auc >= 0.72, "detail": auc}]}
    d3["passed"] = all(c["passed"] for c in d3["checks"]); update_report("D3", d3)
    promoted = d3["passed"]
    if promoted:
        best.save(outm, "uplift"); cm.save_model(outm / "cancel.json")
        (outm / "cancel_isotonic.json").write_text(json.dumps({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}))
        (outm / "feature_schema.json").write_text(json.dumps({"features": FEATURES, "cancel_features": CANCEL_FEATURES}, indent=2))
    (outm / "promotion.json").write_text(json.dumps({"promoted": promoted, "ts": datetime.now(timezone.utc).isoformat(), "gate": d3}, indent=2))

    # Qini curve data + calibration for charts
    x, qc, rnd, _ = qini(y[te], t[te], best.predict(X.iloc[te]))
    pd.DataFrame({"frac": x, "qini": qc, "random": rnd}).to_csv(s.path("metrics", "qini_curve.csv"), index=False)
    cal.to_csv(s.path("metrics", "cancel_calibration.csv"), index=False)
    pd.DataFrame(rows + [cancel_row]).to_csv(s.path("metrics", "model_metrics.csv"), index=False)
    split = np.array(["train"] * len(X), dtype=object); split[va] = "val"; split[te] = "test"
    X.assign(tau_pred=best.predict(X), p_cancel=iso.predict(cm.predict_proba(X[CANCEL_FEATURES])[:, 1]), split=split).to_csv(GEN_DIR / "scored_features.csv", index=False)

    # inference latency CPU vs GPU
    lat = {}
    for d in {"cpu", dev}:
        mm = UpliftModel.load(outm, "uplift", d) if promoted else best
        row = X.iloc[te[:1]]; mm.predict(row)
        t0 = time.perf_counter(); [mm.predict(row) for _ in range(50)]; single = (time.perf_counter() - t0) / 50 * 1000
        bt = X.iloc[te[:5000]]; t0 = time.perf_counter(); mm.predict(bt); batch = (time.perf_counter() - t0) * 1000
        lat[d] = {"single_ms": round(single, 3), "batch5000_ms": round(batch, 3)}
    lat["train_seconds"] = timings
    lat["identity_resolution"] = {"gpu_run": tel_gpu, "cpu_run_5000_rows": tel_cpu}
    s.path("metrics", "latency_gpu.json").write_text(json.dumps(lat, indent=2))
    tele = {**hw, "xgboost_device": dev, "faiss_backend": tel_gpu["faiss_backend"], "torch_cosine_device": tel_gpu["device"], "timings": timings,
            "ts": datetime.now(timezone.utc).isoformat()}
    try:
        import torch
        tele["vram_peak_mb"] = int(torch.cuda.max_memory_allocated() / 2**20) if hw["cuda"] else 0
    except Exception: pass
    s.path("logs", "hardware_telemetry.json").write_text(json.dumps(tele, indent=2))
    summary = {"D1": d1["passed"], "D2": d2["passed"], "D3": d3["passed"], "promoted": promoted, "rows": rows, "cancel": cancel_row, "timings": timings}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
