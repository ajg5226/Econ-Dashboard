"""
C1 helper: score pinned baseline_efb307e pickles on the current (with-C1) feature frame.

Adapted from ``scripts/b3_baseline_on_current_data.py`` but points at the
post-B3 baseline artifacts in ``data/models/baseline_efb307e/``. Writes
``data/models/c1_variants/baseline_efb307e_on_c1_data.json``.

This isolates the code delta (C1 benchmark members added) from retraining
variance: baseline pickles score the new feature frame directly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("c1_baseline_on_current_data")


def _safe(fn, y, p):
    try:
        return float(fn(y, p))
    except Exception:
        return None


def main() -> None:
    baseline_dir = ROOT / "data" / "models" / "baseline_efb307e"
    if not baseline_dir.exists():
        raise SystemExit(f"Baseline dir missing: {baseline_dir}")

    from recession_engine.data_acquisition import RecessionDataAcquisition

    logger.info("Fetching + engineering features (current data)...")
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        raise SystemExit("FRED_API_KEY not set")
    acq = RecessionDataAcquisition(fred_api_key=fred_api_key)
    df_features = acq.fetch_data()
    df_features = acq.engineer_features(df_features)
    df_features = acq.create_forecast_target(df_features, horizon_months=6)

    target_col = "RECESSION_FORWARD_6M"
    labeled = df_features.dropna(subset=[target_col]).copy()
    logger.info("Labeled rows: %d (%s to %s)", len(labeled),
                labeled.index.min().strftime("%Y-%m"),
                labeled.index.max().strftime("%Y-%m"))

    feat_path = baseline_dir / "features.txt"
    feature_cols = [
        line.strip() for line in feat_path.read_text().splitlines() if line.strip()
    ]

    missing = [c for c in feature_cols if c not in labeled.columns]
    if missing:
        logger.warning("Baseline features missing from current data: %s", missing[:5])

    existing = [c for c in feature_cols if c in labeled.columns]
    X = labeled[existing].ffill().fillna(0).values
    y = labeled[target_col].astype(int).values

    scaler = joblib.load(baseline_dir / "scaler.pkl")
    probit_cal = joblib.load(baseline_dir / "probit_calibrated.pkl")
    rf_cal = joblib.load(baseline_dir / "random_forest_calibrated.pkl")
    xgb_cal = None
    xgb_path = baseline_dir / "xgboost_calibrated.pkl"
    if xgb_path.exists():
        try:
            xgb_cal = joblib.load(xgb_path)
        except Exception as exc:
            logger.warning("xgboost_calibrated failed to load: %s", exc)
            xgb_cal = None

    try:
        X_scaled = scaler.transform(X)
    except ValueError:
        scaler_n_expected = getattr(scaler, "n_features_in_", None)
        logger.warning(
            "Scaler expects %s features but current X has %s; reshaping via pad/truncate.",
            scaler_n_expected, X.shape[1],
        )
        if scaler_n_expected is not None and X.shape[1] != scaler_n_expected:
            if X.shape[1] > scaler_n_expected:
                X = X[:, :scaler_n_expected]
            else:
                pad = np.zeros((X.shape[0], scaler_n_expected - X.shape[1]))
                X = np.hstack([X, pad])
        X_scaled = scaler.transform(X)

    def _prep_for(model, base_scaled, base_raw):
        n_expected = getattr(model, "n_features_in_", None)
        if n_expected is None or base_scaled.shape[1] == n_expected:
            return base_scaled
        if n_expected > base_scaled.shape[1]:
            pad = np.zeros((base_scaled.shape[0], n_expected - base_scaled.shape[1]))
            return np.hstack([base_scaled, pad])
        return base_scaled[:, :n_expected]

    def _prep_for_tree(model, base_raw):
        n_expected = getattr(model, "n_features_in_", None)
        if n_expected is None or base_raw.shape[1] == n_expected:
            return base_raw
        if n_expected > base_raw.shape[1]:
            pad = np.zeros((base_raw.shape[0], n_expected - base_raw.shape[1]))
            return np.hstack([base_raw, pad])
        return base_raw[:, :n_expected]

    n_test = 134
    n_total = len(labeled)
    test_slice = slice(max(0, n_total - n_test), n_total)
    X_scaled_test = X_scaled[test_slice]
    X_raw_test = X[test_slice]
    y_test = y[test_slice]

    X_probit_test = _prep_for(probit_cal, X_scaled_test, X_raw_test)
    X_rf_test = _prep_for_tree(rf_cal, X_raw_test)

    p_probit = probit_cal.predict_proba(X_probit_test)[:, 1]
    p_rf = rf_cal.predict_proba(X_rf_test)[:, 1]
    p_xgb = None
    if xgb_cal is not None:
        X_xgb_test = _prep_for_tree(xgb_cal, X_raw_test)
        try:
            p_xgb = xgb_cal.predict_proba(X_xgb_test)[:, 1]
        except Exception as exc:
            logger.warning("XGB predict failed: %s", exc)

    weights_path = baseline_dir / "ensemble_weights.json"
    weights = {}
    if weights_path.exists():
        try:
            weights = json.loads(weights_path.read_text())
        except Exception:
            weights = {}

    models_avail = {"probit": p_probit, "random_forest": p_rf}
    if p_xgb is not None:
        models_avail["xgboost"] = p_xgb
    if weights:
        ensemble = np.zeros_like(p_probit)
        total_w = 0.0
        for name, w in weights.items():
            if name in models_avail:
                ensemble = ensemble + float(w) * models_avail[name]
                total_w += float(w)
        if total_w > 0:
            ensemble = ensemble / total_w
    else:
        ensemble = np.mean(np.column_stack(list(models_avail.values())), axis=1)

    results = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "baseline_artifacts_dir": str(baseline_dir.relative_to(ROOT)),
        "baseline_git_sha": "efb307e",
        "scored_on_feature_frame": "post-B3 production frame (current)",
        "test_window_rows": int(y_test.shape[0]),
        "test_window_dates": [
            str(labeled.index[test_slice][0].date()),
            str(labeled.index[test_slice][-1].date()),
        ],
        "test_recession_rate": float(y_test.mean()),
        "baseline_weights_used": weights or "equal",
        "missing_features_count": len(missing),
        "missing_features_sample": missing[:8],
        "per_model": {
            "probit": {
                "auc": _safe(roc_auc_score, y_test, p_probit),
                "pr_auc": _safe(average_precision_score, y_test, p_probit),
                "brier": _safe(brier_score_loss, y_test, p_probit),
            },
            "random_forest": {
                "auc": _safe(roc_auc_score, y_test, p_rf),
                "pr_auc": _safe(average_precision_score, y_test, p_rf),
                "brier": _safe(brier_score_loss, y_test, p_rf),
            },
        },
        "ensemble": {
            "auc": _safe(roc_auc_score, y_test, ensemble),
            "pr_auc": _safe(average_precision_score, y_test, ensemble),
            "brier": _safe(brier_score_loss, y_test, ensemble),
        },
        "notes": (
            "Pinned baseline_efb307e pickles scored on the current (C1) "
            "feature frame's last 134 labeled rows. Isolates the code delta "
            "(C1 benchmark members added) from retraining variance: the "
            "baseline doesn't use benchmark members, so this should match "
            "at-freeze metrics unless the shared feature space changed."
        ),
    }
    if p_xgb is not None:
        results["per_model"]["xgboost"] = {
            "auc": _safe(roc_auc_score, y_test, p_xgb),
            "pr_auc": _safe(average_precision_score, y_test, p_xgb),
            "brier": _safe(brier_score_loss, y_test, p_xgb),
        }

    out_path = ROOT / "data" / "models" / "c1_variants" / "baseline_efb307e_on_c1_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote %s", out_path)
    logger.info(
        "Baseline ensemble on C1 data: AUC=%s PR-AUC=%s Brier=%s",
        results["ensemble"]["auc"],
        results["ensemble"]["pr_auc"],
        results["ensemble"]["brier"],
    )


if __name__ == "__main__":
    main()
