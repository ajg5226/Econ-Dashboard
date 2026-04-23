"""
B2 drop-ablation harness.

For each B2 labor feature that survived feature selection in the hybrid
challenger (see ``data/models/b2_variants/with_labor/features.txt``),
retrain the hybrid model with that feature zeroed out in the engineered
feature frame and record the marginal CV PR-AUC / test PR-AUC change.

Results land in ``data/models/b2_variants/drop_ablation.json``.

Usage::

    python3 scripts/b2_drop_ablation.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("b2_drop")

from recession_engine.data_acquisition import RecessionDataAcquisition  # noqa: E402
from recession_engine.ensemble_model import RecessionEnsembleModel  # noqa: E402


# Features known to have survived selection in the with_labor run.
# The harness will auto-detect any additional B2 features that appear in
# the production features.txt at run time.
CANDIDATE_DROP_FEATURES = [
    "GOODS_DECLINE_FLAG",
    "GOODS_YoY_MINUS_SERV_YoY",
]


def _safe_pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def _cv_pr_auc_best_of_base(model) -> float:
    """Max-of-base-models CV PR-AUC, matching B1 harness convention."""
    cv = getattr(model, "cv_results", {}) or {}
    vals = []
    for d in cv.values():
        v = d.get("pr_auc")
        if v is not None and not np.isnan(v):
            vals.append(float(v))
    return float(max(vals)) if vals else float("nan")


def train_and_score(df_final: pd.DataFrame, *, horizon: int, max_features: int,
                     label: str) -> dict:
    """Train hybrid model, return compact metrics dict."""
    model = RecessionEnsembleModel(target_horizon=horizon, n_cv_splits=5)
    train_df, test_df = model.prepare_data(df_final)
    model.fit(train_df, max_features=max_features)
    preds = model.predict(test_df)
    y_true = test_df[f"RECESSION_FORWARD_{horizon}M"].values
    y_prob = preds["ensemble"]
    metrics_df = model.evaluate(test_df, preds)
    ens = metrics_df[metrics_df["Model"] == "ensemble"]
    test_pr_auc = float(ens["PR_AUC"].iloc[0]) if "PR_AUC" in ens.columns and not ens.empty else _safe_pr_auc(y_true, y_prob)
    test_auc = float(ens["AUC"].iloc[0]) if "AUC" in ens.columns and not ens.empty else float("nan")
    test_brier = float(ens["Brier"].iloc[0]) if "Brier" in ens.columns and not ens.empty else float("nan")
    cv_pr = _cv_pr_auc_best_of_base(model)

    selected = list(model.feature_cols)
    logger.info(
        "  %s: CV PR-AUC=%.4f test PR-AUC=%.4f test AUC=%.4f Brier=%.4f (selected %d)",
        label, cv_pr, test_pr_auc, test_auc, test_brier, len(selected),
    )
    return {
        "label": label,
        "cv_pr_auc": cv_pr,
        "test_pr_auc": test_pr_auc,
        "test_auc": test_auc,
        "test_brier": test_brier,
        "selected_feature_count": len(selected),
        "selected_contains_dropped": None,  # filled in by caller
    }


def main():
    horizon = 6
    max_features = 50
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    # Use the cached B1 FRED pull if present; otherwise pull fresh.
    cache = ROOT / "data" / "models" / "b1_variants" / "_df_raw_cache.pkl"
    acq = RecessionDataAcquisition(fred_api_key=api_key)

    if cache.exists():
        logger.info("Loading cached df_raw from %s", cache)
        df_raw = pd.read_pickle(cache)
        # Cache was created before B2 was added; missing B2 series — refetch.
        b2_required = [
            "coincident_JTSJOL", "coincident_JTSQUR",
            "coincident_CIVPART", "coincident_EMRATIO",
            "coincident_USGOOD", "coincident_USSERV",
            "coincident_UNEMPLOY",
        ]
        missing = [c for c in b2_required if c not in df_raw.columns]
        if missing:
            logger.info("Cache missing B2 series %s; refetching.", missing)
            df_raw = acq.fetch_data(start_date="1970-01-01")
            cache.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_pickle(cache)
    else:
        logger.info("No cache — fetching from FRED")
        df_raw = acq.fetch_data(start_date="1970-01-01")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_pickle(cache)

    logger.info("df_raw shape=%s", df_raw.shape)

    # Baseline hybrid (no drop)
    df_features_full = acq.engineer_features(df_raw.copy())
    df_final = acq.create_forecast_target(df_features_full, horizon_months=horizon)
    logger.info("=" * 80)
    logger.info("BASELINE: hybrid (no drop)")
    logger.info("=" * 80)
    baseline = train_and_score(df_final, horizon=horizon, max_features=max_features, label="hybrid (baseline)")

    # Detect extra B2 features in baseline's selected list
    from recession_engine.feature_variants import B2_LABOR_FEATURE_NAMES, B2_LABOR_RAW_SERIES
    b2_engineered = set(B2_LABOR_FEATURE_NAMES)
    b2_raw = set(B2_LABOR_RAW_SERIES)
    baseline_selected_features_path = ROOT / "data" / "models" / "b2_variants" / "with_labor" / "features.txt"
    surviving = set(CANDIDATE_DROP_FEATURES)
    if baseline_selected_features_path.exists():
        with open(baseline_selected_features_path) as f:
            for line in f:
                feat = line.strip()
                if feat in b2_engineered:
                    surviving.add(feat)
                elif any(feat == raw or feat.startswith(raw + "_") for raw in b2_raw):
                    surviving.add(feat)
    logger.info("B2 surviving features to ablate: %s", sorted(surviving))

    results = {"baseline": baseline, "drops": {}}
    for drop_feat in sorted(surviving):
        logger.info("=" * 80)
        logger.info("DROP-ABLATION: zero out %s", drop_feat)
        logger.info("=" * 80)
        df_modified = df_features_full.copy()
        if drop_feat not in df_modified.columns:
            logger.warning("  Column %s not present in engineered frame — skipping.", drop_feat)
            continue
        df_modified[drop_feat] = 0.0  # zero-out per task instruction
        df_final_drop = acq.create_forecast_target(df_modified, horizon_months=horizon)
        res = train_and_score(
            df_final_drop, horizon=horizon, max_features=max_features,
            label=f"hybrid (drop={drop_feat})"
        )
        res["delta_cv_pr_auc"] = res["cv_pr_auc"] - baseline["cv_pr_auc"]
        res["delta_test_pr_auc"] = res["test_pr_auc"] - baseline["test_pr_auc"]
        res["delta_test_auc"] = res["test_auc"] - baseline["test_auc"]
        res["delta_test_brier"] = res["test_brier"] - baseline["test_brier"]
        results["drops"][drop_feat] = res

    out_path = ROOT / "data" / "models" / "b2_variants" / "drop_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "ran_at_utc": datetime.utcnow().isoformat() + "Z",
                "horizon_months": horizon,
                "max_features": max_features,
                "features_ablated": sorted(surviving),
                "baseline": baseline,
                "drops": results["drops"],
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Compact summary
    logger.info("=" * 80)
    logger.info("DROP-ABLATION SUMMARY")
    logger.info("=" * 80)
    logger.info("  baseline CV PR-AUC=%.4f test PR-AUC=%.4f", baseline["cv_pr_auc"], baseline["test_pr_auc"])
    for feat, res in results["drops"].items():
        logger.info(
            "  drop %-28s | ΔCV PR-AUC=%+.4f | Δtest PR-AUC=%+.4f | Δtest AUC=%+.4f | ΔBrier=%+.4f",
            feat,
            res["delta_cv_pr_auc"],
            res["delta_test_pr_auc"],
            res["delta_test_auc"],
            res["delta_test_brier"],
        )


if __name__ == "__main__":
    main()
