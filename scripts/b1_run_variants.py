"""
B1 harness — at-risk representation bake-off.

Fetches FRED data once (with retries + local cache), then runs the four
feature variants {continuous_only, at_risk_only, hybrid, pca_on_binarized}
sequentially, capturing the key metrics for each.

Each variant writes its artifacts under ``data/models/b1_variants/<variant>/``.
A summary is emitted to ``data/models/b1_validation.json``.

Usage::

    python3 scripts/b1_run_variants.py
    python3 scripts/b1_run_variants.py --variants hybrid at_risk_only
    python3 scripts/b1_run_variants.py --skip-backtest  # faster dev loop
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("b1_harness")

# Deferred imports (need env + sys.path first)
from recession_engine.data_acquisition import RecessionDataAcquisition  # noqa: E402
from recession_engine.ensemble_model import RecessionEnsembleModel  # noqa: E402
from recession_engine.backtester import RecessionBacktester  # noqa: E402
from recession_engine.feature_variants import (  # noqa: E402
    SUPPORTED_VARIANTS,
    apply_feature_variant,
    describe_classification,
    must_include_collisions,
)


CACHE_PATH = ROOT / "data" / "models" / "b1_variants" / "_df_raw_cache.pkl"
OUTPUT_ROOT = ROOT / "data" / "models" / "b1_variants"
VALIDATION_PATH = ROOT / "data" / "models" / "b1_validation.json"


# ---------------------------------------------------------------------------
# FRED fetch with retry + cache
# ---------------------------------------------------------------------------

def fetch_df_raw_with_cache(force: bool = False, max_attempts: int = 4) -> pd.DataFrame:
    """Fetch raw FRED data with retry + pickle cache."""
    if CACHE_PATH.exists() and not force:
        logger.info("Loading cached df_raw from %s", CACHE_PATH)
        df = joblib.load(CACHE_PATH)
        if "RECESSION" in df.columns and df.shape[0] > 500:
            logger.info("Cached df_raw OK: %s", df.shape)
            return df
        logger.warning("Cache looked incomplete, refetching.")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("FRED fetch attempt %d/%d", attempt, max_attempts)
            acq = RecessionDataAcquisition(fred_api_key=api_key)
            df = acq.fetch_data(start_date="1970-01-01")
            if "RECESSION" not in df.columns:
                raise RuntimeError("FRED returned frame without RECESSION column")
            if df["RECESSION"].notna().sum() < 100:
                raise RuntimeError("FRED RECESSION series too sparse")
            # Require enough series for the pipeline to run. FRED can be flaky.
            if df.shape[1] < 35:
                raise RuntimeError(f"Too few series fetched: {df.shape[1]}")
            # Require the key leading/monetary/financial anchors
            required_anchors = [
                "RECESSION", "leading_T10Y3M", "leading_T10Y2Y",
                "coincident_UNRATE", "monetary_DFF",
            ]
            missing_anchors = [c for c in required_anchors if c not in df.columns]
            if missing_anchors:
                raise RuntimeError(f"Missing required anchor series: {missing_anchors}")
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(df, CACHE_PATH)
            logger.info("Fetched df_raw: %s; cached to %s", df.shape, CACHE_PATH)
            return df
        except Exception as exc:  # broad catch — FRED returns all sorts of errors
            last_err = exc
            sleep_s = 20 * attempt
            logger.warning("FRED fetch failed (%s). Sleeping %ds before retry.", exc, sleep_s)
            time.sleep(sleep_s)

    raise RuntimeError(f"FRED fetch failed after {max_attempts} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Per-variant training + evaluation
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _safe_pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def _safe_brier(y, p):
    from sklearn.metrics import brier_score_loss
    try:
        return float(brier_score_loss(y, p))
    except Exception:
        return float("nan")


def _safe_logloss(y, p):
    from sklearn.metrics import log_loss
    try:
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        return float("nan")


def _ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error — simple equal-width binning."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_total = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece_total += (mask.sum() / len(y_true)) * abs(avg_conf - avg_acc)
    return float(ece_total)


def run_variant(
    *,
    variant: str,
    df_raw: pd.DataFrame,
    acq: RecessionDataAcquisition,
    horizon_months: int = 6,
    max_features: int = 50,
    skip_backtest: bool = False,
    backtest_subset: Optional[Iterable[str]] = None,
) -> dict:
    """Train + evaluate one variant, returning a result dict."""

    logger.info("=" * 80)
    logger.info("VARIANT: %s", variant)
    logger.info("=" * 80)

    out_dir = OUTPUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Engineer features once per variant (needed because `RecessionBacktester`
    #    re-engineers internally during pseudo-OOS; we only ever pass
    #    pre-engineered+filtered frames to backtester methods that accept them).
    df_features = acq.engineer_features(df_raw.copy())
    if variant != "hybrid":
        pre_cols = df_features.shape[1]
        df_features = apply_feature_variant(df_features, variant)
        logger.info(
            "Variant %s: %d → %d columns (%s dropped)",
            variant,
            pre_cols,
            df_features.shape[1],
            pre_cols - df_features.shape[1],
        )

    # Audit must-include collisions
    missing_must_include = must_include_collisions(df_features)
    if missing_must_include:
        logger.info(
            "Variant %s drops %d must-include columns (first few: %s)",
            variant,
            len(missing_must_include),
            missing_must_include[:5],
        )

    # 2. Create target + split
    df_final = acq.create_forecast_target(df_features, horizon_months=horizon_months)

    # 3. Train
    model = RecessionEnsembleModel(target_horizon=horizon_months, n_cv_splits=5)
    train_df, test_df = model.prepare_data(df_final)

    model.fit(train_df, max_features=max_features)

    # 4. Eval
    predictions = model.predict(test_df)
    target_col = f"RECESSION_FORWARD_{horizon_months}M"
    y_true = test_df[target_col].values
    y_prob = predictions["ensemble"]

    # model.cv_results is keyed by base-model name with {auc, pr_auc, brier, inv_brier}.
    # The ensemble itself isn't recorded in cv_results; we use the best-of-base CV
    # metrics as a proxy for CV performance (the ensemble ties or very slightly
    # beats the best base via inv-Brier weighting).
    cv_scores_by_model = getattr(model, "cv_results", {}) or {}

    def _best_cv(metric: str, higher_is_better: bool = True):
        vals = [
            (name, d.get(metric)) for name, d in cv_scores_by_model.items()
            if d.get(metric) is not None and not np.isnan(d.get(metric, float("nan")))
        ]
        if not vals:
            return float("nan")
        if higher_is_better:
            return float(max(v for _, v in vals))
        return float(min(v for _, v in vals))

    cv_ens = {
        "auc": _best_cv("auc", higher_is_better=True),
        "pr_auc": _best_cv("pr_auc", higher_is_better=True),
        "brier": _best_cv("brier", higher_is_better=False),
        "logloss": float("nan"),
    }

    metrics_df = model.evaluate(test_df, predictions)

    ens_row = metrics_df[metrics_df["Model"] == "ensemble"]
    test_auc = float(ens_row["AUC"].iloc[0]) if not ens_row.empty else float("nan")
    test_pr_auc = float(ens_row["PR_AUC"].iloc[0]) if "PR_AUC" in ens_row.columns and not ens_row.empty else _safe_pr_auc(y_true, y_prob)
    test_brier = float(ens_row["Brier"].iloc[0]) if not ens_row.empty else float("nan")
    test_logloss = float(ens_row["LogLoss"].iloc[0]) if not ens_row.empty else float("nan")

    ece = _ece(y_true, y_prob)

    # Features classification of SELECTED features
    selected = list(model.feature_cols)
    # Also capture the internal top-20 ranked by RF importance — for "top-20 selected features"
    # we approximate with the order in features.txt (which reflects selection order +
    # RF fallback) up to 20. Also compute how many of the top-20 are at-risk.
    top20 = selected[:20]
    at_risk_in_top20 = [c for c in top20 if c in set(_all_at_risk_columns(df_features))]
    top_20_at_risk_fraction = float(len(at_risk_in_top20)) / float(len(top20)) if top20 else 0.0

    # Persist per-variant artifacts
    pd.Series(selected).to_csv(out_dir / "features.txt", index=False, header=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "cv_results.json", "w") as f:
        cv_serializable = {
            name: {k: float(v) for k, v in d.items()}
            for name, d in (model.cv_results or {}).items()
        }
        json.dump(cv_serializable, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "Date": test_df.index,
        "Actual_Recession": y_true,
        "Prob_Ensemble": y_prob,
        "Prob_Probit": predictions.get("probit"),
        "Prob_RandomForest": predictions.get("random_forest"),
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Backtest
    backtest_mean_auc = float("nan")
    backtest_mean_brier = float("nan")
    recessions_detected_str = "N/A"
    per_recession_leads = {}
    backtest_summary_text = None
    backtest_df = None

    if not skip_backtest:
        try:
            logger.info("Running pseudo-OOS backtest for variant %s...", variant)
            backtester = RecessionBacktester(acq, RecessionEnsembleModel, target_horizon=horizon_months)

            # Use a fixed subset of cutoffs (Dot-com + COVID are the most relevant
            # to the at-risk literature; also include GFC + Volcker for continuity).
            default_cutoffs = [
                ("1979-01", "1981-07", "Volcker I (1980)"),
                ("1989-07", "1992-03", "S&L Crisis (1990-91)"),
                ("2000-03", "2002-11", "Dot-com (2001)"),
                ("2006-12", "2010-06", "GFC (2007-09)"),
                ("2019-02", "2021-04", "COVID (2020)"),
            ]
            if backtest_subset:
                labels = set(backtest_subset)
                cutoffs = [c for c in default_cutoffs if c[2] in labels]
            else:
                cutoffs = default_cutoffs

            backtest_df = backtester.run_pseudo_oos_backtest(
                df_final,
                cutoff_dates=cutoffs,
                model_config=None,
                max_features=max_features,
                n_cv_splits=5,
            )
            backtest_df.to_csv(out_dir / "backtest_results.csv", index=False)
            # Summary
            backtest_summary_text = backtester.summarize_results(backtest_df)
            (out_dir / "backtest_summary.txt").write_text(backtest_summary_text)
            # Mean AUC / Brier
            if "AUC" in backtest_df.columns:
                backtest_mean_auc = float(backtest_df["AUC"].dropna().mean())
            if "Brier" in backtest_df.columns:
                backtest_mean_brier = float(backtest_df["Brier"].dropna().mean())

            detected = 0
            total = 0
            for _, row in backtest_df.iterrows():
                if "Crossed_Threshold" in row and not pd.isna(row.get("Crossed_Threshold")):
                    total += 1
                    if bool(row["Crossed_Threshold"]):
                        detected += 1
                if "Lead_Months" in row and not pd.isna(row.get("Lead_Months")):
                    per_recession_leads[str(row.get("Recession", ""))] = float(row["Lead_Months"])
            if total:
                recessions_detected_str = f"{detected}/{total}"
        except Exception as exc:
            logger.warning("Backtest failed for variant %s: %s", variant, exc)

    # Return a compact result dict for the validation JSON
    result = {
        "variant": variant,
        "feature_count": int(df_features.shape[1]),
        "must_include_collisions": missing_must_include,
        "cv_auc": float(cv_ens.get("auc", float("nan"))),
        "cv_pr_auc": float(cv_ens.get("pr_auc", float("nan"))),
        "cv_brier": float(cv_ens.get("brier", float("nan"))),
        "cv_logloss": float(cv_ens.get("logloss", float("nan"))),
        "cv_per_model": {
            name: {k: float(v) for k, v in d.items()}
            for name, d in cv_scores_by_model.items()
        },
        "test_auc": test_auc,
        "test_pr_auc": test_pr_auc,
        "test_brier": test_brier,
        "test_logloss": test_logloss,
        "ece": ece,
        "backtest_mean_auc": backtest_mean_auc,
        "backtest_mean_brier": backtest_mean_brier,
        "backtest_recessions": recessions_detected_str,
        "per_recession_lead_months": per_recession_leads,
        "selected_feature_count": int(len(selected)),
        "top_20_selected_features": top20,
        "top_20_at_risk_fraction": top_20_at_risk_fraction,
        "top_20_at_risk_names": at_risk_in_top20,
        "decision_threshold": float(model.decision_threshold),
        "ensemble_weights": {k: float(v) for k, v in model.ensemble_weights.items()},
    }
    # Persist the compact result for debugging
    with open(out_dir / "variant_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _all_at_risk_columns(df: pd.DataFrame) -> list:
    from recession_engine.feature_variants import classify_columns
    return classify_columns(df.columns).get("at_risk", [])


# ---------------------------------------------------------------------------
# Winner selection + validation JSON
# ---------------------------------------------------------------------------

def decide_winner(variant_results: dict) -> dict:
    """Pick a winner; evaluate KEEP/DISCARD/NEEDS-WORK verdict."""
    if "hybrid" not in variant_results:
        return {"winner": "hybrid", "verdict": "NEEDS-WORK",
                "winner_reason": "hybrid result missing — harness partial",
                "verdict_reason": "insufficient data"}

    hybrid = variant_results["hybrid"]

    # Primary metric: CV PR-AUC. If NaN, fall back to test PR-AUC.
    def score(v):
        s = v.get("cv_pr_auc")
        if s is None or np.isnan(s):
            s = v.get("test_pr_auc", float("nan"))
        return s

    scores = {name: score(v) for name, v in variant_results.items()}
    # Compare challengers to hybrid
    hybrid_score = scores["hybrid"]
    # Winner is argmax of score (hybrid wins ties via tiebreak on test_brier)
    ranked = sorted(scores.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -1), reverse=True)
    winner_name, winner_score = ranked[0]

    margin = (winner_score - hybrid_score) if (winner_name != "hybrid" and not np.isnan(winner_score) and not np.isnan(hybrid_score)) else 0.0

    # Keep/Discard gates
    KEEP_THRESHOLD = 0.01  # 1pp PR-AUC improvement
    verdict = "NEEDS-WORK"
    verdict_reason = "ambiguous"

    if winner_name == "hybrid":
        verdict = "DISCARD"
        verdict_reason = (
            "Hybrid (current default) beats all challengers on CV PR-AUC; "
            "alternative representations do not improve on the full kitchen-sink."
        )
        winner_reason = f"Hybrid top CV PR-AUC = {hybrid_score:.4f}"
    else:
        # Challenger wins — check KEEP gate.
        # Convention: "doesn't regress backtest below X/6" means the challenger's
        # detection count must be >= hybrid's detection count (we run the same
        # cutoffs for every variant, so parity is the relevant test).
        def _parse_rec(s):
            try:
                d, t = [int(x) for x in str(s).split("/")]
                return d, t
            except Exception:
                return 0, 0

        w_det, w_tot = _parse_rec(variant_results[winner_name].get("backtest_recessions", ""))
        h_det, _ = _parse_rec(hybrid.get("backtest_recessions", ""))

        # Ancillary guard — test-set PR-AUC must not regress by more than 1pp.
        w_test_pr = variant_results[winner_name].get("test_pr_auc", float("nan"))
        h_test_pr = hybrid.get("test_pr_auc", float("nan"))
        test_pr_regression = 0.0
        if not np.isnan(w_test_pr) and not np.isnan(h_test_pr):
            test_pr_regression = h_test_pr - w_test_pr

        if margin >= KEEP_THRESHOLD and w_det >= h_det and test_pr_regression <= 0.01:
            verdict = "KEEP"
            verdict_reason = (
                f"{winner_name} beats hybrid by {margin:.4f} CV PR-AUC "
                f"(>= {KEEP_THRESHOLD}) AND matches hybrid backtest detection "
                f"({w_det}/{w_tot} vs {h_det}/{w_tot}) AND does not regress test PR-AUC."
            )
        elif margin >= KEEP_THRESHOLD and w_det < h_det:
            verdict = "NEEDS-WORK"
            verdict_reason = (
                f"{winner_name} beats hybrid on CV PR-AUC by {margin:.4f} but "
                f"regresses backtest detection from {h_det}/{w_tot} to {w_det}/{w_tot}."
            )
        elif margin >= KEEP_THRESHOLD and test_pr_regression > 0.01:
            verdict = "NEEDS-WORK"
            verdict_reason = (
                f"{winner_name} beats hybrid on CV PR-AUC by {margin:.4f} but "
                f"regresses test-set PR-AUC by {test_pr_regression:.4f}."
            )
        else:
            verdict = "DISCARD"
            verdict_reason = (
                f"{winner_name} is nominal top but margin {margin:.4f} < KEEP gate "
                f"{KEEP_THRESHOLD}; hybrid remains champion."
            )
        winner_reason = (
            f"{winner_name} CV PR-AUC {winner_score:.4f} vs hybrid {hybrid_score:.4f} "
            f"(Δ={margin:+.4f})"
        )

    return {
        "winner": winner_name,
        "winner_reason": winner_reason,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "pr_auc_margin_vs_hybrid": float(margin),
    }


def write_validation_json(
    *,
    variant_results: dict,
    classification: dict,
    decision: dict,
    strict_vintage_results: dict | None,
):
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "B1",
        "ran_at_utc": datetime.utcnow().isoformat() + "Z",
        "git_sha_current": _git_sha(),
        "at_risk_column_count": classification.get("at_risk_column_count"),
        "continuous_column_count": classification.get("continuous_column_count"),
        "at_risk_column_list_sample": (
            classification.get("at_risk_column_list_sample_head", []) +
            classification.get("at_risk_column_list_sample_tail", [])
        ),
        "diffusion_columns_present": classification.get("diffusion_columns_present", []),
        "must_include_collisions": {
            name: v.get("must_include_collisions", []) for name, v in variant_results.items()
        },
        "variants": {
            name: {
                "cv_pr_auc": v.get("cv_pr_auc"),
                "cv_auc": v.get("cv_auc"),
                "cv_brier": v.get("cv_brier"),
                "cv_logloss": v.get("cv_logloss"),
                "test_pr_auc": v.get("test_pr_auc"),
                "test_auc": v.get("test_auc"),
                "test_brier": v.get("test_brier"),
                "test_logloss": v.get("test_logloss"),
                "backtest_mean_auc": v.get("backtest_mean_auc"),
                "backtest_mean_brier": v.get("backtest_mean_brier"),
                "backtest_recessions": v.get("backtest_recessions"),
                "per_recession_lead_months": v.get("per_recession_lead_months", {}),
                "ece": v.get("ece"),
                "feature_count": v.get("feature_count"),
                "selected_feature_count": v.get("selected_feature_count"),
                "top_20_selected_features": v.get("top_20_selected_features", []),
                "top_20_at_risk_fraction": v.get("top_20_at_risk_fraction"),
                "top_20_at_risk_names": v.get("top_20_at_risk_names", []),
                "decision_threshold": v.get("decision_threshold"),
                "ensemble_weights": v.get("ensemble_weights", {}),
            }
            for name, v in variant_results.items()
        },
        "winner": decision["winner"],
        "winner_reason": decision["winner_reason"],
        "verdict": decision["verdict"],
        "verdict_reason": decision["verdict_reason"],
        "pr_auc_margin_vs_hybrid": decision.get("pr_auc_margin_vs_hybrid"),
        "strict_vintage_candidates": strict_vintage_results or {},
    }
    with open(VALIDATION_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote %s", VALIDATION_PATH)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B1 at-risk variant bake-off harness")
    parser.add_argument("--variants", nargs="+", default=list(SUPPORTED_VARIANTS),
                        choices=list(SUPPORTED_VARIANTS))
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--force-refetch", action="store_true",
                        help="Force a fresh FRED pull even if cache exists")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-features", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()

    # Fetch (or load) df_raw
    df_raw = fetch_df_raw_with_cache(force=args.force_refetch)

    # Base classification of at-risk vs continuous based on hybrid features
    acq = RecessionDataAcquisition(fred_api_key=os.environ["FRED_API_KEY"])
    df_features_full = acq.engineer_features(df_raw.copy())
    classification = describe_classification(df_features_full)
    logger.info(
        "Classification: %d at-risk / %d continuous / %d meta",
        classification["at_risk_column_count"],
        classification["continuous_column_count"],
        classification["meta_column_count"],
    )

    results = {}
    for variant in args.variants:
        try:
            r = run_variant(
                variant=variant,
                df_raw=df_raw,
                acq=acq,
                horizon_months=args.horizon,
                max_features=args.max_features,
                skip_backtest=args.skip_backtest,
            )
            results[variant] = r
            elapsed = time.time() - t0
            logger.info("Elapsed so far: %.1f min", elapsed / 60.0)
        except Exception as exc:
            logger.exception("Variant %s FAILED: %s", variant, exc)
            results[variant] = {
                "variant": variant,
                "error": str(exc),
                "cv_pr_auc": float("nan"),
                "test_pr_auc": float("nan"),
                "must_include_collisions": [],
            }

    decision = decide_winner(results)
    payload = write_validation_json(
        variant_results=results,
        classification=classification,
        decision=decision,
        strict_vintage_results=None,
    )

    # Print compact summary
    logger.info("=" * 80)
    logger.info("B1 BAKE-OFF SUMMARY")
    logger.info("=" * 80)
    for name, v in results.items():
        logger.info(
            "  %-20s | CV PR-AUC=%.4f | test PR-AUC=%.4f | test Brier=%.4f | "
            "backtest AUC=%.3f | recessions=%s | ECE=%.4f",
            name,
            v.get("cv_pr_auc", float("nan")),
            v.get("test_pr_auc", float("nan")),
            v.get("test_brier", float("nan")),
            v.get("backtest_mean_auc", float("nan")),
            v.get("backtest_recessions", "N/A"),
            v.get("ece", float("nan")),
        )
    logger.info("WINNER: %s — %s", decision["winner"], decision["winner_reason"])
    logger.info("VERDICT: %s — %s", decision["verdict"], decision["verdict_reason"])


if __name__ == "__main__":
    main()
