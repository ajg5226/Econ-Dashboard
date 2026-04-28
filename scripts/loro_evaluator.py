"""
LORO evaluator — leave-one-recession-out aggregator over the existing
``RecessionBacktester.run_pseudo_oos_backtest`` per-recession framework.

Why this exists
---------------
The 80/20 chronological train/test split in ``RecessionEnsembleModel.prepare_data``
puts only one recession (COVID, 2 positive months) in the holdout, which makes
combination-method experiments (stackers, mixtures-of-experts, regime-conditional
weights) verdict-noisy. The pseudo-OOS backtester already evaluates per-recession
across all 7 NBER episodes — this wrapper adds:

1. Scope-tagged aggregation (in_scope vs informational, per the 1990-policy
   recorded in ``data/models/eval_origins.json``).
2. ``--emit-oof`` mode that persists per-recession raw OOF base-model
   predictions for downstream combination experiments to consume.

Outputs land under ``data/models/loro_<sha>/`` (or whatever ``--output-dir``
points at):

* ``loro_results.csv``      — backtest_results.csv schema + ``Evaluation_Scope``
* ``loro_summary.json``     — in-scope / informational / per-fold weight blocks
* ``loro_summary.txt``      — human-readable, ledger-ready
* ``oof_<recession>.parquet`` — only with ``--emit-oof``

Run
---
    # Baseline against current production state (uses cached indicators.csv)
    python3 scripts/loro_evaluator.py --output-dir data/models/loro_baseline_$(git rev-parse --short HEAD)/

    # Fresh fetch with extended history (Branch B verification)
    python3 scripts/loro_evaluator.py --start-date 1959-01-01 \\
        --output-dir data/models/loro_extended_$(git rev-parse --short HEAD)/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _candidate in [REPO_ROOT / ".env", *(p / ".env" for p in REPO_ROOT.parents)]:
    if _candidate.exists():
        load_dotenv(_candidate)
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("loro_evaluator")

from recession_engine.backtester import (  # noqa: E402
    PUBLICATION_LAGS,
    RecessionBacktester,
    apply_publication_lags,
)
from recession_engine.data_acquisition import RecessionDataAcquisition  # noqa: E402
from recession_engine.ensemble_model import RecessionEnsembleModel  # noqa: E402

EVAL_ORIGINS_PATH = REPO_ROOT / "data" / "models" / "eval_origins.json"
INDICATORS_CACHE = REPO_ROOT / "data" / "indicators.csv"


def _git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_scope_lookup() -> dict[str, str]:
    """Map NBER recession name → evaluation_scope (in_scope | informational).

    The backtester labels recessions with abbreviations like "GFC (2007-09)"
    while eval_origins.json uses canonical names like "Great Financial Crisis".
    We seed the lookup with both forms so substring tagging works either way.
    """
    if not EVAL_ORIGINS_PATH.exists():
        logger.warning("eval_origins.json missing; defaulting all scopes to 'informational'")
        return {}
    with open(EVAL_ORIGINS_PATH) as fh:
        payload = json.load(fh)
    lookup = {
        entry["name"]: entry["evaluation_scope"]
        for entry in payload.get("nber_recessions", [])
    }
    # Backtester-label aliases for canonical names that don't substring-match.
    aliases = {"GFC": "Great Financial Crisis"}
    for short, canonical in aliases.items():
        if canonical in lookup:
            lookup[short] = lookup[canonical]
    return lookup


def tag_scope(results_df: pd.DataFrame, scope_lookup: dict[str, str]) -> pd.DataFrame:
    """Add Evaluation_Scope column by substring-matching Recession label."""
    def _lookup(label: str) -> str:
        if not isinstance(label, str):
            return "informational"
        for name, scope in scope_lookup.items():
            if name.lower() in label.lower():
                return scope
        return "informational"
    results_df = results_df.copy()
    results_df["Evaluation_Scope"] = results_df["Recession"].map(_lookup)
    return results_df


def build_panel_from_cache(target_horizon: int) -> pd.DataFrame:
    """Use the cached engineered panel + recompute the forward target.

    Faster than a fresh FRED fetch and reproduces production training-frame
    state (1970-2026) since indicators.csv is written by the live update job.
    """
    if not INDICATORS_CACHE.exists():
        raise FileNotFoundError(
            f"{INDICATORS_CACHE} missing; run scheduler/update_job.py first or pass --start-date"
        )
    df = pd.read_csv(INDICATORS_CACHE, index_col=0, parse_dates=True)
    glr_cols = [c for c in df.columns if c.startswith("GLR_")]
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if glr_cols or object_cols:
        df = df.drop(columns=glr_cols + object_cols)
    target_col = f"RECESSION_FORWARD_{target_horizon}M"
    if target_col not in df.columns:
        # Match production target construction in
        # RecessionDataAcquisition.create_forecast_target — rolling-max
        # over the next H months, not a simple shift.
        df[target_col] = (
            df["RECESSION"].rolling(window=target_horizon, min_periods=1).max().shift(-target_horizon)
        )
    return df


def build_panel_from_fred(start_date: str, target_horizon: int) -> pd.DataFrame:
    """Fresh FRED fetch + engineer_features + forward target.

    Mirrors scheduler/update_job.py STEP 1-3 minus GLR (we strip GLR columns
    so feature selection inside the model behaves like production).
    """
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY not set; cannot run with --start-date")
    acq = RecessionDataAcquisition(fred_api_key=fred_api_key)
    df_raw = acq.fetch_data(start_date=start_date)
    df_raw_lagged = apply_publication_lags(df_raw, PUBLICATION_LAGS)
    df_features = acq.engineer_features(df_raw_lagged)
    df_final = acq.create_forecast_target(df_features, horizon_months=target_horizon)
    return df_final


def _model_class_with_optional_skip(skip_markov: bool):
    """Return a model class (subclass if skipping Markov)."""
    if not skip_markov:
        return RecessionEnsembleModel

    class _NoMarkovModel(RecessionEnsembleModel):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.markov_model = None

    return _NoMarkovModel


def emit_oof_for_recessions(
    df_with_target: pd.DataFrame,
    backtester: RecessionBacktester,
    output_dir: Path,
    *,
    target_horizon: int,
    max_features: int,
    n_cv_splits: int,
) -> None:
    """For each recession's training partition, run TimeSeriesSplit CV and
    save per-base-model raw OOF probabilities. Mirrors the engine's CV
    pattern at recession_engine/ensemble_model.py:1551-1640."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    target_col = f"RECESSION_FORWARD_{target_horizon}M"

    # Reuse the same default cutoffs as run_pseudo_oos_backtest.
    cutoff_dates = [
        ("1972-11", "1976-03", "Oil Crisis (1973-75)"),
        ("1979-01", "1981-07", "Volcker I (1980)"),
        ("1980-07", "1983-11", "Volcker II (1981-82)"),
        ("1989-07", "1992-03", "S&L Crisis (1990-91)"),
        ("2000-03", "2002-11", "Dot-com (2001)"),
        ("2006-12", "2010-06", "GFC (2007-09)"),
        ("2019-02", "2021-04", "COVID (2020)"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for train_end, _test_end, label in cutoff_dates:
        logger.info("OOF emit: %s (train through %s)", label, train_end)
        try:
            model = backtester._instantiate_model(n_cv_splits=n_cv_splits)
            train_df, _ = model.prepare_data(df_with_target, train_end_date=train_end)
            # Trigger feature selection so we get model.feature_cols
            model.fit(train_df, max_features=max_features)
            feature_cols = model.feature_cols
            X = train_df[feature_cols].ffill().fillna(0)
            y = train_df[target_col]
            n_components = min(model.n_pca_components, X.shape[1])
            tscv = TimeSeriesSplit(n_splits=n_cv_splits)
            base_names = list(model.models.keys())
            oof = {name: np.full(len(X), np.nan) for name in base_names}
            for tr_idx, va_idx in tscv.split(X):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr = y.iloc[tr_idx]
                sc = StandardScaler().fit(X_tr)
                pca = PCA(n_components=n_components).fit(sc.transform(X_tr))
                X_tr_sc = sc.transform(X_tr)
                X_va_sc = sc.transform(X_va)
                pca_tr = pca.transform(X_tr_sc)
                pca_va = pca.transform(X_va_sc)
                X_tr_probit = np.hstack([X_tr_sc, pca_tr])
                X_va_probit = np.hstack([X_va_sc, pca_va])
                X_tr_tree = np.hstack([X_tr.values, pca_tr])
                X_va_tree = np.hstack([X_va.values, pca_va])
                for name in base_names:
                    fold_model = clone(model.models[name])
                    if name == "probit":
                        fold_model.fit(X_tr_probit, y_tr)
                        proba = fold_model.predict_proba(X_va_probit)[:, 1]
                    else:
                        fold_model.fit(X_tr_tree, y_tr)
                        proba = fold_model.predict_proba(X_va_tree)[:, 1]
                    oof[name][va_idx] = proba
            oof_df = pd.DataFrame(oof, index=X.index)
            oof_df["Actual_Recession"] = y.values
            slug = label.split(" ")[0].lower().replace("&", "and")
            out_path = output_dir / f"oof_{slug}.parquet"
            oof_df.to_parquet(out_path)
            logger.info("  saved %s (%d rows)", out_path.name, len(oof_df))
        except Exception as exc:
            logger.warning("OOF emit failed for %s: %s", label, exc)


def compute_summaries(results_df: pd.DataFrame) -> dict:
    """Per-scope mean metrics. Skips Error rows (failed folds)."""
    clean = results_df[results_df.get("Error").isna()] if "Error" in results_df.columns else results_df
    summaries = {}
    for scope in ("in_scope", "informational"):
        bucket = clean[clean["Evaluation_Scope"] == scope]
        if bucket.empty:
            summaries[scope] = {"n": 0}
            continue
        summaries[scope] = {
            "n": int(len(bucket)),
            "mean_auc": float(bucket["AUC"].mean(skipna=True)) if "AUC" in bucket else None,
            "mean_brier": float(bucket["Brier"].mean(skipna=True)) if "Brier" in bucket else None,
            "mean_peak_prob": float(bucket["Peak_Prob"].mean(skipna=True)) if "Peak_Prob" in bucket else None,
            "mean_lead_months_own": float(bucket["Lead_Months"].mean(skipna=True)) if "Lead_Months" in bucket else None,
            "mean_lead_months_fixed": float(bucket["Lead_Months_Fixed"].mean(skipna=True)) if "Lead_Months_Fixed" in bucket else None,
            "n_crossed_threshold_fixed": int(bucket.get("Crossed_Threshold_Fixed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "recessions": bucket["Recession"].tolist(),
        }
    return summaries


def render_summary_text(results_df: pd.DataFrame, summaries: dict, args: argparse.Namespace) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("LORO EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Git SHA: {_git_sha_short()}")
    lines.append(f"Horizon: {args.horizon} months  |  Max features: {args.max_features}  |  Skip Markov: {args.skip_markov}")
    if args.start_date:
        lines.append(f"Start date (FRED fetch): {args.start_date}  [fresh fetch]")
    else:
        lines.append("Start date: from cached indicators.csv (production state)")
    lines.append("")
    lines.append("Per-recession results (sorted by Recession):")
    cols = ["Recession", "Evaluation_Scope", "AUC", "Brier", "Peak_Prob",
            "Crossed_Threshold_Fixed", "Lead_Months", "Lead_Months_Fixed"]
    cols = [c for c in cols if c in results_df.columns]
    lines.append(results_df[cols].to_string(index=False))
    lines.append("")
    for scope in ("in_scope", "informational"):
        block = summaries.get(scope, {})
        lines.append(f"--- {scope.upper()} (n={block.get('n', 0)}) ---")
        if block.get("n", 0) > 0:
            for key in ("mean_auc", "mean_brier", "mean_peak_prob",
                        "mean_lead_months_own", "mean_lead_months_fixed",
                        "n_crossed_threshold_fixed"):
                if key in block:
                    val = block[key]
                    lines.append(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon in months")
    parser.add_argument("--max-features", type=int, default=50, help="Max features for selection")
    parser.add_argument("--n-cv-splits", type=int, default=5, help="TimeSeriesSplit folds inside each per-recession fit")
    parser.add_argument("--skip-markov", action="store_true", help="Disable Markov-Switching for fast iteration")
    parser.add_argument("--emit-oof", action="store_true", help="Persist per-recession OOF base-model probabilities")
    parser.add_argument("--start-date", type=str, default=None,
                        help="If set, do a fresh FRED fetch from this date (YYYY-MM-DD); else use cached indicators.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory; defaults to data/models/loro_<sha>/")
    parser.add_argument("--in-scope-only", action="store_true",
                        help="Print only in_scope summary block (informational still saved to JSON)")
    args = parser.parse_args()

    sha = _git_sha_short()
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / "models" / f"loro_{sha}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    if args.start_date:
        logger.info("Building panel from fresh FRED fetch (start_date=%s)", args.start_date)
        df_with_target = build_panel_from_fred(args.start_date, args.horizon)
    else:
        logger.info("Building panel from cached indicators.csv")
        df_with_target = build_panel_from_cache(args.horizon)

    logger.info("Panel: %d rows × %d cols (%s → %s)",
                df_with_target.shape[0], df_with_target.shape[1],
                df_with_target.index.min().date(), df_with_target.index.max().date())

    model_class = _model_class_with_optional_skip(args.skip_markov)
    fred_api_key = os.environ.get("FRED_API_KEY")
    acq = RecessionDataAcquisition(fred_api_key=fred_api_key) if fred_api_key else None
    backtester = RecessionBacktester(acq, model_class, target_horizon=args.horizon)

    logger.info("Running pseudo-OOS backtest across all NBER recessions…")
    results_df = backtester.run_pseudo_oos_backtest(
        df_with_target,
        max_features=args.max_features,
        n_cv_splits=args.n_cv_splits,
    )

    scope_lookup = load_scope_lookup()
    results_df = tag_scope(results_df, scope_lookup)
    results_df.to_csv(output_dir / "loro_results.csv", index=False)
    logger.info("Saved results to %s", output_dir / "loro_results.csv")

    summaries = compute_summaries(results_df)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_short": sha,
        "horizon_months": args.horizon,
        "max_features": args.max_features,
        "n_cv_splits": args.n_cv_splits,
        "skip_markov": args.skip_markov,
        "start_date": args.start_date,
        "panel_rows": int(df_with_target.shape[0]),
        "panel_first_date": df_with_target.index.min().strftime("%Y-%m-%d"),
        "panel_last_date": df_with_target.index.max().strftime("%Y-%m-%d"),
        "summaries": summaries,
        "n_recessions_evaluated": int(len(results_df)),
        "ensemble_weights_per_fold": [
            {"recession": row["Recession"], "weights": row.get("Ensemble_Weights")}
            for _, row in results_df.iterrows()
        ],
    }
    with open(output_dir / "loro_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Saved summary JSON to %s", output_dir / "loro_summary.json")

    summary_text = render_summary_text(results_df, summaries, args)
    with open(output_dir / "loro_summary.txt", "w") as fh:
        fh.write(summary_text)
    logger.info("Saved summary text to %s", output_dir / "loro_summary.txt")

    print("\n" + summary_text)

    if args.emit_oof:
        logger.info("Emitting OOF base-model predictions per recession…")
        emit_oof_for_recessions(
            df_with_target, backtester, output_dir,
            target_horizon=args.horizon,
            max_features=args.max_features,
            n_cv_splits=args.n_cv_splits,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
