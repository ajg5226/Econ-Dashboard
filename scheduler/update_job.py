"""
Background job for data refresh and model retraining
Can be run via cron, GitHub Actions, or manually
"""

import os
import sys
import argparse
import json
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Also check Streamlit Cloud secrets
if not os.environ.get('FRED_API_KEY'):
    try:
        import streamlit as st
        os.environ['FRED_API_KEY'] = st.secrets.get('FRED_API_KEY', '')
    except Exception:
        pass

from recession_engine.data_acquisition import RecessionDataAcquisition
from recession_engine.backtester import DEFAULT_SEARCH_CANDIDATES, RecessionBacktester
from recession_engine.ensemble_model import RecessionEnsembleModel
from recession_engine.glr_engine import GLRRegimeEngine
from recession_engine.model_monitor import ModelMonitor
from scheduler.scheduler_config import load_runtime_config
try:
    from app.utils.data_loader import (
        save_predictions, save_indicators, save_executive_report,
        save_glr_components, ensure_data_dir
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.utils.data_loader import (
        save_predictions, save_indicators, save_executive_report,
        save_glr_components, ensure_data_dir
    )

# Configure logging
log_dir = Path(__file__).parent.parent / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_SEARCH_CANDIDATE_MAP = {
    candidate['id']: candidate for candidate in DEFAULT_SEARCH_CANDIDATES
}
PROMOTION_GATE_METRICS = {
    'AUC': 'higher',
    'PR_AUC': 'higher',
    'Brier': 'lower',
    'LogLoss': 'lower',
}


def _get_git_sha() -> str:
    """Best-effort retrieval of current git SHA for provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.debug("Could not determine git SHA for run manifest: %s", exc)
        return "unknown"


def _coerce_int(value, *, default: int, field_name: str, source: str) -> int:
    """Convert an artifact value to int without letting malformed metadata abort the run."""
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s in %s (%r); using %s.",
            field_name,
            source,
            value,
            default,
        )
        return default


def _build_run_manifest(*, horizon_months, train_end_date, max_features,
                        threshold_override, model, metrics_df, predictions_df,
                        selection_metadata=None, model_config=None, n_cv_splits=5):
    """Create a concise provenance manifest for each model refresh run."""
    ensemble_row = metrics_df[metrics_df['Model'] == 'ensemble']
    ensemble_metrics = {}
    if not ensemble_row.empty:
        ensemble_metrics = {
            'auc': float(ensemble_row['AUC'].iloc[0]),
            'pr_auc': float(ensemble_row['PR_AUC'].iloc[0]) if 'PR_AUC' in ensemble_row else None,
            'brier': float(ensemble_row['Brier'].iloc[0]),
            'logloss': float(ensemble_row['LogLoss'].iloc[0]),
        }

    latest_known = predictions_df.dropna(subset=['Actual_Recession'])
    latest_known_date = None
    if len(latest_known) > 0:
        latest_known_date = pd.to_datetime(latest_known['Date'].max()).strftime('%Y-%m-%d')

    manifest = {
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'git_sha': _get_git_sha(),
        'python_version': platform.python_version(),
        'horizon_months': int(horizon_months),
        'train_end_date': train_end_date,
        'max_features': int(max_features),
        'n_cv_splits': int(n_cv_splits),
        'model_config': model_config or {},
        'threshold_override': threshold_override,
        'decision_threshold_used': float(model.decision_threshold),
        'threshold_method': getattr(model, 'threshold_method', 'unknown'),
        'ensemble_method': getattr(model, 'ensemble_method', 'unknown'),
        'active_models': list(getattr(model, 'active_models', [])),
        'selected_features_count': int(len(model.feature_cols)),
        'ensemble_weights': {k: float(v) for k, v in model.ensemble_weights.items()},
        'ensemble_metrics': ensemble_metrics,
        'predictions_rows': int(len(predictions_df)),
        'latest_prediction_date': pd.to_datetime(predictions_df['Date'].max()).strftime('%Y-%m-%d'),
        'latest_known_outcome_date': latest_known_date,
    }
    if selection_metadata:
        manifest['model_selection'] = selection_metadata
    return manifest


def _build_rolling_metrics(predictions_df: pd.DataFrame, threshold: float, window_months: int = 36) -> pd.DataFrame:
    """Compute rolling-window performance diagnostics for known-outcome rows."""
    known_df = predictions_df.dropna(subset=['Actual_Recession']).copy()
    if known_df.empty or len(known_df) < window_months:
        return pd.DataFrame()

    probability_columns = {
        'ensemble': 'Prob_Ensemble',
        'probit': 'Prob_Probit',
        'random_forest': 'Prob_RandomForest',
        'xgboost': 'Prob_XGBoost',
        'markov_switching': 'Prob_MarkovSwitching',
    }

    rows = []
    for model_name, column_name in probability_columns.items():
        if column_name not in known_df.columns:
            continue

        model_df = known_df[['Date', 'Actual_Recession', column_name]].copy()
        for end_idx in range(window_months - 1, len(model_df)):
            window_df = model_df.iloc[end_idx - window_months + 1:end_idx + 1]
            y_true = window_df['Actual_Recession'].astype(int).values
            y_proba = np.clip(window_df[column_name].astype(float).values, 1e-7, 1 - 1e-7)
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            positives = int(y_true.sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) >= 2 else np.nan
            pr_auc = average_precision_score(y_true, y_proba) if positives > 0 else np.nan

            rows.append({
                'Model': model_name,
                'Window_Start': pd.to_datetime(window_df['Date'].iloc[0]).strftime('%Y-%m-%d'),
                'Window_End': pd.to_datetime(window_df['Date'].iloc[-1]).strftime('%Y-%m-%d'),
                'Months': int(len(window_df)),
                'Positives': positives,
                'AUC': auc,
                'PR_AUC': pr_auc,
                'Brier': brier_score_loss(y_true, y_proba),
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Specificity': specificity,
                'Youdens_J': recall + specificity - 1,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
                'TN': int(tn),
            })

    return pd.DataFrame(rows)


def _save_model_selection_artifacts(models_dir: Path, search_result: dict):
    """Persist strict vintage search outputs for auditability."""
    search_df = search_result.get('search_results')
    if search_df is not None and not search_df.empty:
        search_df.to_csv(models_dir / "vintage_search_results.csv", index=False)

    origin_df = search_result.get('origin_results')
    if origin_df is not None and not origin_df.empty:
        origin_df.to_csv(models_dir / "vintage_search_origins.csv", index=False)

    alfred_df = search_result.get('alfred_results')
    if alfred_df is not None and not alfred_df.empty:
        alfred_df.to_csv(models_dir / "vintage_search_alfred.csv", index=False)

    best_candidate = search_result.get('best_candidate')
    if best_candidate:
        with open(models_dir / "vintage_search_best.json", "w") as f:
            json.dump(best_candidate, f, indent=2)

    summary = search_result.get('summary')
    if summary:
        with open(models_dir / "vintage_search_summary.txt", "w") as f:
            f.write(summary)


def _extract_ensemble_metrics(metrics_df: pd.DataFrame) -> dict:
    """Extract ensemble row metrics as plain Python floats."""
    if metrics_df is None or metrics_df.empty or 'Model' not in metrics_df.columns:
        return {}

    ensemble_row = metrics_df[metrics_df['Model'] == 'ensemble']
    if ensemble_row.empty:
        return {}

    fields = [
        'AUC', 'PR_AUC', 'Brier', 'LogLoss',
        'Precision', 'Recall', 'F1', 'Specificity',
    ]
    metrics = {}
    for field in fields:
        if field in ensemble_row.columns:
            raw_value = ensemble_row[field].iloc[0]
            try:
                metrics[field] = float(raw_value)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping invalid incumbent ensemble metric %s=%r.",
                    field,
                    raw_value,
                )
    return metrics


def _resolve_manifest_model_config(manifest: dict) -> dict:
    """Recover the live incumbent config from manifest metadata when available."""
    manifest = manifest or {}
    if manifest.get('model_config') is not None:
        return manifest.get('model_config') or {}

    selection = manifest.get('model_selection') or {}
    applied = selection.get('final_applied_config') or {}
    if applied.get('model_config') is not None:
        return applied.get('model_config') or {}
    if selection.get('model_config') is not None:
        return selection.get('model_config') or {}

    candidate_id = applied.get('candidate_id') or selection.get('candidate_id')
    if candidate_id in DEFAULT_SEARCH_CANDIDATE_MAP:
        return DEFAULT_SEARCH_CANDIDATE_MAP[candidate_id].get('model_config', {}) or {}
    return {}


def _load_incumbent_snapshot(models_dir: Path) -> dict | None:
    """Load the currently live production config and ensemble metrics."""
    metrics_path = models_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    try:
        metrics_df = pd.read_csv(metrics_path)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning(
            "Ignoring incumbent metrics snapshot at %s: %s",
            metrics_path,
            exc,
        )
        return None

    ensemble_metrics = _extract_ensemble_metrics(metrics_df)
    if not ensemble_metrics:
        logger.warning(
            "Ignoring incumbent metrics snapshot at %s: ensemble metrics unavailable.",
            metrics_path,
        )
        return None

    manifest = {}
    manifest_path = models_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Ignoring malformed incumbent run manifest at %s: %s",
                manifest_path,
                exc,
            )
            manifest = {}

    selection = manifest.get('model_selection') or {}
    applied = selection.get('final_applied_config') or {}
    candidate_id = applied.get('candidate_id') or selection.get('candidate_id') or 'baseline_50'
    description = applied.get('description') or selection.get('description') or 'Current live incumbent'
    n_cv_splits = _coerce_int(
        manifest.get(
            'n_cv_splits',
            applied.get(
                'n_cv_splits',
                selection.get(
                    'n_cv_splits',
                    DEFAULT_SEARCH_CANDIDATE_MAP.get(candidate_id, {}).get('n_cv_splits', 5),
                ),
            ),
        ),
        default=5,
        field_name='n_cv_splits',
        source=str(manifest_path),
    )
    max_features = _coerce_int(
        manifest.get('max_features', 50),
        default=50,
        field_name='max_features',
        source=str(manifest_path),
    )
    model_config = _resolve_manifest_model_config(manifest)
    if not isinstance(model_config, dict):
        logger.warning(
            "Invalid model_config in %s (%s); using empty config.",
            manifest_path,
            type(model_config).__name__,
        )
        model_config = {}

    return {
        'candidate_id': candidate_id,
        'description': description,
        'max_features': max_features,
        'n_cv_splits': n_cv_splits,
        'model_config': model_config,
        'metrics': ensemble_metrics,
        'manifest': manifest,
    }


def _same_model_signature(lhs: dict, rhs: dict) -> bool:
    """Check whether two model configs are operationally identical."""
    return (
        int(lhs.get('max_features', 50)) == int(rhs.get('max_features', 50))
        and int(lhs.get('n_cv_splits', 5)) == int(rhs.get('n_cv_splits', 5))
        and json.dumps(lhs.get('model_config') or {}, sort_keys=True)
        == json.dumps(rhs.get('model_config') or {}, sort_keys=True)
    )


def _compare_bundles_for_promotion(candidate_bundle: dict, incumbent_bundle: dict) -> dict:
    """Apply a conservative incumbent safety gate using broad holdout metrics."""
    candidate_metrics = candidate_bundle.get('ensemble_metrics') or {}
    incumbent_metrics = incumbent_bundle.get('ensemble_metrics') or {}

    checks = {}
    passed = True
    improved_any = False
    for metric_name, direction in PROMOTION_GATE_METRICS.items():
        candidate_value = float(candidate_metrics.get(metric_name, np.nan))
        incumbent_value = float(incumbent_metrics.get(metric_name, np.nan))

        if np.isnan(candidate_value) or np.isnan(incumbent_value):
            check_passed = False
            materially_better = False
        elif direction == 'higher':
            check_passed = candidate_value >= incumbent_value - 1e-9
            materially_better = candidate_value > incumbent_value + 1e-6
        else:
            check_passed = candidate_value <= incumbent_value + 1e-9
            materially_better = candidate_value < incumbent_value - 1e-6

        checks[metric_name] = {
            'candidate': candidate_value,
            'incumbent': incumbent_value,
            'direction': direction,
            'passed': check_passed,
        }
        passed = passed and check_passed
        improved_any = improved_any or materially_better

    if passed and improved_any:
        reason = "selected candidate matched or improved all broad holdout guardrails"
    elif passed:
        reason = "selected candidate matched incumbent on all broad holdout guardrails"
    else:
        blocked = [name for name, details in checks.items() if not details['passed']]
        reason = f"selected candidate underperformed incumbent on {', '.join(blocked)}"

    return {
        'passed': passed,
        'reason': reason,
        'checks': checks,
        'candidate_metrics': candidate_metrics,
        'incumbent_metrics': incumbent_metrics,
    }


def _train_model_bundle(*, df_final: pd.DataFrame, df_features: pd.DataFrame,
                        horizon_months: int, train_end_date, max_features: int,
                        threshold_override, model_config=None, n_cv_splits: int = 5,
                        label: str = "model") -> dict:
    """Fit, score, and package a model configuration without writing artifacts."""
    logger.info(
        "Preparing %s configuration | max_features=%s n_cv_splits=%s",
        label,
        max_features,
        n_cv_splits,
    )

    model = RecessionEnsembleModel(
        target_horizon=horizon_months,
        n_cv_splits=n_cv_splits,
        model_config=model_config,
    )
    train_df, test_df = model.prepare_data(df_final, train_end_date=train_end_date)

    target_col = f'RECESSION_FORWARD_{horizon_months}M'
    df_model = df_final.copy()
    feature_cols_for_ffill = [
        c for c in df_model.columns
        if c not in [target_col, 'RECESSION'] and not c.startswith('ref_')
    ]
    df_model[feature_cols_for_ffill] = df_model[feature_cols_for_ffill].ffill()

    nowcast_mask = df_model[target_col].isna()
    nowcast_df = df_model[nowcast_mask].copy()
    nowcast_df = nowcast_df.replace([np.inf, -np.inf], np.nan)

    if len(nowcast_df) > 0:
        logger.info(
            "%s nowcast window: %d months (%s to %s)",
            label,
            len(nowcast_df),
            nowcast_df.index.min().strftime('%Y-%m'),
            nowcast_df.index.max().strftime('%Y-%m'),
        )
    else:
        logger.info("%s nowcast window: none", label)

    model.fit(train_df, max_features=max_features)
    if threshold_override is not None:
        model.decision_threshold = float(threshold_override)
        logger.info("%s applied threshold override: %.3f", label, model.decision_threshold)

    ci_result = model.predict_with_confidence(
        test_df,
        n_bootstrap=200,
        ci_level=0.90,
        method="block_bootstrap",
        block_size_months=12,
        train_df=train_df,
    )
    predictions = ci_result['predictions']
    metrics_df = model.evaluate(test_df, predictions)

    nowcast_ci = None
    nowcast_preds = None
    if len(nowcast_df) > 0:
        nowcast_ci = model.predict_with_confidence(
            nowcast_df,
            n_bootstrap=200,
            ci_level=0.90,
            method="block_bootstrap",
            block_size_months=12,
            train_df=train_df,
        )
        nowcast_preds = nowcast_ci['predictions']
        logger.info(
            "%s nowcast complete — latest probability: %.1f%%",
            label,
            nowcast_preds['ensemble'][-1] * 100,
        )

    data_dict = {
        'Date': test_df.index,
        'Forecast_Horizon': horizon_months,
        'Actual_Recession': test_df[target_col].values,
        'Recession_Current': test_df['RECESSION'].values,
        'Prob_Ensemble': predictions['ensemble'],
        'Prob_Probit': predictions['probit'],
        'Prob_RandomForest': predictions['random_forest'],
        'CI_Lower': ci_result['ensemble_ci_lower'],
        'CI_Upper': ci_result['ensemble_ci_upper'],
        'CI_Std': ci_result['ensemble_std'],
        'Model_Spread': ci_result['model_spread'],
    }
    if 'xgboost' in predictions:
        data_dict['Prob_XGBoost'] = predictions['xgboost']
    if 'markov_switching' in predictions:
        data_dict['Prob_MarkovSwitching'] = predictions['markov_switching']
    if 'lstm' in predictions:
        data_dict['Prob_LSTM'] = predictions['lstm']

    predictions_df = pd.DataFrame(data_dict)

    if len(nowcast_df) > 0:
        nowcast_dict = {
            'Date': nowcast_df.index,
            'Forecast_Horizon': horizon_months,
            'Actual_Recession': np.nan,
            'Recession_Current': nowcast_df['RECESSION'].values,
            'Prob_Ensemble': nowcast_preds['ensemble'],
            'Prob_Probit': nowcast_preds['probit'],
            'Prob_RandomForest': nowcast_preds['random_forest'],
            'CI_Lower': nowcast_ci['ensemble_ci_lower'],
            'CI_Upper': nowcast_ci['ensemble_ci_upper'],
            'CI_Std': nowcast_ci['ensemble_std'],
            'Model_Spread': nowcast_ci['model_spread'],
        }
        if 'xgboost' in nowcast_preds:
            nowcast_dict['Prob_XGBoost'] = nowcast_preds['xgboost']
        if 'markov_switching' in nowcast_preds:
            nowcast_dict['Prob_MarkovSwitching'] = nowcast_preds['markov_switching']
        if 'lstm' in nowcast_preds:
            nowcast_dict['Prob_LSTM'] = nowcast_preds['lstm']

        nowcast_pred_df = pd.DataFrame(nowcast_dict)
        predictions_df = pd.concat([predictions_df, nowcast_pred_df], ignore_index=True)

    predictions_df_dates = pd.to_datetime(predictions_df['Date'])
    for ref_col in ['ref_RECPROUSM156N', 'ref_JHGDPBRINDX']:
        if ref_col in df_features.columns:
            ref_series = df_features[ref_col].reindex(predictions_df_dates)
            if ref_series.notna().any():
                col_name = ref_col.replace('ref_', 'Ref_')
                predictions_df[col_name] = ref_series.values

    rolling_metrics_df = _build_rolling_metrics(predictions_df, threshold=model.decision_threshold)

    return {
        'label': label,
        'model': model,
        'model_config': model_config or {},
        'n_cv_splits': int(n_cv_splits),
        'max_features': int(max_features),
        'train_df': train_df,
        'test_df': test_df,
        'nowcast_df': nowcast_df,
        'ci_result': ci_result,
        'predictions': predictions,
        'metrics_df': metrics_df,
        'ensemble_metrics': _extract_ensemble_metrics(metrics_df),
        'predictions_df': predictions_df,
        'rolling_metrics_df': rolling_metrics_df,
    }


def _persist_model_bundle(*, bundle: dict, models_dir: Path, horizon_months: int,
                          train_end_date, threshold_override, selection_metadata=None):
    """Persist trained model outputs to the canonical production artifact paths."""
    model = bundle['model']
    metrics_df = bundle['metrics_df']
    predictions_df = bundle['predictions_df']
    ci_result = bundle['ci_result']
    rolling_metrics_df = bundle['rolling_metrics_df']

    for name, model_obj in model.models.items():
        filepath = models_dir / f"{name}.pkl"
        joblib.dump(model_obj, filepath)
        logger.info("✓ Saved %s model", name)

    for name, cal_model in model.calibrated_models.items():
        if cal_model is not None:
            filepath = models_dir / f"{name}_calibrated.pkl"
            joblib.dump(cal_model, filepath)
            logger.info("✓ Saved %s calibrated model", name)

    joblib.dump(model.scaler, models_dir / "scaler.pkl")
    logger.info("✓ Saved scaler")

    with open(models_dir / "ensemble_weights.json", 'w') as f:
        json.dump(model.ensemble_weights, f, indent=2)
    logger.info("✓ Saved ensemble weights")

    with open(models_dir / "threshold.json", 'w') as f:
        json.dump({
            'decision_threshold': model.decision_threshold,
            'method': getattr(model, 'threshold_method', 'unknown'),
            'ensemble_method': getattr(model, 'ensemble_method', 'unknown'),
            'active_models': list(getattr(model, 'active_models', [])),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    logger.info("✓ Saved decision threshold (%.3f)", model.decision_threshold)

    threshold_diagnostics = getattr(model, 'threshold_diagnostics', [])
    if threshold_diagnostics:
        threshold_df = pd.DataFrame(threshold_diagnostics)
        threshold_df.to_csv(models_dir / "threshold_sweep.csv", index=False)
        logger.info("✓ Saved threshold sweep (%d rows)", len(threshold_df))

    with open(models_dir / "features.txt", 'w') as f:
        for feature in model.feature_cols:
            f.write(f"{feature}\n")
    logger.info("✓ Saved feature list (%d features)", len(model.feature_cols))

    with open(models_dir / "cv_results.json", 'w') as f:
        cv_serializable = {}
        for name, scores in model.cv_results.items():
            cv_serializable[name] = {k: float(v) for k, v in scores.items()}
        json.dump(cv_serializable, f, indent=2)
    logger.info("✓ Saved cross-validation results")

    metrics_df.to_csv(models_dir / "metrics.csv", index=False)
    logger.info("✓ Saved evaluation metrics")

    save_predictions(predictions_df, bundle['test_df'])

    if rolling_metrics_df is not None and not rolling_metrics_df.empty:
        rolling_metrics_df.to_csv(models_dir / "rolling_metrics.csv", index=False)
        logger.info("✓ Saved rolling metrics (%d rows)", len(rolling_metrics_df))

    run_manifest = _build_run_manifest(
        horizon_months=horizon_months,
        train_end_date=train_end_date,
        max_features=bundle['max_features'],
        threshold_override=threshold_override,
        model=model,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        selection_metadata=selection_metadata,
        model_config=bundle['model_config'],
        n_cv_splits=bundle['n_cv_splits'],
    )
    with open(models_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info("✓ Saved run manifest")

    method_label = ci_result.get('method', 'dirichlet')
    friendly_method = {
        'stationary_block_bootstrap': 'Stationary block bootstrap (Politis-Romano)',
        'dirichlet': 'Dirichlet weight perturbation',
    }.get(method_label, method_label)
    with open(models_dir / "confidence_intervals.json", 'w') as f:
        json.dump({
            'ci_level': ci_result['ci_level'],
            'n_bootstrap': int(ci_result.get('n_bootstrap', 200)),
            'method': friendly_method,
            'method_key': method_label,
            'block_size_months': int(ci_result.get('block_size_months', 12)),
            'empirical_coverage_on_backtest': None,
            'latest_ci_lower': float(ci_result['ensemble_ci_lower'][-1]),
            'latest_ci_upper': float(ci_result['ensemble_ci_upper'][-1]),
            'latest_model_spread': float(ci_result['model_spread'][-1]),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    logger.info("✓ Saved confidence interval metadata (%s)", friendly_method)

    # Persist calibration diagnostics (A1 experiment)
    cal_diag = getattr(model, 'calibration_diagnostics', None) or {}
    cal_choice = getattr(model, 'calibrator_choice', None) or {}
    if cal_diag:
        models_payload = {}
        for name, entry in cal_diag.items():
            entry_copy = dict(entry)
            entry_copy.setdefault('winner', cal_choice.get(name, 'isotonic'))
            models_payload[name] = entry_copy
        with open(models_dir / "calibration_diagnostics.json", 'w') as f:
            json.dump({
                'generated_at_utc': datetime.utcnow().isoformat(),
                'git_sha': _get_git_sha(),
                'models': models_payload,
            }, f, indent=2, default=str)
        logger.info(
            "✓ Saved calibration diagnostics (%d model(s), winners: %s)",
            len(models_payload),
            ", ".join(f"{n}={w}" for n, w in cal_choice.items()),
        )

    report = model.generate_report(bundle['test_df'], bundle['predictions'])
    save_executive_report(report)
    logger.info("✓ Saved executive report")


def run_update_job(horizon_months=None, train_end_date=None, max_features=None,
                   threshold_override=None, strict_vintage_search=False,
                   search_only=False):
    """
    Main update job function

    Args:
        horizon_months: Prediction horizon in months (defaults to runtime config)
        train_end_date: Date to split training/test data.
                        If None, uses expanding window (last 20% as test).
        max_features: Maximum selected features for model fitting (defaults to runtime config)
        threshold_override: Optional manual decision threshold override in [0, 1]
        strict_vintage_search: If True, run strict real-time candidate selection first
        search_only: If True, persist search outputs and exit without final retraining
    """
    runtime_config = load_runtime_config()
    horizon_months = runtime_config['horizon_months'] if horizon_months is None else horizon_months
    train_end_date = runtime_config['train_end_date'] if train_end_date is None else train_end_date
    max_features = runtime_config['max_features'] if max_features is None else max_features
    threshold_override = runtime_config['threshold_override'] if threshold_override is None else threshold_override

    logger.info("=" * 100)
    logger.info("SCHEDULER UPDATE JOB STARTED (v2 — literature-informed)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        "Config | horizon=%s train_end=%s max_features=%s threshold_override=%s strict_vintage_search=%s search_only=%s",
        horizon_months,
        train_end_date,
        max_features,
        threshold_override,
        strict_vintage_search,
        search_only,
    )
    logger.info("=" * 100)

    try:
        # Check for FRED API key
        fred_api_key = os.environ.get('FRED_API_KEY')
        if not fred_api_key:
            raise RuntimeError(
                "FRED_API_KEY environment variable is not set. "
                "Get a free key at https://fred.stlouisfed.org/"
            )

        logger.info("✓ FRED API Key detected")

        # STEP 1: DATA ACQUISITION
        logger.info("=" * 100)
        logger.info("STEP 1: DATA ACQUISITION FROM FRED")
        logger.info("=" * 100)

        acq = RecessionDataAcquisition(fred_api_key=fred_api_key)
        df_raw = acq.fetch_data(start_date='1970-01-01')

        logger.info("✓ Raw data: %d months, %d indicators", df_raw.shape[0], df_raw.shape[1])

        # STEP 2: FEATURE ENGINEERING
        logger.info("=" * 100)
        logger.info("STEP 2: FEATURE ENGINEERING (at-risk, Sahm, spread dynamics)")
        logger.info("=" * 100)

        df_features = acq.engineer_features(df_raw)
        logger.info("✓ Engineered %d total columns", df_features.shape[1])

        # Build GLR composites on a COPY so the ensemble training frame
        # below stays free of GLR_* columns (otherwise feature selection
        # could silently pick them up and truncate the training window).
        try:
            glr_result = GLRRegimeEngine().build(df_features)
            save_glr_components(glr_result['components'])
            df_indicators = df_features.copy()
            for col in glr_result['composites'].columns:
                df_indicators[col] = glr_result['composites'][col]
            for col in glr_result['states'].columns:
                df_indicators[col] = glr_result['states'][col]
            logger.info(
                "✓ GLR composites: %d months, %d composites, %d components",
                len(glr_result['composites']),
                glr_result['composites'].shape[1],
                glr_result['components'].shape[1],
            )
        except Exception as e:
            logger.warning("GLR engine failed, skipping composites: %s", e)
            df_indicators = df_features

        # Save indicators WITH engineered features + GLR composites
        save_indicators(df_indicators)

        # STEP 3: CREATE TARGET
        logger.info("=" * 100)
        logger.info("STEP 3: CREATE FORECAST TARGET (%d-month forward)", horizon_months)
        logger.info("=" * 100)

        df_final = acq.create_forecast_target(df_features, horizon_months=horizon_months)

        ensure_data_dir()
        models_dir = Path(__file__).parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        incumbent_snapshot = _load_incumbent_snapshot(models_dir)
        selection_metadata = None
        selected_model_config = {}
        selected_n_cv_splits = 5
        selected_max_features = max_features

        if strict_vintage_search:
            logger.info("=" * 100)
            logger.info("STEP 3B: STRICT VINTAGE MODEL SEARCH")
            logger.info("=" * 100)
            backtester = RecessionBacktester(acq, RecessionEnsembleModel, target_horizon=horizon_months)
            search_result = backtester.run_model_config_search(df_raw, alfred_top_k=2)
            _save_model_selection_artifacts(models_dir, search_result)
            logger.info("✓ Saved strict vintage search artifacts")

            best_candidate = search_result.get('best_candidate')
            if best_candidate:
                selected_model_config = best_candidate.get('model_config', {}) or {}
                selected_n_cv_splits = int(best_candidate.get('n_cv_splits', 5))
                selected_max_features = int(best_candidate.get('max_features', max_features))
                selection_metrics = best_candidate.get('selection_metrics', {})
                selection_metadata = {
                    'mode': 'strict_vintage_search',
                    'candidate_id': best_candidate.get('id'),
                    'description': best_candidate.get('description', ''),
                    'max_features': selected_max_features,
                    'n_cv_splits': selected_n_cv_splits,
                    'model_config': selected_model_config,
                    'selection_metrics': selection_metrics,
                }
                logger.info(
                    "✓ Selected candidate %s | PR-AUC=%.3f Brier=%.4f F1=%.3f",
                    best_candidate.get('id'),
                    selection_metrics.get('PR_AUC', float('nan')),
                    selection_metrics.get('Brier', float('nan')),
                    selection_metrics.get('F1', float('nan')),
                )

            if search_only:
                logger.info("Search-only mode enabled; exiting after strict vintage search.")
                return True

        # STEP 4: TRAIN / EVALUATE CANDIDATE AND INCUMBENT
        logger.info("=" * 100)
        logger.info("STEP 4: MODEL EVALUATION & PROMOTION GATE")
        if train_end_date:
            logger.info("  Using fixed cutoff: %s", train_end_date)
        else:
            logger.info("  Using expanding window (80/20 split)")
        logger.info("=" * 100)

        candidate_bundle = _train_model_bundle(
            df_final=df_final,
            df_features=df_features,
            horizon_months=horizon_months,
            train_end_date=train_end_date,
            max_features=selected_max_features,
            threshold_override=threshold_override,
            model_config=selected_model_config,
            n_cv_splits=selected_n_cv_splits,
            label="selected candidate" if strict_vintage_search else "production default",
        )

        applied_bundle = candidate_bundle
        applied_candidate_id = selection_metadata.get('candidate_id') if selection_metadata else 'baseline_50'
        applied_description = selection_metadata.get('description', 'Production default') if selection_metadata else 'Production default'

        if strict_vintage_search and selection_metadata and incumbent_snapshot:
            incumbent_config = {
                'candidate_id': incumbent_snapshot.get('candidate_id', 'baseline_50'),
                'description': incumbent_snapshot.get('description', 'Current live incumbent'),
                'max_features': incumbent_snapshot.get('max_features', 50),
                'n_cv_splits': incumbent_snapshot.get('n_cv_splits', 5),
                'model_config': incumbent_snapshot.get('model_config', {}),
            }
            candidate_config = {
                'candidate_id': selection_metadata.get('candidate_id'),
                'description': selection_metadata.get('description', ''),
                'max_features': selected_max_features,
                'n_cv_splits': selected_n_cv_splits,
                'model_config': selected_model_config,
            }

            if _same_model_signature(candidate_config, incumbent_config):
                gate_result = {
                    'passed': True,
                    'reason': 'selected candidate matches incumbent configuration',
                    'candidate_metrics': candidate_bundle['ensemble_metrics'],
                    'incumbent_metrics': candidate_bundle['ensemble_metrics'],
                }
            else:
                incumbent_bundle = _train_model_bundle(
                    df_final=df_final,
                    df_features=df_features,
                    horizon_months=horizon_months,
                    train_end_date=train_end_date,
                    max_features=incumbent_config['max_features'],
                    threshold_override=threshold_override,
                    model_config=incumbent_config['model_config'],
                    n_cv_splits=incumbent_config['n_cv_splits'],
                    label="incumbent guardrail",
                )
                gate_result = _compare_bundles_for_promotion(candidate_bundle, incumbent_bundle)

                if gate_result['passed']:
                    applied_bundle = candidate_bundle
                else:
                    applied_bundle = incumbent_bundle
                    applied_candidate_id = incumbent_config['candidate_id']
                    applied_description = incumbent_config['description']
                    logger.warning(
                        "Promotion gate retained incumbent %s: %s",
                        incumbent_config['candidate_id'],
                        gate_result['reason'],
                    )

            selection_metadata['promotion_gate'] = gate_result
            selection_metadata['promotion_decision'] = (
                'promoted_selected_candidate'
                if gate_result['passed'] else
                'retained_incumbent'
            )
            selection_metadata['final_applied_config'] = {
                'candidate_id': applied_candidate_id,
                'description': applied_description,
                'max_features': applied_bundle['max_features'],
                'n_cv_splits': applied_bundle['n_cv_splits'],
                'model_config': applied_bundle['model_config'],
            }
        elif selection_metadata:
            selection_metadata['promotion_decision'] = 'promoted_selected_candidate'
            selection_metadata['final_applied_config'] = {
                'candidate_id': applied_candidate_id,
                'description': applied_description,
                'max_features': applied_bundle['max_features'],
                'n_cv_splits': applied_bundle['n_cv_splits'],
                'model_config': applied_bundle['model_config'],
            }

        if strict_vintage_search and selection_metadata:
            logger.info(
                "Applying production configuration %s (max_features=%s, n_cv_splits=%s)",
                applied_candidate_id,
                applied_bundle['max_features'],
                applied_bundle['n_cv_splits'],
            )

        model = applied_bundle['model']
        train_df = applied_bundle['train_df']
        test_df = applied_bundle['test_df']
        predictions = applied_bundle['predictions']
        metrics_df = applied_bundle['metrics_df']
        predictions_df = applied_bundle['predictions_df']

        logger.info("=" * 100)
        logger.info("STEP 5: PERSISTING PRODUCTION ARTIFACTS")
        logger.info("=" * 100)
        _persist_model_bundle(
            bundle=applied_bundle,
            models_dir=models_dir,
            horizon_months=horizon_months,
            train_end_date=train_end_date,
            threshold_override=threshold_override,
            selection_metadata=selection_metadata,
        )

        # STEP 9: PSEUDO OUT-OF-SAMPLE BACKTEST
        logger.info("=" * 100)
        logger.info("STEP 9: HISTORICAL BACKTEST (pseudo out-of-sample)")
        logger.info("=" * 100)

        try:
            backtester = RecessionBacktester(acq, type(model), target_horizon=horizon_months)

            backtest_results = backtester.run_pseudo_oos_backtest(
                df_final,
                model_config=applied_bundle['model_config'],
                max_features=applied_bundle['max_features'],
                n_cv_splits=applied_bundle['n_cv_splits'],
            )
            backtest_path = models_dir / "backtest_results.csv"
            backtest_results.to_csv(backtest_path, index=False)
            logger.info("✓ Saved backtest results to %s", backtest_path)

            # Summary
            summary = backtester.summarize_results(backtest_results)
            logger.info("\nBACKTEST SUMMARY:\n%s", summary)

            # Save summary
            with open(models_dir / "backtest_summary.txt", 'w') as f:
                f.write(summary)

            # Optional ALFRED vintage backtest (non-fatal, requires network/method support)
            alfred_results = backtester.run_alfred_vintage_backtest(
                df_raw,
                model_config=applied_bundle['model_config'],
                max_features=applied_bundle['max_features'],
                n_cv_splits=applied_bundle['n_cv_splits'],
            )
            if alfred_results is not None and not alfred_results.empty:
                alfred_path = models_dir / "alfred_vintage_results.csv"
                alfred_results.to_csv(alfred_path, index=False)
                logger.info("✓ Saved ALFRED vintage results to %s", alfred_path)
                alfred_summary = backtester.summarize_alfred_results(alfred_results)
                with open(models_dir / "alfred_vintage_summary.txt", "w") as f:
                    f.write(alfred_summary)
                logger.info("✓ Saved ALFRED vintage summary")

        except Exception as e:
            logger.warning("Backtest failed (non-fatal): %s", e, exc_info=True)

        # STEP 10: MODEL MONITORING
        logger.info("=" * 100)
        logger.info("STEP 10: MODEL MONITORING & DRIFT DETECTION")
        logger.info("=" * 100)

        try:
            monitor = ModelMonitor(data_dir=models_dir)

            # Prepare predictions DataFrame with date index for monitoring
            monitor_df = predictions_df.copy()
            monitor_df['Date'] = pd.to_datetime(monitor_df['Date'])
            monitor_df = monitor_df.set_index('Date').sort_index()

            monitor_report = monitor.run_all_checks(
                predictions_df=monitor_df,
                indicators_df=df_features,
                feature_cols=model.feature_cols,
            )
            monitor.save_report(monitor_report)

            if monitor_report['alert_count'] > 0:
                logger.warning("⚠ Model monitoring raised %d alert(s):", monitor_report['alert_count'])
                for alert in monitor_report['alerts']:
                    logger.warning("  [%s] %s: %s", alert['level'], alert['check'], alert['message'])
            else:
                logger.info("✓ All monitoring checks passed (no alerts)")

        except Exception as e:
            logger.warning("Model monitoring failed (non-fatal): %s", e, exc_info=True)

        # FINAL SUMMARY
        logger.info("=" * 100)
        logger.info("UPDATE JOB COMPLETED SUCCESSFULLY!")
        logger.info("=" * 100)

        try:
            ensemble_row = metrics_df[metrics_df['Model'] == 'ensemble']
            if not ensemble_row.empty:
                auc = ensemble_row['AUC'].iloc[0]
                brier = ensemble_row['Brier'].iloc[0]
                logloss = ensemble_row['LogLoss'].iloc[0]
                logger.info("Ensemble AUC: %.3f | Brier: %.4f | LogLoss: %.4f", auc, brier, logloss)
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not retrieve ensemble metrics: {str(e)}")

        try:
            # Use nowcast probability if available, otherwise fall back to test set
            latest_nowcast = predictions_df[predictions_df['Actual_Recession'].isna()]
            if not latest_nowcast.empty:
                latest_prob = float(latest_nowcast['Prob_Ensemble'].iloc[-1])
                latest_date = pd.to_datetime(latest_nowcast['Date'].iloc[-1]).strftime('%Y-%m')
            elif 'ensemble' in predictions and len(predictions['ensemble']) > 0:
                latest_prob = predictions['ensemble'][-1]
                latest_date = test_df.index[-1].strftime('%Y-%m')
            else:
                latest_prob = None
                latest_date = 'N/A'

            if latest_prob is not None:
                logger.info(
                    "Current %dM Recession Probability (%s): %.1f%% (threshold: %.1f%%)",
                    horizon_months, latest_date, latest_prob * 100, model.decision_threshold * 100
                )
            else:
                logger.warning("Latest probability not available")
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not retrieve latest probability: {str(e)}")

        # Log ensemble weights
        logger.info("Ensemble weights: %s",
                     ", ".join(f"{n}={w:.3f}" for n, w in model.ensemble_weights.items()))
        logger.info(
            "Decision threshold: %.3f (%s)",
            model.decision_threshold,
            getattr(model, 'threshold_method', 'unknown'),
        )
        logger.info("=" * 100)

        return True

    except Exception as exc:
        logger.exception("Fatal error during update job: %s", exc)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recession prediction update job")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in months")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Training data end date (YYYY-MM-DD). If not set, uses expanding window.")
    parser.add_argument("--max-features", type=int, default=None, help="Maximum selected features")
    parser.add_argument("--threshold-override", type=float, default=None,
                        help="Optional manual decision threshold override in [0,1]")
    parser.add_argument("--strict-vintage-search", action="store_true",
                        help="Run strict real-time candidate selection before final training")
    parser.add_argument("--search-only", action="store_true",
                        help="Persist strict vintage search artifacts and exit without retraining")

    args = parser.parse_args()

    success = run_update_job(
        horizon_months=args.horizon,
        train_end_date=args.train_end,
        max_features=args.max_features,
        threshold_override=args.threshold_override,
        strict_vintage_search=args.strict_vintage_search,
        search_only=args.search_only,
    )

    sys.exit(0 if success else 1)
