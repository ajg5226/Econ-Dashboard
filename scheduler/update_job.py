"""
Background job for data refresh and model retraining
Can be run via cron, GitHub Actions, or manually
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

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
from recession_engine.ensemble_model import RecessionEnsembleModel
try:
    from app.utils.data_loader import (
        save_predictions, save_indicators, save_executive_report,
        ensure_data_dir
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.utils.data_loader import (
        save_predictions, save_indicators, save_executive_report,
        ensure_data_dir
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


def run_update_job(horizon_months=6, train_end_date=None):
    """
    Main update job function

    Args:
        horizon_months: Prediction horizon in months
        train_end_date: Date to split training/test data.
                        If None, uses expanding window (last 20% as test).
                        Default changed from fixed '2015-12-31' to None.
    """
    logger.info("=" * 100)
    logger.info("SCHEDULER UPDATE JOB STARTED (v2 — literature-informed)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

        # Save indicators WITH engineered features (needed by Indicators page)
        save_indicators(df_features)

        # STEP 3: CREATE TARGET
        logger.info("=" * 100)
        logger.info("STEP 3: CREATE FORECAST TARGET (%d-month forward)", horizon_months)
        logger.info("=" * 100)

        df_final = acq.create_forecast_target(df_features, horizon_months=horizon_months)

        # STEP 4: TRAIN/TEST SPLIT
        logger.info("=" * 100)
        logger.info("STEP 4: TRAIN/TEST SPLIT")
        if train_end_date:
            logger.info("  Using fixed cutoff: %s", train_end_date)
        else:
            logger.info("  Using expanding window (80/20 split)")
        logger.info("=" * 100)

        model = RecessionEnsembleModel(target_horizon=horizon_months)
        train_df, test_df = model.prepare_data(df_final, train_end_date=train_end_date)

        # STEP 4b: BUILD NOWCAST SET
        # The test_df only contains rows with known forward targets. For a
        # production dashboard we also need predictions on the most recent
        # months where RECESSION_FORWARD_6M is NaN (the "nowcast" window).
        #
        # Critical: forward-fill features BEFORE slicing so that the last
        # known indicator values carry into the nowcast months. Without this,
        # features like PAYEMS/INDPRO that haven't been published yet would
        # be NaN, and predict()'s .fillna(0) would replace them with zero —
        # which the model interprets as economic collapse.
        target_col = f'RECESSION_FORWARD_{horizon_months}M'
        feature_cols_for_ffill = [c for c in df_final.columns
                                  if c not in [target_col, 'RECESSION']
                                  and not c.startswith('ref_')]
        df_final[feature_cols_for_ffill] = df_final[feature_cols_for_ffill].ffill()

        nowcast_mask = df_final[target_col].isna()
        nowcast_df = df_final[nowcast_mask].copy()
        nowcast_df = nowcast_df.replace([np.inf, -np.inf], np.nan)

        if len(nowcast_df) > 0:
            logger.info("Nowcast window: %d months (%s to %s)",
                        len(nowcast_df),
                        nowcast_df.index.min().strftime('%Y-%m'),
                        nowcast_df.index.max().strftime('%Y-%m'))
        else:
            logger.info("No nowcast months (all months have known targets)")

        # STEP 5: MODEL TRAINING (with calibration + threshold optimization)
        logger.info("=" * 100)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 100)

        model.fit(train_df)

        # Save model artifacts
        ensure_data_dir()
        models_dir = Path(__file__).parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save base models
        for name, model_obj in model.models.items():
            filepath = models_dir / f"{name}.pkl"
            joblib.dump(model_obj, filepath)
            logger.info(f"✓ Saved {name} model")

        # Save calibrated models
        for name, cal_model in model.calibrated_models.items():
            if cal_model is not None:
                filepath = models_dir / f"{name}_calibrated.pkl"
                joblib.dump(cal_model, filepath)
                logger.info(f"✓ Saved {name} calibrated model")

        # Save scaler
        joblib.dump(model.scaler, models_dir / "scaler.pkl")
        logger.info("✓ Saved scaler")

        # Save ensemble weights
        import json
        with open(models_dir / "ensemble_weights.json", 'w') as f:
            json.dump(model.ensemble_weights, f, indent=2)
        logger.info("✓ Saved ensemble weights")

        # Save decision threshold
        with open(models_dir / "threshold.json", 'w') as f:
            json.dump({
                'decision_threshold': model.decision_threshold,
                'method': "Youden's J statistic",
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        logger.info("✓ Saved decision threshold (%.3f)", model.decision_threshold)

        # Save feature list
        with open(models_dir / "features.txt", 'w') as f:
            for feature in model.feature_cols:
                f.write(f"{feature}\n")
        logger.info("✓ Saved feature list (%d features)", len(model.feature_cols))

        # Save CV results
        with open(models_dir / "cv_results.json", 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            cv_serializable = {}
            for name, scores in model.cv_results.items():
                cv_serializable[name] = {k: float(v) for k, v in scores.items()}
            json.dump(cv_serializable, f, indent=2)
        logger.info("✓ Saved cross-validation results")

        # STEP 6: PREDICTION & EVALUATION
        logger.info("=" * 100)
        logger.info("STEP 6: PREDICTION & EVALUATION")
        logger.info("=" * 100)

        # STEP 6a: PREDICTIONS WITH CONFIDENCE INTERVALS
        logger.info("=" * 100)
        logger.info("STEP 6: PREDICTION WITH CONFIDENCE INTERVALS")
        logger.info("=" * 100)

        ci_result = model.predict_with_confidence(test_df, n_bootstrap=200, ci_level=0.90)
        predictions = ci_result['predictions']
        metrics_df = model.evaluate(test_df, predictions)

        # Save metrics
        metrics_df.to_csv(models_dir / "metrics.csv", index=False)
        logger.info("✓ Saved evaluation metrics")

        # STEP 6b: NOWCAST PREDICTIONS (most recent months without known targets)
        if len(nowcast_df) > 0:
            logger.info("Generating nowcast predictions for %d months...", len(nowcast_df))
            nowcast_ci = model.predict_with_confidence(nowcast_df, n_bootstrap=200, ci_level=0.90)
            nowcast_preds = nowcast_ci['predictions']
            logger.info("✓ Nowcast complete — latest probability: %.1f%%",
                        nowcast_preds['ensemble'][-1] * 100)

        # STEP 7: SAVE PREDICTIONS (with confidence intervals)
        logger.info("=" * 100)
        logger.info("STEP 7: SAVING PREDICTIONS")
        logger.info("=" * 100)

        # Build test-set predictions
        # IMPORTANT: Actual_Recession must be the FORWARD target (RECESSION_FORWARD_6M),
        # not the current month's RECESSION flag. The model predicts P(recession in next
        # 6 months), so evaluation must compare against whether a recession actually
        # occurred in the next 6 months.
        target_col = f'RECESSION_FORWARD_{horizon_months}M'
        data_dict = {
            'Date': test_df.index,
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

        predictions_df = pd.DataFrame(data_dict)

        # Append nowcast rows (most recent months, Actual_Recession = NaN)
        if len(nowcast_df) > 0:
            nowcast_dict = {
                'Date': nowcast_df.index,
                'Actual_Recession': np.nan,  # Forward target unknown for nowcast
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

            nowcast_pred_df = pd.DataFrame(nowcast_dict)
            predictions_df = pd.concat([predictions_df, nowcast_pred_df], ignore_index=True)
            logger.info("✓ Appended %d nowcast months (through %s)",
                        len(nowcast_df), nowcast_df.index.max().strftime('%Y-%m'))

        # Add peer/reference model probabilities if available in the raw data
        predictions_df_dates = pd.to_datetime(predictions_df['Date'])
        for ref_col in ['ref_RECPROUSM156N', 'ref_JHGDPBRINDX']:
            if ref_col in df_features.columns:
                ref_series = df_features[ref_col].reindex(predictions_df_dates)
                if ref_series.notna().any():
                    col_name = ref_col.replace('ref_', 'Ref_')
                    predictions_df[col_name] = ref_series.values
                    logger.info(f"✓ Added reference series: {col_name}")

        save_predictions(predictions_df, test_df)

        # Save CI metadata
        with open(models_dir / "confidence_intervals.json", 'w') as f:
            json.dump({
                'ci_level': ci_result['ci_level'],
                'n_bootstrap': 200,
                'method': 'Dirichlet weight perturbation',
                'latest_ci_lower': float(ci_result['ensemble_ci_lower'][-1]),
                'latest_ci_upper': float(ci_result['ensemble_ci_upper'][-1]),
                'latest_model_spread': float(ci_result['model_spread'][-1]),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        logger.info("✓ Saved confidence interval metadata")

        # STEP 8: GENERATE REPORT
        logger.info("=" * 100)
        logger.info("STEP 8: GENERATING EXECUTIVE REPORT")
        logger.info("=" * 100)

        report = model.generate_report(test_df, predictions)
        save_executive_report(report)

        # STEP 9: PSEUDO OUT-OF-SAMPLE BACKTEST
        logger.info("=" * 100)
        logger.info("STEP 9: HISTORICAL BACKTEST (pseudo out-of-sample)")
        logger.info("=" * 100)

        try:
            from recession_engine.backtester import RecessionBacktester
            backtester = RecessionBacktester(acq, type(model), target_horizon=horizon_months)

            backtest_results = backtester.run_pseudo_oos_backtest(df_final)
            backtest_path = models_dir / "backtest_results.csv"
            backtest_results.to_csv(backtest_path, index=False)
            logger.info("✓ Saved backtest results to %s", backtest_path)

            # Summary
            summary = backtester.summarize_results(backtest_results)
            logger.info("\nBACKTEST SUMMARY:\n%s", summary)

            # Save summary
            with open(models_dir / "backtest_summary.txt", 'w') as f:
                f.write(summary)

        except Exception as e:
            logger.warning("Backtest failed (non-fatal): %s", e)
            import traceback
            traceback.print_exc()

        # STEP 10: MODEL MONITORING
        logger.info("=" * 100)
        logger.info("STEP 10: MODEL MONITORING & DRIFT DETECTION")
        logger.info("=" * 100)

        try:
            from recession_engine.model_monitor import ModelMonitor
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
            logger.warning("Model monitoring failed (non-fatal): %s", e)

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
            if len(nowcast_df) > 0 and 'ensemble' in nowcast_preds:
                latest_prob = nowcast_preds['ensemble'][-1]
                latest_date = nowcast_df.index[-1].strftime('%Y-%m')
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
        logger.info("Decision threshold: %.3f (Youden's J)", model.decision_threshold)
        logger.info("=" * 100)

        return True

    except Exception as exc:
        logger.exception("Fatal error during update job: %s", exc)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run recession prediction update job")
    parser.add_argument("--horizon", type=int, default=6, help="Prediction horizon in months")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Training data end date (YYYY-MM-DD). If not set, uses expanding window.")

    args = parser.parse_args()

    success = run_update_job(
        horizon_months=args.horizon,
        train_end_date=args.train_end
    )

    sys.exit(0 if success else 1)
