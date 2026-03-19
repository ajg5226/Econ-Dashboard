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

        # STEP 7: SAVE PREDICTIONS (with confidence intervals)
        logger.info("=" * 100)
        logger.info("STEP 7: SAVING PREDICTIONS")
        logger.info("=" * 100)

        data_dict = {
            'Date': test_df.index,
            'Actual_Recession': test_df['RECESSION'].values,
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
            if 'ensemble' in predictions and len(predictions['ensemble']) > 0:
                latest_prob = predictions['ensemble'][-1]
                logger.info(
                    "Current %dM Recession Probability: %.1f%% (threshold: %.1f%%)",
                    horizon_months, latest_prob * 100, model.decision_threshold * 100
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
