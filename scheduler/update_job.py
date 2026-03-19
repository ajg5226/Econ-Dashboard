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


def run_update_job(horizon_months=6, train_end_date='2015-12-31'):
    """
    Main update job function
    
    Args:
        horizon_months: Prediction horizon in months
        train_end_date: Date to split training/test data
    """
    logger.info("=" * 100)
    logger.info("SCHEDULER UPDATE JOB STARTED")
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
        
        # Save raw indicators
        save_indicators(df_raw)
        
        # STEP 2: FEATURE ENGINEERING
        logger.info("=" * 100)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 100)
        
        df_features = acq.engineer_features(df_raw)
        logger.info("✓ Engineered %d features", df_features.shape[1])
        
        # STEP 3: CREATE TARGET
        logger.info("=" * 100)
        logger.info("STEP 3: CREATE FORECAST TARGET")
        logger.info("=" * 100)
        
        df_final = acq.create_forecast_target(df_features, horizon_months=horizon_months)
        
        # STEP 4: TRAIN/TEST SPLIT
        logger.info("=" * 100)
        logger.info("STEP 4: TRAIN/TEST SPLIT")
        logger.info("=" * 100)
        
        model = RecessionEnsembleModel(target_horizon=horizon_months)
        train_df, test_df = model.prepare_data(df_final, train_end_date=train_end_date)
        
        # STEP 5: MODEL TRAINING
        logger.info("=" * 100)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 100)
        
        model.fit(train_df)
        
        # Save model artifacts
        ensure_data_dir()
        models_dir = Path(__file__).parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model_obj in model.models.items():
            filepath = models_dir / f"{name}.pkl"
            joblib.dump(model_obj, filepath)
            logger.info(f"✓ Saved {name} model")
        
        # Save meta-model
        joblib.dump(model.meta_model, models_dir / "meta_model.pkl")
        logger.info("✓ Saved meta-model")
        
        # Save scaler
        joblib.dump(model.scaler, models_dir / "scaler.pkl")
        logger.info("✓ Saved scaler")
        
        # Save feature list
        with open(models_dir / "features.txt", 'w') as f:
            for feature in model.feature_cols:
                f.write(f"{feature}\n")
        logger.info("✓ Saved feature list")
        
        # STEP 6: PREDICTION & EVALUATION
        logger.info("=" * 100)
        logger.info("STEP 6: PREDICTION & EVALUATION")
        logger.info("=" * 100)
        
        predictions = model.predict(test_df)
        metrics_df = model.evaluate(test_df, predictions)
        
        # STEP 7: SAVE PREDICTIONS
        logger.info("=" * 100)
        logger.info("STEP 7: SAVING PREDICTIONS")
        logger.info("=" * 100)
        
        data_dict = {
            'Date': test_df.index,
            'Actual_Recession': test_df['RECESSION'].values,
            'Prob_Ensemble': predictions['ensemble'],
            'Prob_Probit': predictions['probit'],
            'Prob_RandomForest': predictions['random_forest'],
        }
        if 'xgboost' in predictions:
            data_dict['Prob_XGBoost'] = predictions['xgboost']
        
        predictions_df = pd.DataFrame(data_dict)
        save_predictions(predictions_df, test_df)
        
        # STEP 8: GENERATE REPORT
        logger.info("=" * 100)
        logger.info("STEP 8: GENERATING EXECUTIVE REPORT")
        logger.info("=" * 100)
        
        report = model.generate_report(test_df, predictions)
        save_executive_report(report)
        
        # FINAL SUMMARY
        logger.info("=" * 100)
        logger.info("UPDATE JOB COMPLETED SUCCESSFULLY!")
        logger.info("=" * 100)
        # BUG FIX 25: Handle missing ensemble model or empty predictions
        try:
            ensemble_row = metrics_df[metrics_df['Model'] == 'ensemble']
            if not ensemble_row.empty and 'AUC' in ensemble_row.columns:
                best_auc = ensemble_row['AUC'].iloc[0]
                logger.info("🎯 Ensemble AUC: %.3f", best_auc)
            else:
                logger.warning("⚠️ Ensemble AUC not available")
        except (KeyError, IndexError) as e:
            logger.warning(f"⚠️ Could not retrieve ensemble AUC: {str(e)}")
        
        try:
            if 'ensemble' in predictions and len(predictions['ensemble']) > 0:
                latest_prob = predictions['ensemble'][-1]
                logger.info("📈 Current %dM Recession Probability: %.1f%%", horizon_months, latest_prob * 100)
            else:
                logger.warning("⚠️ Latest probability not available")
        except (KeyError, IndexError) as e:
            logger.warning(f"⚠️ Could not retrieve latest probability: {str(e)}")
        
        logger.info("=" * 100)
        
        return True
        
    except Exception as exc:
        logger.exception("Fatal error during update job: %s", exc)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run recession prediction update job")
    parser.add_argument("--horizon", type=int, default=6, help="Prediction horizon in months")
    parser.add_argument("--train-end", type=str, default='2015-12-31', help="Training data end date")
    
    args = parser.parse_args()
    
    success = run_update_job(
        horizon_months=args.horizon,
        train_end_date=args.train_end
    )
    
    sys.exit(0 if success else 1)

