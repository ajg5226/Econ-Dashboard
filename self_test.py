"""
Quick self-test for the recession engine without hitting the FRED API.

This script:
  - Builds a tiny synthetic monthly dataset with a few indicators + RECESSION
  - Runs the full modeling pipeline (feature engineering, target, prepare_data, fit, predict, evaluate)
  - Prints metrics and confirms that everything runs end-to-end

Usage:
    python self_test.py
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from data_acquisition import RecessionDataAcquisition
from ensemble_model import RecessionEnsembleModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_synthetic_dataset():
    """Create a simple synthetic monthly dataset for testing."""
    dates = pd.date_range(start="2000-01-01", periods=240, freq="ME")  # 20 years

    rng = np.random.default_rng(42)

    # Simple synthetic leading / coincident / lagging proxies
    leading = np.cumsum(rng.normal(0, 1, size=len(dates)))
    coincident = np.cumsum(rng.normal(0, 0.7, size=len(dates)))
    lagging = np.cumsum(rng.normal(0, 0.5, size=len(dates)))

    # Define a couple of "recession" windows
    recession = np.zeros(len(dates), dtype=int)
    # 2001 mini recession
    recession[(dates >= "2001-03-01") & (dates <= "2002-02-28")] = 1
    # 2008 GFC-style event
    recession[(dates >= "2008-09-01") & (dates <= "2009-06-30")] = 1
    # Late-cycle slowdown to ensure test split contains positive class examples
    recession[(dates >= "2017-03-01") & (dates <= "2018-01-31")] = 1

    df = pd.DataFrame(
        {
            "leading_USSLIND": leading,
            "coincident_INDPRO": coincident,
            "lagging_CPIAUCSL": lagging,
            "RECESSION": recession,
        },
        index=dates,
    )
    return df


def main():
    logger.info("=" * 80)
    logger.info("RUNNING SELF-TEST WITH SYNTHETIC DATA")
    logger.info("=" * 80)

    df_raw = build_synthetic_dataset()
    logger.info(f"Synthetic raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Use RecessionDataAcquisition methods for feature / target logic
    dummy_acq = RecessionDataAcquisition(fred_api_key="DUMMY_KEY")
    df_features = dummy_acq.engineer_features(df_raw)
    df_final = dummy_acq.create_forecast_target(df_features, horizon_months=6)

    model = RecessionEnsembleModel(target_horizon=6)
    train_df, test_df = model.prepare_data(df_final)

    model.fit(train_df)
    preds = model.predict(test_df)
    metrics = model.evaluate(test_df, preds)

    logger.info("Self-test metrics:")
    logger.info("\n%s", metrics.to_string(index=False))

    latest_prob = preds["ensemble"][-1]
    logger.info(
        "Latest synthetic 6M recession probability as of %s: %.1f%%",
        test_df.index[-1].strftime("%Y-%m-%d"),
        latest_prob * 100,
    )

    logger.info("SELF-TEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()


