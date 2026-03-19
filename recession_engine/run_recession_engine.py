#!/usr/bin/env python3
"""
Goldman Sachs Recession Prediction Engine
Main Execution Script

Usage:
    export FRED_API_KEY='your_api_key_here'
    python run_recession_engine.py
"""

import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: E402
import warnings

from data_acquisition import RecessionDataAcquisition
from ensemble_model import RecessionEnsembleModel

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


def main():
    """Main execution pipeline"""
    logger.info("=" * 100)
    logger.info("GOLDMAN SACHS RECESSION PREDICTION ENGINE")
    logger.info("=" * 100)
    
    # Get API key (must be set in environment)
    fred_api_key = os.environ.get('FRED_API_KEY')
    if not fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY environment variable is not set. "
            "Get a free key at https://fred.stlouisfed.org/ and export it, e.g. "
            "export FRED_API_KEY='your_api_key_here'"
        )
    
    logger.info("✓ FRED API Key detected (length %d characters)", len(fred_api_key))
    
    # STEP 1: DATA ACQUISITION
    logger.info("=" * 100)
    logger.info("STEP 1: DATA ACQUISITION FROM FRED")
    logger.info("=" * 100)
    
    acq = RecessionDataAcquisition(fred_api_key=fred_api_key)
    df_raw = acq.fetch_data(start_date='1970-01-01')
    
    logger.info("✓ Raw data: %d months, %d indicators", df_raw.shape[0], df_raw.shape[1])
    
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
    
    df_final = acq.create_forecast_target(df_features, horizon_months=6)
    
    # STEP 4: TRAIN/TEST SPLIT
    logger.info("=" * 100)
    logger.info("STEP 4: TRAIN/TEST SPLIT")
    logger.info("=" * 100)
    
    model = RecessionEnsembleModel(target_horizon=6)
    train_df, test_df = model.prepare_data(df_final, train_end_date='2015-12-31')
    
    # STEP 5: MODEL TRAINING
    logger.info("=" * 100)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("=" * 100)
    
    model.fit(train_df)
    
    # STEP 6: PREDICTION & EVALUATION
    logger.info("=" * 100)
    logger.info("STEP 6: PREDICTION & EVALUATION")
    logger.info("=" * 100)
    
    predictions = model.predict(test_df)
    metrics_df = model.evaluate(test_df, predictions)
    
    # STEP 7: VISUALIZATION
    logger.info("=" * 100)
    logger.info("STEP 7: GENERATING VISUALIZATIONS")
    logger.info("=" * 100)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Recession probability
    ax1 = axes[0]
    ax1.plot(test_df.index, predictions['ensemble'], 
             label='Ensemble', linewidth=2, color='darkred')
    ax1.plot(test_df.index, predictions['probit'], 
             label='Probit', linewidth=1.5, alpha=0.7, color='blue')
    if 'xgboost' in predictions:
        ax1.plot(test_df.index, predictions['xgboost'], 
                 label='XGBoost', linewidth=1.5, alpha=0.7, color='green')
    
    recession_periods = test_df[test_df['RECESSION'] == 1]
    if len(recession_periods) > 0:
        for idx in recession_periods.index:
            ax1.axvspan(idx, idx, alpha=0.2, color='gray')
    
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recession Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Recession Probability - Out-of-Sample (2016-Present)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Model comparison
    ax2 = axes[1]
    models = metrics_df['Model'].tolist()
    aucs = metrics_df['AUC'].tolist()
    # Generate a color list matching number of models
    base_colors = ['blue', 'green', 'orange', 'darkred', 'purple', 'cyan']
    colors = base_colors[: len(models)]
    
    bars = ax2.bar(models, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance (AUC)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/recession_probability.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Visualization saved: output/recession_probability.png")
    
    # STEP 8: GENERATE REPORT
    logger.info("=" * 100)
    logger.info("STEP 8: EXECUTIVE REPORT")
    logger.info("=" * 100)
    
    report = model.generate_report(test_df, predictions)
    
    with open('output/executive_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Executive report:\n%s", report)
    logger.info("✓ Report saved: output/executive_report.txt")
    
    # STEP 9: DASHBOARD DATA
    logger.info("=" * 100)
    logger.info("STEP 9: DASHBOARD DATA")
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
    dashboard_df = pd.DataFrame(data_dict)
    
    dashboard_df.to_csv('output/dashboard_data.csv', index=False)
    logger.info("✓ Dashboard data saved: output/dashboard_data.csv")
    logger.info("Latest predictions (last 5 rows):\n%s", dashboard_df.tail(5).to_string(index=False))
    
    # FINAL SUMMARY
    logger.info("=" * 100)
    logger.info("DEPLOYMENT COMPLETE!")
    logger.info("=" * 100)
    best_auc = metrics_df.loc[metrics_df['Model'] == 'ensemble', 'AUC'].values[0]
    latest_prob = predictions['ensemble'][-1]
    logger.info("🎯 Ensemble AUC: %.3f", best_auc)
    logger.info("📈 Current 6M Recession Probability: %.1f%%", latest_prob * 100)
    logger.info("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Fatal error during recession engine run: %s", exc)
        sys.exit(1)
