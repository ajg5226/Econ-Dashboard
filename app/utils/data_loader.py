"""
Data loading utilities for the web app
Handles reading/writing predictions and indicators to persistent storage
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Base data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def ensure_data_dir():
    """Ensure data directory structure exists"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "models").mkdir(exist_ok=True)
    (DATA_DIR / "reports").mkdir(exist_ok=True)
    (DATA_DIR / "logs").mkdir(exist_ok=True)


def save_predictions(predictions_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save predictions to CSV file
    
    Args:
        predictions_df: DataFrame with predictions (Date, Actual_Recession, Prob_*)
        test_df: Original test DataFrame with index
    """
    ensure_data_dir()
    filepath = DATA_DIR / "predictions.csv"
    predictions_df.to_csv(filepath, index=False)
    logger.info(f"Saved predictions to {filepath}")


def load_predictions() -> pd.DataFrame:
    """
    Load predictions from CSV file
    
    Returns:
        DataFrame with predictions, or empty DataFrame if file doesn't exist
    """
    filepath = DATA_DIR / "predictions.csv"
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            # BUG FIX 16: Handle missing Date column
            if 'Date' not in df.columns:
                logger.error(f"Date column missing from {filepath}")
                return pd.DataFrame()
            # BUG FIX 17: Handle invalid dates gracefully
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Remove rows with invalid dates
            df = df.dropna(subset=['Date'])
            return df
        except Exception as e:
            logger.error(f"Error loading predictions from {filepath}: {str(e)}")
            return pd.DataFrame()
    else:
        logger.warning(f"Predictions file not found: {filepath}")
        return pd.DataFrame()


def save_indicators(indicators_df: pd.DataFrame):
    """
    Save indicator data to CSV file
    
    Args:
        indicators_df: DataFrame with indicator values
    """
    ensure_data_dir()
    filepath = DATA_DIR / "indicators.csv"
    indicators_df.to_csv(filepath)
    logger.info(f"Saved indicators to {filepath}")


def load_indicators() -> pd.DataFrame:
    """
    Load indicator data from CSV file
    
    Returns:
        DataFrame with indicators, or empty DataFrame if file doesn't exist
    """
    filepath = DATA_DIR / "indicators.csv"
    if filepath.exists():
        try:
            # BUG FIX 18: Handle parse_dates errors
            df = pd.read_csv(filepath, index_col=0)
            # Try to parse dates if index is string
            if df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # Remove rows with invalid dates
                    df = df[df.index.notna()]
                except Exception:
                    logger.warning(f"Could not parse dates in indicators file")
            return df
        except Exception as e:
            logger.error(f"Error loading indicators from {filepath}: {str(e)}")
            return pd.DataFrame()
    else:
        logger.warning(f"Indicators file not found: {filepath}")
        return pd.DataFrame()


def save_executive_report(report_text: str):
    """
    Save executive report to text file
    
    Args:
        report_text: Report content as string
    """
    ensure_data_dir()
    filepath = DATA_DIR / "reports" / "executive_report.txt"
    with open(filepath, 'w') as f:
        f.write(report_text)
    logger.info(f"Saved executive report to {filepath}")


def load_executive_report() -> str:
    """
    Load executive report from text file
    
    Returns:
        Report content as string, or empty string if file doesn't exist
    """
    filepath = DATA_DIR / "reports" / "executive_report.txt"
    if filepath.exists():
        with open(filepath, 'r') as f:
            return f.read()
    else:
        logger.warning(f"Executive report not found: {filepath}")
        return ""


def get_last_update_time() -> datetime:
    """
    Get the last update time from predictions file
    
    Returns:
        Last modification time of predictions.csv, or None if file doesn't exist
    """
    filepath = DATA_DIR / "predictions.csv"
    if filepath.exists():
        return datetime.fromtimestamp(filepath.stat().st_mtime)
    return None


def is_data_stale(days_threshold: int = 7) -> bool:
    """
    Check if data is stale (older than threshold)
    
    Args:
        days_threshold: Number of days before data is considered stale
        
    Returns:
        True if data is stale or missing, False otherwise
    """
    last_update = get_last_update_time()
    if last_update is None:
        return True
    
    days_old = (datetime.now() - last_update).days
    return days_old > days_threshold

