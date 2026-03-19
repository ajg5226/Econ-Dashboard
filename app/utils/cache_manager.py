"""
Cache management utilities for Streamlit
Handles caching strategies and cache invalidation
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_cached() -> pd.DataFrame:
    """
    Load predictions with Streamlit caching (1 hour TTL)
    
    Returns:
        DataFrame with predictions
    """
    # BUG FIX 19: Fix circular import by using absolute import
    try:
        from app.utils.data_loader import load_predictions
    except ImportError:
        from utils.data_loader import load_predictions
    return load_predictions()


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_indicators_cached() -> pd.DataFrame:
    """
    Load indicators with Streamlit caching (24 hour TTL)
    
    Returns:
        DataFrame with indicators
    """
    # BUG FIX 20: Fix circular import
    try:
        from app.utils.data_loader import load_indicators
    except ImportError:
        from utils.data_loader import load_indicators
    return load_indicators()


@st.cache_resource  # Cache until manual refresh
def load_model_artifacts():
    """
    Load model artifacts (models, scaler, features) with persistent caching
    
    Returns:
        Dictionary with model artifacts or None if not available
    """
    import joblib
    models_dir = DATA_DIR / "models"
    
    artifacts = {}
    try:
        # Try to load model files
        for model_name in ['probit', 'random_forest', 'xgboost', 'meta_model', 'scaler']:
            filepath = models_dir / f"{model_name}.pkl"
            if filepath.exists():
                artifacts[model_name] = joblib.load(filepath)
        
        # Load feature list
        features_file = models_dir / "features.txt"
        if features_file.exists():
            with open(features_file, 'r') as f:
                artifacts['features'] = [line.strip() for line in f.readlines()]
        
        if artifacts:
            logger.info(f"Loaded {len(artifacts)} model artifacts")
            return artifacts
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
    
    return None


def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Cleared all Streamlit caches")


def get_cache_info() -> dict:
    """
    Get information about cached data
    
    Returns:
        Dictionary with cache information
    """
    predictions_file = DATA_DIR / "predictions.csv"
    indicators_file = DATA_DIR / "indicators.csv"
    
    info = {
        'predictions_exists': predictions_file.exists(),
        'indicators_exists': indicators_file.exists(),
        'predictions_size': 0,
        'indicators_size': 0,
    }
    
    # BUG FIX: Handle file access errors
    try:
        if predictions_file.exists():
            info['predictions_size'] = predictions_file.stat().st_size
            info['predictions_last_modified'] = datetime.fromtimestamp(
                predictions_file.stat().st_mtime
            )
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access predictions file: {str(e)}")
    
    try:
        if indicators_file.exists():
            info['indicators_size'] = indicators_file.stat().st_size
            info['indicators_last_modified'] = datetime.fromtimestamp(
                indicators_file.stat().st_mtime
            )
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access indicators file: {str(e)}")
    
    return info

