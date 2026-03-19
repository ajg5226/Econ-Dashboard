"""
Model Performance Page
Displays model evaluation metrics and performance comparisons
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.utils.data_loader import load_predictions
    from app.utils.cache_manager import load_predictions_cached
    from app.utils.plotting import plot_model_performance
    from app.auth import check_authentication
except ImportError:
    from utils.data_loader import load_predictions
    from utils.cache_manager import load_predictions_cached
    from utils.plotting import plot_model_performance
    from auth import check_authentication

# Check authentication
authenticated, username, name = check_authentication()
if not authenticated:
    st.stop()

st.title("🎯 Model Performance")
st.markdown("### Evaluation Metrics and Comparison")

# Load predictions
predictions_df = load_predictions_cached()

if predictions_df.empty:
    st.warning("⚠️ No prediction data available. Please run the data refresh in Settings.")
    st.stop()

# Convert Date to datetime and set as index
predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
predictions_df = predictions_df.set_index('Date').sort_index()

# Calculate metrics if we have actual recession data
if 'Actual_Recession' in predictions_df.columns:
    # Get prediction columns
    prob_cols = [col for col in predictions_df.columns if col.startswith('Prob_')]
    
    # Calculate metrics for each model
    metrics_list = []
    threshold = 0.5
    
    for prob_col in prob_cols:
        model_name = prob_col.replace('Prob_', '').replace('_', ' ').title()
        y_true = predictions_df['Actual_Recession'].values
        y_pred_proba = predictions_df[prob_col].values
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
        
        # BUG FIX 13: Better error handling for AUC calculation
        try:
            # Check if we have both classes
            if len(set(y_true)) < 2:
                auc = 0.0
            else:
                auc = roc_auc_score(y_true, y_pred_proba)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not calculate AUC for {model_name}: {str(e)}")
            auc = 0.0
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics_list.append({
            'Model': model_name,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Display metrics table
    st.markdown("### Performance Metrics")
    st.dataframe(metrics_df, use_container_width=True)
    
    # Plot performance comparison
    st.markdown("---")
    st.markdown("### Performance Comparison")
    
    fig = plot_model_performance(metrics_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.markdown("---")
    st.markdown("### Confusion Matrices")
    
    # Create confusion matrices for each model
    # BUG FIX 10: Handle empty prob_cols list
    if not prob_cols:
        st.warning("⚠️ No prediction columns found")
        st.stop()
    
    cols = st.columns(len(prob_cols))
    
    for idx, prob_col in enumerate(prob_cols):
        with cols[idx]:
            model_name = prob_col.replace('Prob_', '').replace('_', ' ').title()
            st.markdown(f"**{model_name}**")
            
            y_true = predictions_df['Actual_Recession'].values
            y_pred_proba = predictions_df[prob_col].values
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            # Display as table
            cm_df = pd.DataFrame(
                cm,
                index=['No Recession', 'Recession'],
                columns=['Predicted No', 'Predicted Yes']
            )
            st.dataframe(cm_df, use_container_width=True)
            
            # Calculate and display metrics
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    # Training/Test Split Info
    st.markdown("---")
    st.markdown("### Data Split Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Total Observations:** {len(predictions_df)}")
        st.info(f"**Date Range:** {predictions_df.index.min().strftime('%Y-%m-%d')} to {predictions_df.index.max().strftime('%Y-%m-%d')}")
    
    with col2:
        recession_count = predictions_df['Actual_Recession'].sum()
        st.info(f"**Recession Periods:** {recession_count}")
        st.info(f"**Non-Recession Periods:** {len(predictions_df) - recession_count}")
    
    # Highlight best model
    st.markdown("---")
    st.markdown("### Best Performing Model")
    
    # BUG FIX 11: Handle empty metrics_df or missing AUC column
    if metrics_df.empty or 'AUC' not in metrics_df.columns:
        st.warning("⚠️ No metrics available to determine best model")
    else:
        try:
            best_idx = metrics_df['AUC'].idxmax()
            if pd.isna(best_idx):
                st.warning("⚠️ Could not determine best model")
            else:
                best_model = metrics_df.loc[best_idx]
                st.success(f"🏆 **{best_model['Model']}** achieves the highest AUC score of **{best_model['AUC']:.4f}**")
        except (KeyError, IndexError) as e:
            st.warning(f"⚠️ Error determining best model: {str(e)}")
    
    # Ensemble highlight
    # BUG FIX 12: Handle missing Model column or empty results
    if not metrics_df.empty and 'Model' in metrics_df.columns:
        if 'Ensemble' in metrics_df['Model'].values:
            ensemble_df = metrics_df[metrics_df['Model'] == 'Ensemble']
            if not ensemble_df.empty:
                ensemble_metrics = ensemble_df.iloc[0]
                st.info(f"📊 **Ensemble Model** combines all base models with AUC: **{ensemble_metrics['AUC']:.4f}**")
    
else:
    st.warning("Actual recession data not available. Cannot calculate performance metrics.")
    st.info("Performance metrics require both predictions and actual recession indicators.")

