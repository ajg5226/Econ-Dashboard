"""
Model Performance Page
Displays model evaluation metrics, calibration analysis, and ensemble weights
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import json
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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Check authentication
authenticated, username, name = check_authentication()
if not authenticated:
    st.stop()

st.title("Model Performance")
st.markdown("### Evaluation Metrics, Calibration, and Ensemble Analysis")

# Load predictions
predictions_df = load_predictions_cached()

if predictions_df.empty:
    st.warning("No prediction data available. Please run the data refresh in Settings.")
    st.stop()

# Convert Date to datetime and set as index
predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
predictions_df = predictions_df.set_index('Date').sort_index()

# ── Load model metadata ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "models"

# Load saved metrics if available
saved_metrics = None
metrics_file = DATA_DIR / "metrics.csv"
if metrics_file.exists():
    try:
        saved_metrics = pd.read_csv(metrics_file)
    except Exception:
        pass

# Load ensemble weights
ensemble_weights = {}
weights_file = DATA_DIR / "ensemble_weights.json"
if weights_file.exists():
    try:
        with open(weights_file) as f:
            ensemble_weights = json.load(f)
    except Exception:
        pass

# Load threshold
threshold_info = {}
threshold_file = DATA_DIR / "threshold.json"
if threshold_file.exists():
    try:
        with open(threshold_file) as f:
            threshold_info = json.load(f)
    except Exception:
        pass

# Load CV results
cv_results = {}
cv_file = DATA_DIR / "cv_results.json"
if cv_file.exists():
    try:
        with open(cv_file) as f:
            cv_results = json.load(f)
    except Exception:
        pass

# ── Decision Threshold Info ───────────────────────────────────────────────────
threshold = threshold_info.get('decision_threshold', 0.5)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Decision Threshold", f"{threshold:.3f}",
              help="Optimized via Youden's J statistic (maximizes sensitivity + specificity)")
with col2:
    st.metric("Optimization Method", threshold_info.get('method', 'Default 0.5'))
with col3:
    n_models = len(ensemble_weights) if ensemble_weights else 'N/A'
    st.metric("Ensemble Models", n_models)

# ── Ensemble Weights ──────────────────────────────────────────────────────────
if ensemble_weights:
    st.markdown("---")
    st.markdown("### Ensemble Weights (Performance-Weighted)")
    st.markdown("*Weights derived from inverse Brier score on time-series cross-validation (BMA-inspired)*")

    weight_cols = st.columns(len(ensemble_weights))
    for i, (model_name, weight) in enumerate(ensemble_weights.items()):
        with weight_cols[i]:
            display_name = model_name.replace('_', ' ').title()
            st.metric(display_name, f"{weight:.1%}")

# ── Cross-Validation Results ─────────────────────────────────────────────────
if cv_results:
    st.markdown("---")
    st.markdown("### Cross-Validation Results (Time-Series CV)")

    cv_rows = []
    for model_name, scores in cv_results.items():
        cv_rows.append({
            'Model': model_name.replace('_', ' ').title(),
            'CV AUC': scores.get('auc', 0),
            'CV Brier': scores.get('brier', 0),
        })
    cv_df = pd.DataFrame(cv_rows)
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

# ── Performance Metrics Table ─────────────────────────────────────────────────
if saved_metrics is not None and not saved_metrics.empty:
    st.markdown("---")
    st.markdown("### Out-of-Sample Performance Metrics")
    st.markdown(f"*Evaluated at threshold = {threshold:.3f}*")

    # Format the metrics for display
    display_metrics = saved_metrics.copy()
    display_metrics['Model'] = display_metrics['Model'].str.replace('_', ' ').str.title()

    # Round numeric columns
    numeric_cols = display_metrics.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        display_metrics[col] = display_metrics[col].round(4)

    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    # ── Metrics Visualization ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Performance Comparison")

    if HAS_PLOTLY:
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('AUC-ROC', 'Brier Score (lower=better)', 'Log Loss (lower=better)',
                            'Sensitivity (Recall)', 'Specificity', "Youden's J"),
        )

        metrics_to_plot = [
            ('AUC', 1, 1), ('Brier', 1, 2), ('LogLoss', 1, 3),
            ('Sensitivity', 2, 1), ('Specificity', 2, 2), ('Youdens_J', 2, 3)
        ]

        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

        for metric, row, col in metrics_to_plot:
            if metric in saved_metrics.columns:
                fig.add_trace(
                    go.Bar(
                        x=saved_metrics['Model'].str.replace('_', ' ').str.title(),
                        y=saved_metrics[metric],
                        marker_color=colors[:len(saved_metrics)],
                        text=[f'{v:.3f}' for v in saved_metrics[metric]],
                        textposition='outside',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_model_performance(saved_metrics)
        st.plotly_chart(fig, use_container_width=True)

    # ── Best Model Highlight ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Best Performing Model")

    if 'AUC' in saved_metrics.columns:
        best_auc_idx = saved_metrics['AUC'].idxmax()
        best_auc = saved_metrics.loc[best_auc_idx]
        st.success(f"**Best AUC:** {best_auc['Model'].replace('_', ' ').title()} — {best_auc['AUC']:.4f}")

    if 'Brier' in saved_metrics.columns:
        best_brier_idx = saved_metrics['Brier'].idxmin()
        best_brier = saved_metrics.loc[best_brier_idx]
        st.success(f"**Best Calibration (Brier):** {best_brier['Model'].replace('_', ' ').title()} — {best_brier['Brier']:.4f}")

    if 'Youdens_J' in saved_metrics.columns:
        best_j_idx = saved_metrics['Youdens_J'].idxmax()
        best_j = saved_metrics.loc[best_j_idx]
        st.success(f"**Best Youden's J:** {best_j['Model'].replace('_', ' ').title()} — {best_j['Youdens_J']:.4f}")

# ── Confusion Matrices ────────────────────────────────────────────────────────
if 'Actual_Recession' in predictions_df.columns:
    st.markdown("---")
    st.markdown("### Confusion Matrices")
    st.markdown(f"*At threshold = {threshold:.3f}*")

    prob_cols = [col for col in predictions_df.columns if col.startswith('Prob_')]

    if prob_cols:
        cols = st.columns(len(prob_cols))

        for idx, prob_col in enumerate(prob_cols):
            with cols[idx]:
                model_name = prob_col.replace('Prob_', '').replace('_', ' ').title()
                st.markdown(f"**{model_name}**")

                y_true = predictions_df['Actual_Recession'].values
                y_pred_proba = predictions_df[prob_col].values
                y_pred = (y_pred_proba >= threshold).astype(int)

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                cm_df = pd.DataFrame(
                    cm,
                    index=['No Recession', 'Recession'],
                    columns=['Predicted No', 'Predicted Yes']
                )
                st.dataframe(cm_df, use_container_width=True)

                tn, fp, fn, tp = cm.ravel()
                st.caption(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # ── Data Split Info ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Data Split Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Test Observations:** {len(predictions_df)}")
        st.info(f"**Date Range:** {predictions_df.index.min().strftime('%Y-%m-%d')} to {predictions_df.index.max().strftime('%Y-%m-%d')}")

    with col2:
        recession_count = int(predictions_df['Actual_Recession'].sum())
        st.info(f"**Recession Months:** {recession_count}")
        st.info(f"**Non-Recession Months:** {len(predictions_df) - recession_count}")
        if len(predictions_df) > 0:
            st.info(f"**Recession Rate:** {recession_count / len(predictions_df):.1%}")

    # ── Methodology ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Methodology & References"):
        st.markdown("""
        ### Model Architecture
        - **Base Models:** L1-Probit (Estrella-Mishkin style), Random Forest (300 trees), XGBoost (400 rounds)
        - **Class Weighting:** All models use balanced class weights to handle ~12% recession base rate
        - **Calibration:** Isotonic regression via time-series cross-validation (avoids SMOTE miscalibration)
        - **Ensemble:** Performance-weighted average using inverse Brier score (BMA-inspired)
        - **Threshold:** Youden's J statistic — maximizes (sensitivity + specificity - 1)

        ### Feature Engineering
        - **Standard:** MoM/3M/6M/YoY percent changes, 3M/6M rolling means, 6M rolling volatility
        - **Term Spread Dynamics:** Inversion flag, depth, duration, momentum (Engstrom & Sharpe 2019)
        - **Sahm Rule:** 3M unemployment MA rise above 12M trailing low (Sahm 2019)
        - **At-Risk Transformation:** Expanding-window percentile flags (Billakanti & Shin 2025, Philly Fed)
        - **Credit Stress Index:** Standardized composite of Baa spread + TED spread
        - **Monetary Policy Stance:** FFR × term spread interaction (Wright 2006)

        ### Key References
        - Estrella & Mishkin (1998). *Predicting U.S. Recessions: Financial Variables as Leading Indicators*
        - Wright (2006). *The Yield Curve and Predicting Recessions*
        - Gilchrist & Zakrajsek (2012). *Credit Spreads and Business Cycle Fluctuations*
        - Sahm (2019). *Direct Stimulus Payments to Individuals*
        - Billakanti & Shin (2025). *At-Risk Transformation for U.S. Recession Prediction*
        - Engstrom & Sharpe (2019). *The Near-Term Forward Yield Spread as a Leading Indicator*
        """)

else:
    st.warning("Actual recession data not available. Cannot calculate performance metrics.")
