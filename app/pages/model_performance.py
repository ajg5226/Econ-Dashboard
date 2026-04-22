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

# Load backtest results
backtest_df = None
backtest_file = DATA_DIR / "backtest_results.csv"
if backtest_file.exists():
    try:
        backtest_df = pd.read_csv(backtest_file)
    except Exception:
        pass

# Load backtest summary
backtest_summary = ""
summary_file = DATA_DIR / "backtest_summary.txt"
if summary_file.exists():
    try:
        with open(summary_file) as f:
            backtest_summary = f.read()
    except Exception:
        pass

# Load confidence interval metadata
ci_info = {}
ci_file = DATA_DIR / "confidence_intervals.json"
if ci_file.exists():
    try:
        with open(ci_file) as f:
            ci_info = json.load(f)
    except Exception:
        pass

# Load run manifest / provenance metadata
run_manifest = {}
manifest_file = DATA_DIR / "run_manifest.json"
if manifest_file.exists():
    try:
        with open(manifest_file) as f:
            run_manifest = json.load(f)
    except Exception:
        pass

# Load ALFRED vintage summary (if available)
alfred_summary = ""
alfred_summary_file = DATA_DIR / "alfred_vintage_summary.txt"
if alfred_summary_file.exists():
    try:
        with open(alfred_summary_file) as f:
            alfred_summary = f.read()
    except Exception:
        pass

# Load threshold sweep diagnostics (if available)
threshold_sweep_df = None
threshold_sweep_file = DATA_DIR / "threshold_sweep.csv"
if threshold_sweep_file.exists():
    try:
        threshold_sweep_df = pd.read_csv(threshold_sweep_file)
    except Exception:
        pass

# Load rolling performance diagnostics (if available)
rolling_metrics_df = None
rolling_metrics_file = DATA_DIR / "rolling_metrics.csv"
if rolling_metrics_file.exists():
    try:
        rolling_metrics_df = pd.read_csv(rolling_metrics_file)
    except Exception:
        pass

# ── Decision Threshold Info ───────────────────────────────────────────────────
threshold = threshold_info.get('decision_threshold', 0.5)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Decision Threshold", f"{threshold:.3f}",
              help="Optimized on calibrated training probabilities using an F1 objective with precision/recall tie-breaks")
with col2:
    st.metric("Optimization Method", threshold_info.get('method', 'Default 0.5'))
with col3:
    n_models = len(ensemble_weights) if ensemble_weights else 'N/A'
    st.metric("Ensemble Models", n_models)

if run_manifest:
    st.markdown("---")
    st.markdown("### Run Provenance")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Horizon", f"{run_manifest.get('horizon_months', 'N/A')}M")
    with p2:
        st.metric("Max Features", run_manifest.get('max_features', 'N/A'))
    with p3:
        st.metric("Selected Features", run_manifest.get('selected_features_count', 'N/A'))
    with p4:
        ts = run_manifest.get('timestamp_utc', '')
        st.metric("Run Timestamp", ts[:19] if ts else 'N/A')
    st.caption(f"Git SHA: `{run_manifest.get('git_sha', 'unknown')}`")
    active_models = run_manifest.get('active_models', [])
    if active_models:
        st.caption(
            "Ensemble method: "
            f"`{run_manifest.get('ensemble_method', 'unknown')}` | "
            f"Active models: `{', '.join(active_models)}`"
        )

# ── Ensemble Weights ──────────────────────────────────────────────────────────
if ensemble_weights:
    st.markdown("---")
    st.markdown("### Ensemble Weights")
    st.markdown("*Live ensemble blend after CV gating; current defaults favor an equal-weight active subset over noisy fully optimized weights*")

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

if threshold_sweep_df is not None and not threshold_sweep_df.empty:
    st.markdown("---")
    st.markdown("### Threshold Sweep Diagnostics")
    chosen_row = threshold_sweep_df[
        threshold_sweep_df['threshold'].round(2) == round(float(threshold), 2)
    ]
    if not chosen_row.empty:
        chosen = chosen_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Chosen Threshold", f"{chosen['threshold']:.2f}")
        with c2:
            st.metric("Chosen F1", f"{chosen.get('f1', 0):.3f}")
        with c3:
            st.metric("Chosen Precision", f"{chosen.get('precision', 0):.3f}")

    top_thresholds = threshold_sweep_df.sort_values(
        ['score', 'precision', 'specificity'],
        ascending=[False, False, False],
    ).head(10)
    st.dataframe(top_thresholds, use_container_width=True, hide_index=True)

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

# ── Calibration Diagnostics (A1) ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### Calibration Diagnostics")
st.markdown(
    "*Reliability curves, ECE, calibration slope/intercept, and Brier "
    "decomposition per model, plus the isotonic/sigmoid/beta A/B winner. "
    "Source: `data/models/calibration_diagnostics.json`*"
)

cal_diag_file = DATA_DIR / "calibration_diagnostics.json"
cal_diag = {}
if cal_diag_file.exists():
    try:
        with open(cal_diag_file) as f:
            cal_diag = json.load(f)
    except Exception:
        cal_diag = {}

if not cal_diag or not cal_diag.get("models"):
    st.info(
        "Run `python -m scheduler.update_job` to populate calibration diagnostics."
    )
else:
    cal_models = cal_diag.get("models", {})

    # Reliability curve for ensemble
    ensemble_entry = cal_models.get("ensemble", {})
    reliability_raw = ensemble_entry.get("raw", {}).get("reliability", []) if ensemble_entry else []
    if HAS_PLOTLY and reliability_raw:
        bins = np.array(reliability_raw, dtype=float)
        if bins.ndim == 2 and bins.shape[1] == 4:
            rel_fig = go.Figure()
            rel_fig.add_trace(
                go.Scatter(
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="Perfect calibration",
                    showlegend=True,
                )
            )
            rel_fig.add_trace(
                go.Scatter(
                    x=bins[:, 2],
                    y=bins[:, 1],
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(color="#d62728"),
                    name="Ensemble (raw CV)",
                )
            )
            rel_fig.update_layout(
                title="Ensemble reliability curve (quantile bins)",
                xaxis_title="Mean predicted probability",
                yaxis_title="Observed recession frequency",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(rel_fig, use_container_width=True)

    # ECE per model (bar)
    ece_rows = []
    slope_rows = []
    winner_rows = []
    for name, entry in cal_models.items():
        raw = entry.get("raw", {}) if isinstance(entry, dict) else {}
        ece_rows.append({"Model": name, "ECE": raw.get("ece")})
        slope_rows.append({
            "Model": name,
            "Slope": raw.get("slope"),
            "Intercept": raw.get("intercept"),
            "Brier_Total": raw.get("brier_total"),
            "Brier_Reliability": raw.get("brier_reliability"),
            "Brier_Resolution": raw.get("brier_resolution"),
            "N_Samples": raw.get("n_samples"),
        })
        winner_rows.append({
            "Model": name,
            "Winner": entry.get("winner", "isotonic"),
            "Isotonic_ECE": (entry.get("isotonic") or {}).get("ece"),
            "Sigmoid_ECE": (entry.get("sigmoid") or {}).get("ece"),
            "Beta_ECE": (entry.get("beta") or {}).get("ece"),
        })

    if HAS_PLOTLY and ece_rows:
        ece_df_plot = pd.DataFrame(ece_rows).dropna()
        if not ece_df_plot.empty:
            ece_fig = go.Figure()
            ece_fig.add_trace(
                go.Bar(
                    x=ece_df_plot["Model"].astype(str),
                    y=ece_df_plot["ECE"].astype(float),
                    text=[f"{v:.3f}" for v in ece_df_plot["ECE"].astype(float)],
                    textposition="outside",
                    marker_color="#1f77b4",
                )
            )
            ece_fig.update_layout(
                title="Expected Calibration Error (raw CV) per model",
                yaxis_title="ECE (lower is better)",
                template="plotly_white",
                height=360,
            )
            st.plotly_chart(ece_fig, use_container_width=True)

    slope_df = pd.DataFrame(slope_rows)
    if not slope_df.empty:
        st.markdown("**Slope / intercept & Brier decomposition**")
        st.dataframe(slope_df, use_container_width=True, hide_index=True)

    winner_df = pd.DataFrame(winner_rows)
    if not winner_df.empty:
        st.markdown("**Calibrator A/B (holdout ECE — winner per model)**")
        st.dataframe(winner_df, use_container_width=True, hide_index=True)
        st.caption(
            "Measurement only — production calibrator stays isotonic until the "
            "validator confirms a promotion on vintage origins."
        )

    generated = cal_diag.get("generated_at_utc", "")
    git_sha = cal_diag.get("git_sha", "unknown")
    st.caption(
        f"Generated at {generated[:19]} UTC | Git SHA: `{git_sha}` | "
        "CI coverage on backtest: (pending — populated by validator)"
    )

# ── Confusion Matrices ────────────────────────────────────────────────────────
if 'Actual_Recession' in predictions_df.columns:
    st.markdown("---")
    st.markdown("### Confusion Matrices")
    st.markdown(f"*At threshold = {threshold:.3f}*")

    prob_cols = [col for col in predictions_df.columns if col.startswith('Prob_')]

    if prob_cols:
        # Filter to rows with known actuals (exclude nowcast months where Actual_Recession is NaN)
        eval_df = predictions_df.dropna(subset=['Actual_Recession'])
        if len(eval_df) == 0:
            st.warning("No rows with known recession outcomes available for confusion matrices.")
        else:
            cols = st.columns(len(prob_cols))

            for idx, prob_col in enumerate(prob_cols):
                with cols[idx]:
                    model_name = prob_col.replace('Prob_', '').replace('_', ' ').title()
                    st.markdown(f"**{model_name}**")

                    y_true = eval_df['Actual_Recession'].values.astype(int)
                    y_pred_proba = eval_df[prob_col].values.astype(float)
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
        known_df = predictions_df.dropna(subset=['Actual_Recession'])
        recession_count = int(known_df['Actual_Recession'].sum())
        nowcast_count = len(predictions_df) - len(known_df)
        st.info(f"**Positive (recession within 6M):** {recession_count}")
        st.info(f"**Negative (no recession within 6M):** {len(known_df) - recession_count}")
        if nowcast_count > 0:
            st.info(f"**Nowcast Months (outcome unknown):** {nowcast_count}")
        if len(known_df) > 0:
            st.info(f"**Positive Rate:** {recession_count / len(known_df):.1%}")

    if rolling_metrics_df is not None and not rolling_metrics_df.empty:
        st.markdown("---")
        st.markdown("### Rolling Window Diagnostics")
        latest_windows = (
            rolling_metrics_df
            .sort_values(['Model', 'Window_End'])
            .groupby('Model', as_index=False)
            .tail(1)
        )
        st.dataframe(latest_windows, use_container_width=True, hide_index=True)

    # ── Confidence Intervals ─────────────────────────────────────────────────
    if ci_info:
        st.markdown("---")
        st.markdown("### Confidence Intervals (Current)")
        ci_cols = st.columns(4)
        with ci_cols[0]:
            st.metric("CI Level", f"{ci_info.get('ci_level', 0.9):.0%}")
        with ci_cols[1]:
            st.metric("Lower Bound", f"{ci_info.get('latest_ci_lower', 0):.1%}")
        with ci_cols[2]:
            st.metric("Upper Bound", f"{ci_info.get('latest_ci_upper', 0):.1%}")
        with ci_cols[3]:
            st.metric("Model Spread", f"{ci_info.get('latest_model_spread', 0):.1%}",
                      help="Max - min probability across base models (epistemic uncertainty)")
        st.caption(f"Method: {ci_info.get('method', 'N/A')} ({ci_info.get('n_bootstrap', 'N/A')} samples)")

    # ── Model Monitoring & Drift Detection ────────────────────────────────────
    monitor_report = {}
    monitor_file = DATA_DIR / "monitor_report.json"
    if monitor_file.exists():
        try:
            with open(monitor_file) as f:
                monitor_report = json.load(f)
        except Exception:
            pass

    if monitor_report:
        st.markdown("---")
        st.markdown("### Model Monitoring")

        status = monitor_report.get('status', 'UNKNOWN')
        alert_count = monitor_report.get('alert_count', 0)
        timestamp = monitor_report.get('timestamp', 'N/A')

        if status == 'OK':
            st.success(f"**Status: ✅ All Checks Passed** — Last run: {timestamp[:19]}")
        else:
            st.warning(f"**Status: ⚠️ {alert_count} Alert(s)** — Last run: {timestamp[:19]}")

        checks = monitor_report.get('checks', {})

        # Display checks in a grid
        mon_cols = st.columns(3)

        # Prediction stability
        stability = checks.get('prediction_stability', {})
        with mon_cols[0]:
            st.markdown("**Prediction Stability**")
            details = stability.get('details', {})
            vol = details.get('current_6m_vol')
            mom = details.get('last_mom_change')
            if vol is not None:
                st.metric("6M Rolling Vol", f"{vol:.4f}")
            if mom is not None:
                st.metric("Last MoM Change", f"{mom:.1%}")

        # Model disagreement
        disagreement = checks.get('model_disagreement', {})
        with mon_cols[1]:
            st.markdown("**Model Disagreement**")
            details = disagreement.get('details', {})
            spread = details.get('current_spread')
            if spread is not None:
                st.metric("Current Spread", f"{spread:.1%}")
            preds = details.get('model_predictions', {})
            if preds:
                for m, p in preds.items():
                    st.caption(f"{m}: {p:.1%}")

        # Feature drift
        drift = checks.get('feature_drift', {})
        with mon_cols[2]:
            st.markdown("**Feature Drift (PSI)**")
            details = drift.get('details', {})
            checked = details.get('features_checked', 0)
            drifted = details.get('features_drifted', 0)
            mean_psi = details.get('mean_psi', 0)
            if checked > 0:
                st.metric("Features Checked", checked)
                st.metric("Drifted (PSI > 0.20)", drifted)
                st.caption(f"Mean PSI: {mean_psi:.4f}")
            top_drifted = details.get('top_drifted', {})
            if top_drifted:
                st.caption("Top drifted: " + ", ".join(f"{k}={v}" for k, v in list(top_drifted.items())[:3]))

        # Show alerts if any
        alerts = monitor_report.get('alerts', [])
        if alerts:
            with st.expander(f"⚠️ {len(alerts)} Alert(s)", expanded=True):
                for alert in alerts:
                    level = alert.get('level', 'INFO')
                    icon = '🔴' if level == 'WARNING' else 'ℹ️'
                    st.markdown(f"{icon} **[{alert.get('check', '')}]** {alert.get('message', '')}")

    # ── Historical Backtest Results ───────────────────────────────────────────
    if backtest_df is not None and not backtest_df.empty:
        st.markdown("---")
        st.markdown("### Historical Backtest (Pseudo Out-of-Sample)")
        st.markdown("*Each recession tested by training ONLY on data before it occurred*")

        if backtest_summary:
            st.info(backtest_summary)

        # Show results table
        display_cols = [c for c in ['Recession', 'AUC', 'Brier', 'Peak_Prob', 'Peak_Date',
                                     'Crossed_Threshold', 'Lead_Months', 'Threshold']
                        if c in backtest_df.columns]
        bt_display = backtest_df[display_cols].copy()

        # Format
        for col in ['AUC', 'Brier']:
            if col in bt_display.columns:
                bt_display[col] = bt_display[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        if 'Peak_Prob' in bt_display.columns:
            bt_display['Peak_Prob'] = bt_display['Peak_Prob'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        if 'Lead_Months' in bt_display.columns:
            bt_display['Lead_Months'] = bt_display['Lead_Months'].apply(
                lambda x: f"{x:.0f}mo" if pd.notna(x) else "N/A")
        if 'Threshold' in bt_display.columns:
            bt_display['Threshold'] = bt_display['Threshold'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        st.dataframe(bt_display, use_container_width=True, hide_index=True)

        # Highlight GFC result
        gfc_rows = backtest_df[backtest_df['Recession'].str.contains('GFC', na=False)]
        if not gfc_rows.empty:
            gfc = gfc_rows.iloc[0]
            crossed = gfc.get('Crossed_Threshold', False)
            peak = gfc.get('Peak_Prob', 0)
            if crossed:
                st.success(f"**GFC Detection:** Peak probability {peak:.1%} — crossed threshold. "
                           f"Lead time: {gfc.get('Lead_Months', 'N/A'):.0f} months before recession.")
            else:
                st.warning(f"**GFC Detection:** Peak probability {peak:.1%} — did NOT cross threshold.")

    if alfred_summary:
        st.markdown("---")
        st.markdown("### ALFRED Vintage Evaluation")
        st.info(alfred_summary)

    # ── Methodology ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Methodology & References"):
        st.markdown("""
        ### Model Architecture
        - **Base Models:** L1-Probit (Estrella-Mishkin style), Random Forest (300 trees), XGBoost (400 rounds)
        - **Class Weighting:** All models use balanced class weights to handle ~12% recession base rate
        - **Calibration:** Isotonic regression via time-series cross-validation (avoids SMOTE miscalibration)
        - **Ensemble:** CV-gated forecast combination with a default equal-weight active subset
        - **Threshold:** F1 optimization on calibrated training probabilities with precision/recall tie-breaks

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
