"""
Plotting utilities for Streamlit dashboard
Creates interactive charts using Plotly and matplotlib
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    # Fallback to matplotlib if plotly not available
    import matplotlib.pyplot as plt


def plot_recession_probability(
    df: pd.DataFrame,
    predictions: dict,
    start_date: str = None,
    end_date: str = None,
    ci_lower: np.ndarray = None,
    ci_upper: np.ndarray = None,
    peer_models: dict = None,
    threshold: float = None,
):
    """
    Create interactive recession probability chart
    
    Args:
        df: DataFrame with Date index and RECESSION column
        predictions: Dictionary with model predictions
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        Plotly figure object (or matplotlib figure if plotly not available)
    """
    if not HAS_PLOTLY:
        # Fallback to matplotlib
        return _plot_recession_probability_matplotlib(df, predictions, start_date, end_date)
    # Filter by date range if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    fig = go.Figure()
    
    # Plot ensemble prediction (main line)
    # BUG FIX: Handle length mismatch between df and predictions
    if 'ensemble' in predictions:
        pred_array = predictions['ensemble']
        # Ensure predictions match dataframe length
        if len(pred_array) != len(df):
            logger.warning(f"Prediction length ({len(pred_array)}) doesn't match dataframe length ({len(df)})")
            # Truncate or pad as needed
            if len(pred_array) > len(df):
                pred_array = pred_array[:len(df)]
            else:
                # Pad with last value
                pred_array = np.pad(pred_array, (0, len(df) - len(pred_array)), mode='edge')
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=pred_array,
            mode='lines',
            name='Ensemble',
            line=dict(color='darkred', width=3),
            hovertemplate='<b>Ensemble</b><br>Date: %{x}<br>Probability: %{y:.1%}<extra></extra>'
        ))

    # Plot confidence interval band if provided
    if ci_lower is not None and ci_upper is not None:
        ci_lo = np.array(ci_lower)
        ci_hi = np.array(ci_upper)
        # Adjust length to match df
        if len(ci_lo) != len(df):
            if len(ci_lo) > len(df):
                ci_lo = ci_lo[:len(df)]
                ci_hi = ci_hi[:len(df)]
            else:
                ci_lo = np.pad(ci_lo, (0, len(df) - len(ci_lo)), mode='edge')
                ci_hi = np.pad(ci_hi, (0, len(df) - len(ci_hi)), mode='edge')

        fig.add_trace(go.Scatter(
            x=df.index, y=ci_hi, mode='lines',
            line=dict(width=0), showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=ci_lo, mode='lines',
            line=dict(width=0), showlegend=True,
            name='90% CI',
            fill='tonexty',
            fillcolor='rgba(178,34,34,0.15)',
            hoverinfo='skip',
        ))

    # Plot base model predictions
    model_colors = {
        'probit': 'blue',
        'random_forest': 'green',
        'xgboost': 'orange',
        'markov_switching': 'purple',
    }
    model_display_names = {
        'probit': 'Probit',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'markov_switching': 'Markov Switching',
    }

    for model_name in ['probit', 'random_forest', 'xgboost', 'markov_switching']:
        if model_name in predictions:
            pred_array = np.array(predictions[model_name])
            # Ensure length matches
            if len(pred_array) != len(df):
                if len(pred_array) > len(df):
                    pred_array = pred_array[:len(df)]
                else:
                    pred_array = np.pad(pred_array, (0, len(df) - len(pred_array)), mode='edge')

            display_name = model_display_names.get(model_name, model_name.replace('_', ' ').title())
            fig.add_trace(go.Scatter(
                x=df.index,
                y=pred_array,
                mode='lines',
                name=display_name,
                line=dict(color=model_colors.get(model_name, 'gray'), width=2, dash='dash'),
                opacity=0.7,
                hovertemplate=f'<b>{display_name}</b><br>Date: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>'
            ))
    
    # Plot peer/reference model probabilities (if provided)
    if peer_models:
        peer_colors = ['#8c564b', '#7f7f7f', '#bcbd22']  # Brown, gray, olive
        for i, (peer_name, peer_proba) in enumerate(peer_models.items()):
            peer_array = np.array(peer_proba)
            # Adjust length to match df
            if len(peer_array) != len(df):
                if len(peer_array) > len(df):
                    peer_array = peer_array[:len(df)]
                else:
                    peer_array = np.pad(peer_array, (0, len(df) - len(peer_array)), mode='edge')
            fig.add_trace(go.Scatter(
                x=df.index,
                y=peer_array,
                mode='lines',
                name=f'📌 {peer_name}',
                line=dict(color=peer_colors[i % len(peer_colors)], width=2, dash='dot'),
                opacity=0.8,
                hovertemplate=f'<b>{peer_name}</b><br>Date: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>'
            ))

    # Shade recession periods as contiguous bands
    if 'RECESSION' in df.columns:
        recession_mask = df['RECESSION'].fillna(0).astype(float) == 1
        # Find contiguous recession blocks
        if recession_mask.any():
            # Detect transitions: start when mask goes True, end when it goes False
            shifted = recession_mask.shift(1, fill_value=False)
            starts = df.index[recession_mask & ~shifted]
            ends = df.index[~recession_mask & shifted]

            # Handle case where recession extends to the end of the series
            if len(starts) > len(ends):
                ends = ends.append(pd.DatetimeIndex([df.index[-1]]))

            for start, end in zip(starts, ends):
                # Extend end by ~1 month to fill the gap visually
                fig.add_vrect(
                    x0=start,
                    x1=end + pd.DateOffset(months=1),
                    fillcolor="gray",
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    annotation_text="",
                )

    # Add threshold line
    thresh_val = threshold if threshold is not None else 0.5
    fig.add_hline(
        y=thresh_val,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold ({thresh_val:.0%})",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Recession Probability Forecast',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Recession Probability',
        yaxis=dict(range=[-0.05, 1.05], tickformat='.0%'),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500,
        template='plotly_white'
    )
    
    return fig


def _plot_recession_probability_matplotlib(df, predictions, start_date, end_date):
    """Matplotlib fallback for recession probability plot"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Filter by date range if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    # Plot ensemble
    if 'ensemble' in predictions:
        ax.plot(df.index, predictions['ensemble'], label='Ensemble', linewidth=2, color='darkred')
    
    # Plot base models
    for model_name in ['probit', 'random_forest', 'xgboost', 'markov_switching']:
        if model_name in predictions:
            ax.plot(df.index, predictions[model_name],
                   label=model_name.replace('_', ' ').title(),
                   linewidth=1.5, alpha=0.7, linestyle='--')
    
    # Shade recession periods
    recession_periods = df[df['RECESSION'] == 1]
    for idx in recession_periods.index:
        ax.axvspan(idx, idx, alpha=0.2, color='gray')
    
    # Add threshold line
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Recession Probability')
    ax.set_title('Recession Probability Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    return fig


def plot_model_performance(metrics_df: pd.DataFrame):
    """
    Create bar chart comparing model performance metrics
    
    Args:
        metrics_df: DataFrame with Model, AUC, Precision, Recall, F1, Accuracy columns
        
    Returns:
        Plotly figure object (or matplotlib figure if plotly not available)
    """
    if not HAS_PLOTLY:
        return _plot_model_performance_matplotlib(metrics_df)
    fig = go.Figure()
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('AUC Score', 'Precision', 'Recall', 'F1 Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['AUC', 'Precision', 'Recall', 'F1']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    colors = ['darkred', 'blue', 'green', 'orange', 'purple']
    
    for metric, (row, col) in zip(metrics, positions):
        fig.add_trace(
            go.Bar(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                name=metric,
                marker_color=colors[:len(metrics_df)],
                text=[f'{v:.3f}' for v in metrics_df[metric]],
                textposition='outside',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add benchmark line for AUC
        if metric == 'AUC':
            fig.add_hline(
                y=0.9,
                line_dash="dash",
                line_color="green",
                annotation_text="Benchmark (0.9)",
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Model Performance Metrics",
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def _plot_model_performance_matplotlib(metrics_df):
    """Matplotlib fallback for model performance plot"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['AUC', 'Precision', 'Recall', 'F1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(metrics_df['Model'], metrics_df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        if metric == 'AUC':
            ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_indicator_timeseries(
    df: pd.DataFrame,
    indicator_name: str,
    show_features: bool = False
) -> go.Figure:
    """
    Plot a single indicator over time with optional engineered features
    
    Args:
        df: DataFrame with indicator data
        indicator_name: Name of indicator column to plot
        show_features: Whether to show engineered features (MoM, MA, etc.)
        
    Returns:
        Plotly figure object (or matplotlib figure if plotly not available)
    """
    if not HAS_PLOTLY:
        return _plot_indicator_timeseries_matplotlib(df, indicator_name, show_features)
    fig = go.Figure()
    
    # Plot main indicator
    if indicator_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[indicator_name],
            mode='lines',
            name=indicator_name,
            line=dict(color='blue', width=2)
        ))
    
    # Add engineered features if requested
    if show_features:
        feature_suffixes = ['_MoM', '_3M', '_6M', '_YoY', '_MA3', '_MA6']
        feature_colors = ['red', 'orange', 'yellow', 'green', 'purple', 'brown']
        
        for suffix, color in zip(feature_suffixes, feature_colors):
            feature_col = f"{indicator_name}{suffix}"
            if feature_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[feature_col],
                    mode='lines',
                    name=feature_col,
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.6
                ))
    
    fig.update_layout(
        title=f"Indicator: {indicator_name}",
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

