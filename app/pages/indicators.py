"""
Indicators Exploration Page
Allows users to explore economic indicators and their engineered features
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.utils.data_loader import load_indicators
    from app.utils.cache_manager import load_indicators_cached
    from app.utils.plotting import plot_indicator_timeseries
    from app.auth import check_authentication
except ImportError:
    from utils.data_loader import load_indicators
    from utils.cache_manager import load_indicators_cached
    from utils.plotting import plot_indicator_timeseries
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

st.title("Economic Indicators")
st.markdown("### Explore Leading, Coincident, Lagging, and Monetary Indicators")

# Load indicator data
indicators_df = load_indicators_cached()

if indicators_df.empty:
    st.warning("No indicator data available. Please run the data refresh in Settings.")
    st.stop()

# Indicator descriptions — includes new monetary/financial indicators
INDICATOR_DESCRIPTIONS = {
    # Leading indicators
    'leading_USSLIND': {
        'name': 'Conference Board Leading Index',
        'description': 'Composite index of 10 leading indicators that predict economic activity 6-12 months ahead.',
        'category': 'Leading'
    },
    'leading_T10Y2Y': {
        'name': '10Y-2Y Treasury Spread',
        'description': 'Yield curve spread. Inversion (negative values) historically precedes recessions by 6-18 months. (Estrella & Mishkin 1998)',
        'category': 'Leading'
    },
    'leading_T10Y3M': {
        'name': '10Y-3M Treasury Spread',
        'description': 'Primary yield curve recession predictor. Outperforms 10Y-2Y in formal testing. Every US recession since 1960s preceded by inversion. (Estrella & Mishkin 1998)',
        'category': 'Leading'
    },
    'leading_PERMIT': {
        'name': 'Building Permits',
        'description': 'Number of new building permits issued. Leading indicator of construction activity and housing investment.',
        'category': 'Leading'
    },
    'leading_HOUST': {
        'name': 'Housing Starts',
        'description': 'Number of new housing units started. Interest-rate sensitive, predicts economic activity.',
        'category': 'Leading'
    },
    'leading_ICSA': {
        'name': 'Initial Unemployment Claims',
        'description': 'Weekly initial claims for unemployment insurance. Rising claims signal labor market deterioration. Fast-moving, minimal revision.',
        'category': 'Leading'
    },
    'leading_UMCSENT': {
        'name': 'Consumer Sentiment',
        'description': 'University of Michigan consumer sentiment. Low sentiment predicts reduced consumer spending.',
        'category': 'Leading'
    },
    'leading_NEWORDER': {
        'name': 'New Orders, Consumer Goods',
        'description': 'Manufacturing new orders for consumer goods. Predicts production activity.',
        'category': 'Leading'
    },
    'leading_DGORDER': {
        'name': 'New Orders, Durable Goods',
        'description': 'New orders for durable goods. Capital investment indicator.',
        'category': 'Leading'
    },
    # Coincident indicators
    'coincident_PAYEMS': {
        'name': 'Nonfarm Payrolls',
        'description': 'Total nonfarm employment. Real-time measure of labor market health. One of 4 indicators in Chauvet-Hamilton model.',
        'category': 'Coincident'
    },
    'coincident_UNRATE': {
        'name': 'Unemployment Rate',
        'description': 'Percentage of labor force unemployed. Key coincident indicator. Basis for Sahm Rule recession trigger.',
        'category': 'Coincident'
    },
    'coincident_INDPRO': {
        'name': 'Industrial Production',
        'description': 'Index of manufacturing, mining, and utility output. Real-time economic activity. One of 4 Chauvet-Hamilton indicators.',
        'category': 'Coincident'
    },
    'coincident_PI': {
        'name': 'Personal Income',
        'description': 'Total personal income. Measures consumer purchasing power. One of 4 Chauvet-Hamilton indicators.',
        'category': 'Coincident'
    },
    'coincident_RSXFS': {
        'name': 'Retail Sales',
        'description': 'Total retail sales. Real-time consumer spending measure.',
        'category': 'Coincident'
    },
    'coincident_CMRMTSPL': {
        'name': 'Real Manufacturing Sales',
        'description': 'Real manufacturing and trade sales. Business activity indicator. One of 4 Chauvet-Hamilton indicators.',
        'category': 'Coincident'
    },
    # Lagging indicators
    'lagging_UEMPMEAN': {
        'name': 'Avg Unemployment Duration',
        'description': 'Average weeks unemployed. Confirms recession after it starts.',
        'category': 'Lagging'
    },
    'lagging_CPIAUCSL': {
        'name': 'Consumer Price Index',
        'description': 'Inflation measure. Confirms economic conditions.',
        'category': 'Lagging'
    },
    'lagging_ISRATIO': {
        'name': 'Inventory-to-Sales Ratio',
        'description': 'Ratio of inventories to sales. High ratios indicate economic slowdown.',
        'category': 'Lagging'
    },
    # Monetary/Financial indicators (new — Wright 2006, Gilchrist-Zakrajsek 2012)
    'monetary_DFF': {
        'name': 'Federal Funds Rate',
        'description': 'Effective federal funds rate. Wright (2006) showed adding FFR level to yield curve probit significantly improves recession prediction.',
        'category': 'Monetary'
    },
    'monetary_BAA10Y': {
        'name': 'Baa Corporate - 10Y Treasury Spread',
        'description': 'Credit spread proxy for the Excess Bond Premium (Gilchrist & Zakrajsek 2012). A 50bps rise increases 12-month recession probability by ~15pp.',
        'category': 'Monetary'
    },
    'monetary_TEDRATE': {
        'name': 'TED Spread (3M LIBOR - 3M T-Bill)',
        'description': 'Interbank credit stress indicator. Spikes during financial crises signal banking system strain.',
        'category': 'Monetary'
    },
    # Financial conditions indices (Chicago Fed)
    'financial_NFCI': {
        'name': 'Chicago Fed NFCI',
        'description': 'National Financial Conditions Index. Widely cited composite of 105 financial indicators. Positive = tighter-than-average conditions.',
        'category': 'Financial'
    },
    'financial_ANFCI': {
        'name': 'Chicago Fed Adjusted NFCI',
        'description': 'NFCI adjusted for prevailing economic conditions. Isolates financial conditions from the business cycle.',
        'category': 'Financial'
    },
}

# Derived/computed indicators (shown separately)
DERIVED_DESCRIPTIONS = {
    'SAHM_INDICATOR': {
        'name': 'Sahm Rule Indicator',
        'description': 'Rise in 3-month average unemployment rate above its 12-month low. Trigger at 0.50pp. Only 2 false positives since 1959. (Sahm 2019)',
        'category': 'Derived'
    },
    'SAHM_TRIGGER': {
        'name': 'Sahm Rule Trigger',
        'description': 'Binary flag: 1 when Sahm indicator >= 0.50pp. Designed as real-time coincident recession signal.',
        'category': 'Derived'
    },
    'CREDIT_STRESS_INDEX': {
        'name': 'Credit Stress Index',
        'description': 'Standardized composite of Baa-Treasury spread and TED spread. Higher values indicate elevated financial stress.',
        'category': 'Derived'
    },
    'AT_RISK_DIFFUSION': {
        'name': 'At-Risk Diffusion Index',
        'description': 'Fraction of indicators in "unusually weak" state (below expanding 10th percentile). (Billakanti & Shin 2025, Philly Fed)',
        'category': 'Derived'
    },
    'FFR_STANCE': {
        'name': 'Monetary Policy Stance',
        'description': 'Federal funds rate relative to its 3-year moving average. Positive = tighter than recent norm.',
        'category': 'Derived'
    },
    'NFCI_Z': {
        'name': 'NFCI Z-Score',
        'description': 'Standardized NFCI using expanding window. Measures how unusual current financial conditions are relative to history.',
        'category': 'Derived'
    },
    'FINANCIAL_STRESS_COMPOSITE': {
        'name': 'Financial Stress Composite',
        'description': 'Equal-weighted composite of NFCI z-score and Baa credit spread z-score. Broader measure of financial system stress.',
        'category': 'Derived'
    },
}

# Peer/reference models (for benchmarking)
REFERENCE_DESCRIPTIONS = {
    'ref_RECPROUSM156N': {
        'name': 'NY Fed Recession Probability (Chauvet-Piger)',
        'description': 'Smoothed recession probabilities from the Chauvet-Piger Markov-switching model. Published by the NY Fed. Widely used benchmark for recession nowcasting.',
        'category': 'Reference'
    },
    'ref_JHGDPBRINDX': {
        'name': 'Hamilton GDP-Based Recession Index',
        'description': 'GDP-based recession indicator index. Uses a nonlinear filter applied to real GDP growth. Values above 67% indicate recession. (Hamilton 2005)',
        'category': 'Reference'
    },
}

# Combine all descriptions
ALL_DESCRIPTIONS = {**INDICATOR_DESCRIPTIONS, **DERIVED_DESCRIPTIONS, **REFERENCE_DESCRIPTIONS}

# Sidebar filters
st.sidebar.markdown("### Filters")

# Category filter
categories = ['All', 'Leading', 'Coincident', 'Lagging', 'Monetary', 'Financial', 'Derived', 'Reference']
selected_category = st.sidebar.selectbox("Indicator Category", categories)

# Filter indicators by category
available_indicators = [col for col in indicators_df.columns
                        if col in ALL_DESCRIPTIONS]

if selected_category != 'All':
    available_indicators = [
        ind for ind in available_indicators
        if ALL_DESCRIPTIONS.get(ind, {}).get('category') == selected_category
    ]

if not available_indicators:
    st.warning("No indicators available in selected category.")
    st.stop()

# Show engineered features toggle
show_features = st.sidebar.checkbox("Show Engineered Features", value=False)

# ─── Overview Table ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Indicator Summary")

# Build summary table for all visible indicators
summary_rows = []
for col in available_indicators:
    info = ALL_DESCRIPTIONS[col]
    series = indicators_df[col].dropna()
    if len(series) > 0:
        current_val = series.iloc[-1]
        current_date = series.index[-1]
        yoy_change = (current_val - series.iloc[-12]) if len(series) >= 12 else np.nan
        # Percentage change for YoY
        if len(series) >= 12 and series.iloc[-12] != 0:
            yoy_pct = ((current_val - series.iloc[-12]) / abs(series.iloc[-12])) * 100
        else:
            yoy_pct = np.nan
    else:
        current_val = np.nan
        current_date = None
        yoy_change = np.nan
        yoy_pct = np.nan

    summary_rows.append({
        'Indicator': info['name'],
        'Category': info['category'],
        'Current Value': current_val,
        'As Of': current_date.strftime('%Y-%m') if current_date is not None else 'N/A',
        'YoY Change': yoy_change,
        'YoY %': yoy_pct,
        '_key': col,
    })

summary_df = pd.DataFrame(summary_rows)

# Format and display
display_df = summary_df[['Indicator', 'Category', 'Current Value', 'As Of', 'YoY Change', 'YoY %']].copy()
display_df['Current Value'] = display_df['Current Value'].apply(
    lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
)
display_df['YoY Change'] = display_df['YoY Change'].apply(
    lambda x: f"{x:+,.2f}" if pd.notna(x) else "N/A"
)
display_df['YoY %'] = display_df['YoY %'].apply(
    lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ─── Mini Sparkline Grid ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Trend Overview (Last 5 Years)")

if HAS_PLOTLY:
    # Show mini charts in a grid: 3 columns
    cols_per_row = 3
    for i in range(0, len(available_indicators), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col_widget in enumerate(cols):
            idx = i + j
            if idx >= len(available_indicators):
                break
            ind_key = available_indicators[idx]
            info = ALL_DESCRIPTIONS[ind_key]
            series = indicators_df[ind_key].dropna()

            # Last 5 years
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
            series_recent = series[series.index >= cutoff]

            with col_widget:
                if len(series_recent) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=series_recent.index,
                        y=series_recent.values,
                        mode='lines',
                        line=dict(color='steelblue', width=2),
                        hovertemplate='%{x|%Y-%m}: %{y:,.2f}<extra></extra>'
                    ))
                    fig.update_layout(
                        title=dict(text=info['name'], font=dict(size=12)),
                        height=180,
                        margin=dict(l=5, r=5, t=30, b=5),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=True, showgrid=True, tickfont=dict(size=9)),
                        template='plotly_white',
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"spark_{ind_key}")
                else:
                    st.caption(info['name'])
                    st.info("Insufficient data")

# ─── Detailed Single-Indicator View ──────────────────────────────────────────
st.markdown("---")
st.markdown("### Indicator Detail")

selected_indicator = st.selectbox(
    "Select an indicator for detailed view",
    options=available_indicators,
    format_func=lambda x: ALL_DESCRIPTIONS.get(x, {}).get('name', x)
)

if selected_indicator in ALL_DESCRIPTIONS:
    info = ALL_DESCRIPTIONS[selected_indicator]
    st.markdown(f"**{info['name']}** ({info['category']})")
    st.info(info['description'])

    # Warn if indicator data is stale (last valid > 6 months old)
    last_valid = indicators_df[selected_indicator].last_valid_index()
    if last_valid is not None:
        from datetime import datetime
        staleness_days = (pd.Timestamp.now() - last_valid).days
        if staleness_days > 180:
            st.warning(f"This indicator has not been updated since {last_valid.strftime('%Y-%m')}. "
                       f"It may be discontinued or delayed.")

# Full time series plot
fig = plot_indicator_timeseries(
    indicators_df,
    selected_indicator,
    show_features=show_features
)
st.plotly_chart(fig, use_container_width=True)

# Statistics row
col1, col2, col3, col4 = st.columns(4)

series = indicators_df[selected_indicator].dropna()

with col1:
    if len(series) > 0:
        current_val = series.iloc[-1]
        current_date = series.index[-1]
        st.metric("Current Value", f"{current_val:,.2f}",
                   help=f"As of {current_date.strftime('%Y-%m')}")
    else:
        st.metric("Current Value", "N/A")

with col2:
    if len(series) >= 12:
        change = series.iloc[-1] - series.iloc[-12]
        st.metric("YoY Change", f"{change:+,.2f}")
    else:
        st.metric("YoY Change", "N/A")

with col3:
    mean_val = series.mean()
    st.metric("Mean", f"{mean_val:,.2f}" if pd.notna(mean_val) else "N/A")

with col4:
    std_val = series.std()
    st.metric("Std Dev", f"{std_val:,.2f}" if pd.notna(std_val) else "N/A")

# Recent data table
st.markdown("#### Recent Data")
valid_data = indicators_df[[selected_indicator]].dropna()
recent_data = valid_data.tail(12)
if recent_data.empty:
    st.info("No recent data available")
else:
    st.dataframe(recent_data, use_container_width=True)
