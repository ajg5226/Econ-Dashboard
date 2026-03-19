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

# Check authentication
authenticated, username, name = check_authentication()
if not authenticated:
    st.stop()

st.title("📈 Economic Indicators")
st.markdown("### Explore Leading, Coincident, and Lagging Indicators")

# Load indicator data
indicators_df = load_indicators_cached()

if indicators_df.empty:
    st.warning("⚠️ No indicator data available. Please run the data refresh in Settings.")
    st.stop()

# Indicator descriptions
INDICATOR_DESCRIPTIONS = {
    # Leading indicators
    'leading_USSLIND': {
        'name': 'Conference Board Leading Index',
        'description': 'Composite index of 10 leading indicators that predict economic activity 6-12 months ahead.',
        'category': 'Leading'
    },
    'leading_T10Y2Y': {
        'name': '10-Year minus 2-Year Treasury Spread',
        'description': 'Yield curve spread. Inversion (negative values) historically precedes recessions.',
        'category': 'Leading'
    },
    'leading_T10Y3M': {
        'name': '10-Year minus 3-Month Treasury Spread',
        'description': 'Short-term yield curve spread. Another recession predictor.',
        'category': 'Leading'
    },
    'leading_PERMIT': {
        'name': 'Building Permits',
        'description': 'Number of new building permits issued. Leading indicator of construction activity.',
        'category': 'Leading'
    },
    'leading_HOUST': {
        'name': 'Housing Starts',
        'description': 'Number of new housing units started. Predicts economic activity.',
        'category': 'Leading'
    },
    'leading_ICSA': {
        'name': 'Initial Unemployment Claims',
        'description': 'Weekly initial claims for unemployment insurance. Rising claims signal economic weakness.',
        'category': 'Leading'
    },
    'leading_UMCSENT': {
        'name': 'Consumer Sentiment Index',
        'description': 'University of Michigan consumer sentiment. Low sentiment predicts reduced spending.',
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
        'description': 'Total nonfarm employment. Real-time measure of labor market health.',
        'category': 'Coincident'
    },
    'coincident_UNRATE': {
        'name': 'Unemployment Rate',
        'description': 'Percentage of labor force unemployed. Key coincident indicator.',
        'category': 'Coincident'
    },
    'coincident_INDPRO': {
        'name': 'Industrial Production Index',
        'description': 'Index of manufacturing, mining, and utility output. Real-time economic activity.',
        'category': 'Coincident'
    },
    'coincident_PI': {
        'name': 'Personal Income',
        'description': 'Total personal income. Measures consumer purchasing power.',
        'category': 'Coincident'
    },
    'coincident_RSXFS': {
        'name': 'Retail Sales',
        'description': 'Total retail sales. Real-time consumer spending measure.',
        'category': 'Coincident'
    },
    'coincident_CMRMTSPL': {
        'name': 'Real Manufacturing Sales',
        'description': 'Real manufacturing and trade sales. Business activity indicator.',
        'category': 'Coincident'
    },
    # Lagging indicators
    'lagging_UEMPMEAN': {
        'name': 'Average Unemployment Duration',
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
    }
}

# Sidebar filters
st.sidebar.markdown("### Filters")

# Category filter
categories = ['All', 'Leading', 'Coincident', 'Lagging']
selected_category = st.sidebar.selectbox("Indicator Category", categories)

# Indicator selector — show only raw indicators (not engineered features)
available_indicators = [col for col in indicators_df.columns
                        if col in INDICATOR_DESCRIPTIONS]

if selected_category != 'All':
    available_indicators = [
        ind for ind in available_indicators
        if INDICATOR_DESCRIPTIONS.get(ind, {}).get('category') == selected_category
    ]

if not available_indicators:
    st.warning("No indicators available in selected category.")
    st.stop()

selected_indicator = st.sidebar.selectbox(
    "Select Indicator",
    options=available_indicators,
    format_func=lambda x: INDICATOR_DESCRIPTIONS.get(x, {}).get('name', x)
)

# Show engineered features toggle
show_features = st.sidebar.checkbox("Show Engineered Features", value=False)

# Main content
if selected_indicator in INDICATOR_DESCRIPTIONS:
    info = INDICATOR_DESCRIPTIONS[selected_indicator]
    st.markdown(f"### {info['name']}")
    st.markdown(f"**Category:** {info['category']}")
    st.info(info['description'])

    # Warn if indicator data is stale (last valid > 6 months old)
    last_valid = indicators_df[selected_indicator].last_valid_index()
    if last_valid is not None:
        from datetime import datetime
        staleness_days = (pd.Timestamp.now() - last_valid).days
        if staleness_days > 180:
            st.warning(f"⚠️ This indicator has not been updated since {last_valid.strftime('%Y-%m')}. "
                       f"It may be discontinued or delayed.")

# Plot indicator
st.markdown("---")
st.markdown("### Time Series")

fig = plot_indicator_timeseries(
    indicators_df,
    selected_indicator,
    show_features=show_features
)

st.plotly_chart(fig, use_container_width=True)

# Statistics
st.markdown("---")
st.markdown("### Statistics")

col1, col2, col3, col4 = st.columns(4)

# BUG FIX 14: Handle empty dataframe and missing values
if indicators_df.empty or selected_indicator not in indicators_df.columns:
    st.error("❌ No data available for selected indicator")
    st.stop()

with col1:
    try:
        # Use last valid (non-null) value instead of last row
        series = indicators_df[selected_indicator].dropna()
        if len(series) > 0:
            current_val = series.iloc[-1]
            current_date = series.index[-1]
            st.metric("Current Value", f"{current_val:,.2f}",
                       help=f"As of {current_date.strftime('%Y-%m')}")
        else:
            st.metric("Current Value", "N/A")
    except (IndexError, KeyError):
        st.metric("Current Value", "N/A")

with col2:
    try:
        series = indicators_df[selected_indicator].dropna()
        if len(series) >= 12:
            current = series.iloc[-1]
            past = series.iloc[-12]
            change = current - past
            st.metric("YoY Change", f"{change:,.2f}")
        else:
            st.metric("YoY Change", "N/A")
    except (IndexError, KeyError):
        st.metric("YoY Change", "N/A")

with col3:
    try:
        mean_val = indicators_df[selected_indicator].mean()
        if pd.isna(mean_val):
            st.metric("Mean", "N/A")
        else:
            st.metric("Mean", f"{mean_val:,.2f}")
    except (KeyError, ValueError):
        st.metric("Mean", "N/A")

with col4:
    try:
        std_val = indicators_df[selected_indicator].std()
        if pd.isna(std_val):
            st.metric("Std Dev", "N/A")
        else:
            st.metric("Std Dev", f"{std_val:,.2f}")
    except (KeyError, ValueError):
        st.metric("Std Dev", "N/A")

# Data table
st.markdown("---")
st.markdown("### Recent Data")

# Show last 12 months
# BUG FIX 15: Handle empty dataframe
if indicators_df.empty or selected_indicator not in indicators_df.columns:
    st.warning("⚠️ No data available to display")
else:
    # Show last 12 non-null values to avoid trailing NaN rows
    valid_data = indicators_df[[selected_indicator]].dropna()
    recent_data = valid_data.tail(12)
    if recent_data.empty:
        st.info("No recent data available")
    else:
        st.dataframe(recent_data, use_container_width=True)

