"""
Main Dashboard Page
Displays recession probability forecasts and key metrics
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.utils.data_loader import load_predictions, is_data_stale, get_last_update_time
    from app.utils.cache_manager import load_predictions_cached
    from app.utils.plotting import plot_recession_probability
    from app.auth import check_authentication, is_admin
except ImportError:
    from utils.data_loader import load_predictions, is_data_stale, get_last_update_time
    from utils.cache_manager import load_predictions_cached
    from utils.plotting import plot_recession_probability
    from auth import check_authentication, is_admin

# Check authentication
authenticated, username, name = check_authentication()
if not authenticated:
    st.stop()

st.title("📊 Dashboard")
st.markdown("### Recession Probability Forecast")

# Load data
predictions_df = load_predictions_cached()

if predictions_df.empty:
    st.warning("⚠️ No prediction data available. Please run the data refresh in Settings.")
    st.info("The scheduler should automatically update data weekly, or you can trigger a manual refresh.")
    st.stop()

# Convert Date column to datetime and set as index
# BUG FIX 1: Handle missing or invalid Date column
if 'Date' not in predictions_df.columns:
    st.error("❌ Date column missing from predictions data")
    st.stop()

try:
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'], errors='coerce')
    # BUG FIX 2: Remove rows with invalid dates
    predictions_df = predictions_df.dropna(subset=['Date'])
    if predictions_df.empty:
        st.error("❌ No valid dates in predictions data")
        st.stop()
    predictions_df = predictions_df.set_index('Date').sort_index()
except Exception as e:
    st.error(f"❌ Error processing dates: {str(e)}")
    st.stop()

# Check data freshness
if is_data_stale(days_threshold=7):
    st.warning("⚠️ Data is more than 7 days old. Consider refreshing.")
    
last_update = get_last_update_time()
if last_update:
    st.info(f"📅 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar filters
st.sidebar.markdown("### Filters")

# Date range selector
# BUG FIX 3: Handle empty index after filtering
if len(predictions_df) == 0:
    st.error("❌ No data available after processing")
    st.stop()

min_date = predictions_df.index.min()
max_date = predictions_df.index.max()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(max_date - timedelta(days=365*2), max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = predictions_df[
        (predictions_df.index >= pd.to_datetime(start_date)) &
        (predictions_df.index <= pd.to_datetime(end_date))
    ]
    # BUG FIX 4: Handle empty filtered dataframe
    if filtered_df.empty:
        st.warning("⚠️ No data in selected date range. Showing all data.")
        filtered_df = predictions_df
        start_date = None
        end_date = None
else:
    filtered_df = predictions_df
    start_date = None
    end_date = None

# Prediction horizon selector
horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    options=[3, 6, 12],
    index=1,  # Default to 6 months
    format_func=lambda x: f"{x} months"
)

# Threshold selector
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Probability threshold for recession prediction"
)

# Main content
col1, col2, col3, col4 = st.columns(4)

# Latest ensemble probability
# BUG FIX 5: Handle empty dataframe and missing column
if filtered_df.empty:
    st.error("❌ No data available for display")
    st.stop()

if 'Prob_Ensemble' not in filtered_df.columns:
    st.warning("⚠️ Ensemble predictions not available")
    latest_prob = 0.0
else:
    try:
        latest_prob = float(filtered_df['Prob_Ensemble'].iloc[-1])
        if pd.isna(latest_prob):
            latest_prob = 0.0
    except (IndexError, KeyError):
        latest_prob = 0.0

with col1:
    st.metric(
        "Current Probability",
        f"{latest_prob:.1%}",
        delta=f"{(latest_prob - threshold):.1%}" if latest_prob > threshold else None
    )

# Risk level
if latest_prob < 0.15:
    risk_level = "🟢 LOW"
    risk_color = "green"
elif latest_prob < 0.35:
    risk_level = "🟡 MODERATE"
    risk_color = "orange"
elif latest_prob < 0.60:
    risk_level = "🟠 ELEVATED"
    risk_color = "red"
else:
    risk_level = "🔴 HIGH"
    risk_color = "red"

with col2:
    st.metric("Risk Level", risk_level)

# Latest date
# BUG FIX 6: Handle empty index
with col3:
    if len(filtered_df) > 0:
        st.metric("Latest Date", filtered_df.index[-1].strftime("%Y-%m-%d"))
    else:
        st.metric("Latest Date", "N/A")

# Data points
with col4:
    st.metric("Data Points", len(filtered_df))

# Prepare predictions dictionary for plotting
predictions_dict = {}
if 'Prob_Ensemble' in filtered_df.columns:
    predictions_dict['ensemble'] = filtered_df['Prob_Ensemble'].values
if 'Prob_Probit' in filtered_df.columns:
    predictions_dict['probit'] = filtered_df['Prob_Probit'].values
if 'Prob_RandomForest' in filtered_df.columns:
    predictions_dict['random_forest'] = filtered_df['Prob_RandomForest'].values
if 'Prob_XGBoost' in filtered_df.columns:
    predictions_dict['xgboost'] = filtered_df['Prob_XGBoost'].values

# Create recession indicator series
# BUG FIX 7: Fix get() returning scalar instead of Series
if 'Actual_Recession' in filtered_df.columns:
    recession_series = filtered_df['Actual_Recession']
else:
    recession_series = pd.Series(0, index=filtered_df.index)

# Plot
st.markdown("---")
st.markdown("### Recession Probability Over Time")

# BUG FIX 8: Handle undefined start_date/end_date
start_date_str = start_date.strftime("%Y-%m-%d") if start_date is not None else None
end_date_str = end_date.strftime("%Y-%m-%d") if end_date is not None else None

fig = plot_recession_probability(
    pd.DataFrame({'RECESSION': recession_series}),
    predictions_dict,
    start_date=start_date_str,
    end_date=end_date_str
)

st.plotly_chart(fig, use_container_width=True)

# Summary table
st.markdown("---")
st.markdown("### Latest Predictions")

# Get latest row
# BUG FIX 9: Handle empty dataframe
if filtered_df.empty:
    st.warning("⚠️ No data to display in summary table")
else:
    latest_row = filtered_df.iloc[-1:].copy()
    latest_row.index = [latest_row.index[0].strftime("%Y-%m-%d")]

# Format probabilities as percentages
display_cols = [col for col in latest_row.columns if col.startswith('Prob_')]
if display_cols:
    for col in display_cols:
        latest_row[col] = latest_row[col].apply(lambda x: f"{float(x):.1%}" if pd.notna(x) else "N/A")
    st.dataframe(latest_row[display_cols], use_container_width=True)
else:
    st.info("No probability columns available")

# Download buttons
st.markdown("---")
st.markdown("### Downloads")

col1, col2, col3 = st.columns(3)

with col1:
    # Download CSV
    csv = filtered_df.to_csv()
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv,
        file_name=f"recession_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Download executive report
    try:
        from app.utils.data_loader import load_executive_report
    except ImportError:
        from utils.data_loader import load_executive_report
    report = load_executive_report()
    if report:
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"executive_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    else:
        st.info("Report not available")

with col3:
    # Download chart as PNG (requires plotly)
    if st.button("📊 Download Chart (PNG)"):
        fig.write_image("recession_probability.png")
        with open("recession_probability.png", "rb") as file:
            st.download_button(
                label="⬇️ Download PNG",
                data=file.read(),
                file_name=f"recession_probability_{datetime.now().strftime('%Y%m%d')}.png",
                mime="image/png"
            )

