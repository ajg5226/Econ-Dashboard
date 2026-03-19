"""
Main Streamlit application entry point
Handles authentication and routing to pages
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env (local dev)
load_dotenv(Path(__file__).parent.parent / '.env')

# Streamlit Cloud: secrets.toml overrides .env
# This allows deployment without a .env file on Streamlit Community Cloud
try:
    if "FRED_API_KEY" in st.secrets:
        os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]
    if "SECRET_KEY" in st.secrets:
        os.environ["SECRET_KEY"] = st.secrets["SECRET_KEY"]
except FileNotFoundError:
    pass  # No secrets.toml — using .env instead (local dev)

try:
    from app.auth import render_login, logout, get_user_role
except ImportError:
    # Fallback for direct execution
    from auth import render_login, logout, get_user_role

# Page configuration
st.set_page_config(
    page_title="Recession Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Render login widget (only main.py does this — pages just check session state)
    authenticated, username, name = render_login()
    
    if not authenticated:
        st.stop()
    
    # User is authenticated - show main app
    st.sidebar.title("📊 Recession Prediction")
    st.sidebar.markdown(f"**Welcome, {name}!**")
    
    # Logout button
    logout()
    
    # Display user role
    role = get_user_role(username)
    if role == 'admin':
        st.sidebar.success("🔑 Admin Access")
    else:
        st.sidebar.info("👤 Viewer Access")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    # Main header
    st.markdown('<p class="main-header">📊 Recession Prediction Dashboard</p>', unsafe_allow_html=True)

    # Landing page content
    st.markdown("""
    Welcome to the **Recession Prediction Dashboard** — a quantitative forecasting system
    that estimates the probability of a U.S. recession within the next 6 months.

    ### How It Works
    The engine combines **26 economic indicators** from the Federal Reserve (FRED) with
    literature-backed feature engineering to produce calibrated recession probabilities:

    - **Yield curve dynamics** — term spread inversion depth, duration, and momentum (Estrella & Mishkin 1998, Engstrom & Sharpe 2019)
    - **Monetary policy stance** — federal funds rate interactions (Wright 2006)
    - **Credit & financial stress** — Baa spreads, TED spread, Chicago Fed NFCI (Gilchrist & Zakrajsek 2012)
    - **Labor market** — Sahm Rule unemployment trigger (Sahm 2019)
    - **At-risk transformation** — percentile-based weakness flags across all indicators (Billakanti & Shin 2025, Philadelphia Fed)

    Three base models — **L1-Logistic (Probit)**, **Random Forest**, and **XGBoost** — are
    calibrated with isotonic regression and combined using performance-weighted (inverse Brier score)
    ensembling with bootstrap confidence intervals.

    ### Pages
    - **Dashboard** — Current recession probability, confidence intervals, and peer model comparison
    - **Indicators** — Explore all 28+ economic indicators and derived features
    - **Model Performance** — Evaluation metrics, backtesting, and model monitoring
    - **Settings** — Data refresh, cache management, and user administration

    ### Data & Scheduling
    Data is refreshed weekly via automated pipeline. The model retrains on each refresh
    using an expanding training window, ensuring the latest economic conditions are incorporated.

    *Use the sidebar to navigate between pages.*
    """)

    # Pages are automatically discovered from app/pages/ directory


if __name__ == "__main__":
    main()

