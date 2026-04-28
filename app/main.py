"""
Main Streamlit application entry point
Handles authentication and routing to pages
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
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
    from app.utils.cache_manager import load_indicators_cached, load_predictions_cached
    from app.utils.data_loader import get_last_update_time, is_data_stale
except ImportError:
    # Fallback for direct execution
    from auth import render_login, logout, get_user_role
    from utils.cache_manager import load_indicators_cached, load_predictions_cached
    from utils.data_loader import get_last_update_time, is_data_stale

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


GLR_STATE_EMOJI = {'strong': '🟢', 'neutral': '🟡', 'weak': '🔴'}
REPO_ROOT = Path(__file__).parent.parent


def _risk_label(prob: float) -> str:
    """Match the breakpoints used on the Dashboard page (`app/pages/dashboard.py:202-213`)."""
    if prob < 0.15:
        return "🟢 LOW"
    if prob < 0.35:
        return "🟡 MODERATE"
    if prob < 0.60:
        return "🟠 ELEVATED"
    return "🔴 HIGH"


def _read_threshold(path: Path, default: float) -> float:
    try:
        if path.exists():
            with open(path) as f:
                return float(json.load(f).get('decision_threshold', default))
    except Exception:
        pass
    return default


def _load_18m_predictions() -> pd.DataFrame:
    p = REPO_ROOT / "data" / "models" / "horizon_18m" / "predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.dropna(subset=['Date']).sort_values('Date')
    except Exception:
        return pd.DataFrame()


def _render_status_panel():
    """Live snapshot: 6M / 18M recession probabilities and GLR composites."""
    try:
        if is_data_stale(days_threshold=7):
            st.warning("⚠️ Data is more than 7 days old — see Settings to refresh.")

        # ----- Recession row -----
        st.markdown("#### Recession Probability")
        rec_cols = st.columns(3)

        preds = load_predictions_cached()
        latest_data_date = None
        rendered_6m = False
        if not preds.empty and 'Prob_Ensemble' in preds.columns:
            preds = preds.copy()
            preds['Date'] = pd.to_datetime(preds['Date'], errors='coerce')
            preds = preds.dropna(subset=['Date', 'Prob_Ensemble']).sort_values('Date')
            if not preds.empty:
                latest_6m = preds.iloc[-1]
                latest_data_date = latest_6m['Date']
                threshold_6m = _read_threshold(REPO_ROOT / "data" / "models" / "threshold.json", 0.32)
                prob_6m = float(latest_6m['Prob_Ensemble'])
                with rec_cols[0]:
                    st.metric("Latest data point", latest_data_date.strftime("%Y-%m-%d"))
                with rec_cols[1]:
                    st.metric(
                        "6-Month Probability",
                        f"{prob_6m:.1%}",
                        delta=f"{(prob_6m - threshold_6m):+.1%} vs {threshold_6m:.0%} threshold",
                        delta_color="inverse",
                    )
                    st.caption(_risk_label(prob_6m))
                rendered_6m = True

        if not rendered_6m:
            with rec_cols[0]:
                st.info("Recession predictions not available — run a refresh in Settings.")

        preds_18m = _load_18m_predictions()
        if not preds_18m.empty and 'Prob_Ensemble' in preds_18m.columns:
            valid = preds_18m.dropna(subset=['Prob_Ensemble'])
            if not valid.empty:
                latest_18m = valid.iloc[-1]
                threshold_18m = _read_threshold(
                    REPO_ROOT / "data" / "models" / "horizon_18m" / "threshold.json",
                    0.21,
                )
                prob_18m = float(latest_18m['Prob_Ensemble'])
                with rec_cols[2]:
                    st.metric(
                        "18-Month Probability",
                        f"{prob_18m:.1%}",
                        delta=f"{(prob_18m - threshold_18m):+.1%} vs {threshold_18m:.0%} threshold",
                        delta_color="inverse",
                    )
                    st.caption(_risk_label(prob_18m))

        # ----- GLR row -----
        indicators = load_indicators_cached()
        glr_defs = [
            ('GLR_GROWTH', 'GLR_GROWTH_STATE', 'Growth'),
            ('GLR_LIQUIDITY', 'GLR_LIQUIDITY_STATE', 'Liquidity'),
            ('GLR_RISK_APPETITE', 'GLR_RISK_APPETITE_STATE', 'Risk Appetite'),
        ]
        glr_available = (
            not indicators.empty
            and all(val_col in indicators.columns for val_col, _, _ in glr_defs)
        )
        if glr_available:
            st.markdown("#### GLR Composite Scores")
            glr_cols = st.columns(3)
            for col_idx, (val_col, state_col, label) in enumerate(glr_defs):
                val_series = indicators[val_col].dropna()
                state_series = (
                    indicators[state_col].dropna()
                    if state_col in indicators.columns else pd.Series(dtype=object)
                )
                with glr_cols[col_idx]:
                    if val_series.empty:
                        st.metric(label, "N/A")
                        continue
                    latest_val = float(val_series.iloc[-1])
                    latest_state = str(state_series.iloc[-1]) if not state_series.empty else ""
                    emoji = GLR_STATE_EMOJI.get(latest_state, '')
                    as_of = val_series.index[-1]
                    as_of_str = (
                        as_of.strftime("%Y-%m-%d") if hasattr(as_of, 'strftime') else str(as_of)
                    )
                    st.metric(label, f"{latest_val:+.2f}σ")
                    caption = f"{emoji} {latest_state.title()}".strip() if latest_state else ""
                    st.caption((caption + f" · as of {as_of_str}") if caption else f"as of {as_of_str}")

        # Last refresh footer
        last_update = get_last_update_time()
        if last_update:
            st.caption(f"📅 Data refreshed: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as exc:
        st.error(f"Status panel unavailable: {exc}")


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

    # Live status snapshot
    _render_status_panel()
    st.markdown("---")

    # Landing page content
    st.markdown("""
    Welcome to the **Recession Prediction Dashboard** — a quantitative forecasting system
    that estimates the probability of a U.S. recession at both **6-month** (primary) and
    **18-month** (early-warning) horizons.

    ### How It Works
    The engine pulls **~45 economic indicators across 7 categories** from the Federal Reserve (FRED) —
    yield curve, monetary, credit/financial, labor, housing, term-structure, and a separate set
    of Growth / Liquidity / Risk Appetite (GLR) regime series — and applies literature-backed
    feature engineering (500+ engineered columns) to produce calibrated recession probabilities:

    - **Yield curve dynamics** — term spread inversion depth, duration, and momentum (Estrella & Mishkin 1998, Engstrom & Sharpe 2019)
    - **Monetary policy stance** — federal funds rate interactions (Wright 2006)
    - **Credit & financial stress** — Baa spreads, TED spread, Chicago Fed NFCI, SLOOS tightening (Gilchrist & Zakrajsek 2012)
    - **Labor market** — Sahm Rule unemployment trigger (Sahm 2019), JOLTS V/U gap, SOS insured-unemployment signal (Scavette & O'Trakoun 2025)
    - **Housing & term premium** — Case-Shiller confirming indicator (Grigoli & Sandri 2024), Kim-Wright term-premium-adjusted spread (Ajello et al. 2022)
    - **At-risk transformation** — percentile-based weakness flags across all indicators (Billakanti & Shin 2025)
    - **GLR regime monitor** — separate Growth / Liquidity / Risk Appetite z-score composites for macro-regime context

    Four base models — **L1-Logistic (Probit)**, **Random Forest**, **XGBoost**, and a
    **3-state TVTP Markov-Switching** filter — are calibrated with isotonic regression and
    combined using performance-weighted ensembling with bootstrap confidence intervals.
    Active models per horizon are gated by cross-validated performance thresholds.

    ### Pages
    - **Dashboard** — 6-month recession probability, confidence intervals, peer-model comparison
    - **Indicators** — Browse all FRED inputs and derived features
    - **Model Performance** — Backtesting, rolling metrics, drift / PSI monitoring, calibration diagnostics
    - **GLR Monitor** — Growth / Liquidity / Risk Appetite composites with state classification
    - **Settings** — Data refresh, cache management, runtime configuration, user administration

    ### Data & Scheduling
    Data refreshes weekly via a GitHub Actions pipeline that re-fetches FRED, retrains on an
    expanding window, and re-writes the prediction artifacts. The pipeline includes **drift
    monitoring** (per-tier PSI alerts, dead-series detection, capped rolling-NaN propagation)
    and an **18-month secondary horizon** for early-stage monitoring of slower-moving signals.

    *Use the sidebar to navigate between pages.*
    """)

    # Pages are automatically discovered from app/pages/ directory


if __name__ == "__main__":
    main()
