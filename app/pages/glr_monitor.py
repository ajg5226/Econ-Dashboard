"""
GLR Monitor Page — Growth / Liquidity / Risk Appetite Composite Signals

Three equal-weighted composites of FRED indicators, standardized to
expanding z-scores (min_periods=12). Higher = risk-on / expansionary /
easy liquidity.
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.utils.cache_manager import (
        load_indicators_cached,
        load_glr_components_cached,
        load_predictions_cached,
    )
    from app.utils.plotting import (
        plot_glr_composites,
        plot_glr_sparkline,
        GLR_COLORS,
    )
    from app.auth import check_authentication
    from recession_engine.glr_engine import GLRRegimeEngine
except ImportError:
    from utils.cache_manager import (
        load_indicators_cached,
        load_glr_components_cached,
        load_predictions_cached,
    )
    from utils.plotting import (
        plot_glr_composites,
        plot_glr_sparkline,
        GLR_COLORS,
    )
    from auth import check_authentication
    from recession_engine.glr_engine import GLRRegimeEngine


STATE_EMOJI = {'strong': '🟢', 'neutral': '🟡', 'weak': '🔴'}

COMPOSITE_META = [
    ('GLR_GROWTH', 'GLR_GROWTH_STATE', 'Growth', 'growth'),
    ('GLR_LIQUIDITY', 'GLR_LIQUIDITY_STATE', 'Liquidity', 'liquidity'),
    ('GLR_RISK_APPETITE', 'GLR_RISK_APPETITE_STATE', 'Risk Appetite', 'risk'),
]

COMPOSITE_KEY_TO_COL = {
    'growth': 'GLR_GROWTH',
    'liquidity': 'GLR_LIQUIDITY',
    'risk': 'GLR_RISK_APPETITE',
}


def _format_zscore(value) -> str:
    if value is None or pd.isna(value):
        return 'N/A'
    return f"{float(value):+.2f}σ"


def _format_delta(value) -> str:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):+.2f}"


def _state_label(state) -> str:
    if state is None or pd.isna(state):
        return 'State: —'
    state = str(state)
    return f"{STATE_EMOJI.get(state, '')} State: {state}"


def _latest_value(series, date):
    """Return the last non-NaN value at or before `date`."""
    if series is None or len(series) == 0:
        return np.nan
    slice_ = series.loc[:date].dropna()
    if slice_.empty:
        return np.nan
    return slice_.iloc[-1]


def _three_month_delta(series, end_date):
    """Current z-score minus z-score 3 months prior (approximately)."""
    end_val = _latest_value(series, end_date)
    prior_cutoff = end_date - pd.DateOffset(months=3)
    prior_val = _latest_value(series, prior_cutoff)
    if pd.isna(end_val) or pd.isna(prior_val):
        return np.nan
    return end_val - prior_val


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or isinstance(df.index, pd.DatetimeIndex):
        return df
    parsed = pd.to_datetime(df.index, errors='coerce')
    mask = pd.notna(parsed)
    if not mask.any():
        return df
    out = df.loc[mask].copy()
    out.index = pd.DatetimeIndex(parsed[mask])
    return out.sort_index()


# ---------------------------------------------------------------- auth

authenticated, username, name = check_authentication()
if not authenticated:
    st.stop()

# ---------------------------------------------------------------- header

st.title("🎯 GLR Monitor")
st.markdown("### Growth / Liquidity / Risk Appetite Composite Signals")
st.caption(
    "Three equal-weighted composites of FRED indicators, standardized to "
    "expanding z-scores (min_periods=12). Sign convention: **higher = "
    "risk-on / expansionary / easy liquidity**."
)

# ---------------------------------------------------------------- data

indicators_df = _normalize_index(load_indicators_cached())
components_df = _normalize_index(load_glr_components_cached())

if indicators_df.empty:
    st.warning("No indicator data available. Run the scheduler or trigger a refresh in Settings.")
    st.stop()

required_cols = {'GLR_GROWTH', 'GLR_LIQUIDITY', 'GLR_RISK_APPETITE'}
if not required_cols.issubset(indicators_df.columns):
    st.warning(
        "GLR composites not found in indicators.csv. "
        "Run the scheduler once to populate them (composites are computed during the weekly refresh)."
    )
    st.stop()

composites_df = indicators_df[list(required_cols)].copy()
state_cols = ['GLR_GROWTH_STATE', 'GLR_LIQUIDITY_STATE', 'GLR_RISK_APPETITE_STATE']
states_df = indicators_df[[c for c in state_cols if c in indicators_df.columns]].copy()

# ---------------------------------------------------------------- sidebar

st.sidebar.markdown("### GLR Filters")

min_allowed = pd.Timestamp('2010-01-01')
data_min = max(composites_df.index.min(), min_allowed)
data_max = composites_df.index.max()
default_start = max(data_max - pd.DateOffset(years=5), data_min)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(default_start.date(), data_max.date()),
    min_value=data_min.date(),
    max_value=data_max.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = default_start, data_max

show_bands = st.sidebar.checkbox("Show state bands (33rd/67th percentile)", value=True)
show_recession_shading = st.sidebar.checkbox("Show recession shading", value=True)

mask = (composites_df.index >= start_date) & (composites_df.index <= end_date)
composites_view = composites_df.loc[mask]
states_view = states_df.loc[mask] if not states_df.empty else pd.DataFrame()

# ---------------------------------------------------------------- row 1: KPIs

st.markdown("---")
kpi_cols = st.columns(4)

latest_date = composites_view.index.max() if not composites_view.empty else composites_df.index.max()

for idx, (comp_col, state_col, display, _key) in enumerate(COMPOSITE_META):
    series = composites_df[comp_col]
    latest_val = _latest_value(series, latest_date)
    delta = _three_month_delta(series, latest_date)
    state_series = states_df[state_col] if state_col in states_df.columns else None
    latest_state = _latest_value(state_series, latest_date) if state_series is not None else None

    label = display
    if latest_state is not None and not pd.isna(latest_state):
        emoji = STATE_EMOJI.get(str(latest_state), '')
        label = f"{emoji} {display}"

    with kpi_cols[idx]:
        st.metric(
            label,
            _format_zscore(latest_val),
            delta=_format_delta(delta),
        )
        st.caption(_state_label(latest_state))

predictions_df = load_predictions_cached()
with kpi_cols[3]:
    prob_label = "Recession Probability (6M)"
    if predictions_df.empty or 'Prob_Ensemble' not in predictions_df.columns:
        st.metric(prob_label, "N/A")
        st.caption("Ensemble context unavailable")
    else:
        try:
            preds = predictions_df.copy()
            if 'Date' in preds.columns:
                preds['Date'] = pd.to_datetime(preds['Date'], errors='coerce')
                preds = preds.dropna(subset=['Date']).sort_values('Date')
            latest_prob = float(preds['Prob_Ensemble'].iloc[-1])
        except Exception:
            latest_prob = float('nan')

        if pd.isna(latest_prob):
            st.metric(prob_label, "N/A")
            st.caption("No data")
        else:
            if latest_prob < 0.15:
                risk_label = "🟢 LOW"
            elif latest_prob < 0.35:
                risk_label = "🟡 MODERATE"
            elif latest_prob < 0.60:
                risk_label = "🟠 ELEVATED"
            else:
                risk_label = "🔴 HIGH"
            st.metric(prob_label, f"{latest_prob:.1%}")
            st.caption(risk_label)

# ---------------------------------------------------------------- row 2: chart

st.markdown("---")
st.markdown("### Composites over time")

recession_series = None
if show_recession_shading and 'RECESSION' in indicators_df.columns:
    recession_series = indicators_df.loc[mask, 'RECESSION']

if composites_view.empty:
    st.info("No data in the selected date range.")
else:
    fig = plot_glr_composites(
        composites_view,
        recession_series=recession_series,
        show_bands=show_bands,
        show_recession_shading=show_recession_shading,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- row 3: drill-down

st.markdown("---")
st.markdown("### Component drill-down")

engine = GLRRegimeEngine()
components_list = engine.components


def _transform_label(transform: str, sign: int) -> str:
    base = {
        'zscore': 'Level z-score',
        'yoy': 'YoY z-score',
        'mom3': '3-month diff z-score',
        'mom1': '1-month diff z-score',
        'level': 'Level z-score (WALCL − WTREGEN − RRPONTSYD)',
        'level_mom3': '3-month diff of net liquidity',
        'ratio': 'Ratio z-score',
        'termstruct': '1 − VIX/VXV, z-score',
    }.get(transform, transform)
    return base + (", negated" if sign == -1 else "")


def _render_composite_section(display: str, composite_key: str, emoji: str):
    members = [c for c in components_list if c.composite == composite_key]
    header = f"{emoji} {display} Components ({len(members)})"
    with st.expander(header, expanded=(composite_key == 'growth')):
        rows = []
        for comp in members:
            col_name = comp.name + ('_neg' if comp.sign == -1 else '')
            latest_z = np.nan
            if not components_df.empty and col_name in components_df.columns:
                latest_z = _latest_value(components_df[col_name], latest_date)
            contribution = latest_z / len(members) if not pd.isna(latest_z) else np.nan
            direction = '🔼' if (not pd.isna(latest_z) and latest_z > 0) else '🔽' if not pd.isna(latest_z) else '—'
            rows.append({
                'Component': comp.description,
                'FRED Series': ', '.join(comp.fred_ids),
                'Transform': _transform_label(comp.transform, comp.sign),
                'Latest z-score': _format_zscore(latest_z),
                'Contribution': _format_zscore(contribution),
                'Direction': direction,
            })
        table_df = pd.DataFrame(rows)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        if components_df.empty:
            st.caption("Component z-score sidecar (data/glr_components.csv) not found — run the scheduler to populate sparklines.")
            return

        five_year_start = latest_date - pd.DateOffset(years=5)
        grid = st.columns(3)
        for i, comp in enumerate(members):
            col_name = comp.name + ('_neg' if comp.sign == -1 else '')
            with grid[i % 3]:
                st.caption(comp.description)
                if col_name in components_df.columns:
                    series = components_df[col_name].loc[five_year_start:latest_date]
                    if series.dropna().empty:
                        st.write("Insufficient history")
                    else:
                        spark_color = GLR_COLORS[COMPOSITE_KEY_TO_COL[composite_key]]
                        fig = plot_glr_sparkline(series, color=spark_color)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.write("Missing component column")


_render_composite_section('Growth', 'growth', '🔵')
_render_composite_section('Liquidity', 'liquidity', '🟢')
_render_composite_section('Risk Appetite', 'risk', '🟠')

# ---------------------------------------------------------------- methodology

with st.expander("ℹ️ Methodology"):
    st.markdown(
        """
        **Construction.** Each composite is the equal-weighted NaN-skipping mean of its
        component z-scores. Each component is one of: a raw FRED series, its
        year-over-year pct change, a 1- or 3-month difference, a derived level
        (net liquidity = WALCL − WTREGEN − RRPONTSYD), or a ratio (copper/gold,
        VIX term structure 1 − VIX/VXV). Every series is then transformed to an
        **expanding-window z-score** with `min_periods=12`, ensuring no look-ahead.

        **Sign convention.** Higher composite = risk-on / expansionary / easy
        liquidity. Indicators that *rise* during stress (ICSA, VIX, HY OAS, NFCI,
        DTWEXBGS, RRPONTSYD, WTREGEN) enter with a negative sign so they pull
        the composite lower when they rise.

        **State labels (weak / neutral / strong).** Assigned via rolling
        **33rd/67th-percentile bands** over a 60-month window (minimum 12 obs).

        **Known caveats.**
        - VXVCLS (3-month VIX) starts Dec 2007 — Risk Appetite is sparse before ~2009.
        - Fed balance sheet series (WALCL, WTREGEN, RRPONTSYD, WLRRAL) begin 2002–2005 —
          Liquidity is sparse before ~2006.
        - Because z-scores use an expanding window, values late in the sample
          reflect the full history, while early values reflect only the window
          available at that time.

        **Not wired into the recession ensemble.** The GLR composites are
        isolated from the ensemble's feature pool. Consult `features.txt` to
        verify the ensemble is not using GLR_* columns.
        """
    )
