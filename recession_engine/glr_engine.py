"""
GLR Regime Engine — Growth / Liquidity / Risk Appetite composite signals.

Three equal-weighted composites of FRED indicators, each transformed to
expanding z-scores (min_periods=12) then averaged. Sign convention:
higher composite = risk-on / expansionary / easy liquidity.

Consumes the monthly panel produced by
RecessionDataAcquisition.engineer_features and returns three aligned
frames: per-component z-scores, composites, and tertiary state labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .transforms import expanding_zscore, rolling_tertile_state

logger = logging.getLogger(__name__)

Transform = Literal[
    'zscore',       # raw series → z-score
    'yoy',          # 12-month pct_change → z-score
    'mom3',         # 3-month diff → z-score
    'mom1',         # 1-month diff → z-score
    'level',        # source[0] − source[1] − source[2] → z-score
    'level_mom3',   # level, then 3-month diff → z-score
    'ratio',        # source[0] / source[1] → z-score
    'termstruct',   # 1 − source[0]/source[1] → z-score
]

Composite = Literal['growth', 'liquidity', 'risk']

COMPOSITE_COLS = {
    'growth': 'GLR_GROWTH',
    'liquidity': 'GLR_LIQUIDITY',
    'risk': 'GLR_RISK_APPETITE',
}

STATE_COLS = {
    'growth': 'GLR_GROWTH_STATE',
    'liquidity': 'GLR_LIQUIDITY_STATE',
    'risk': 'GLR_RISK_APPETITE_STATE',
}


@dataclass(frozen=True)
class GLRComponent:
    """Declarative description of one GLR composite input."""
    name: str                  # e.g. 'growth_INDPRO' — column in components frame
    source_cols: tuple         # input column names from indicators_df
    transform: Transform
    sign: int                  # +1 or -1, applied after z-score
    composite: Composite
    fred_ids: tuple            # FRED series IDs surfaced in drill-down table
    description: str           # human-readable label for drill-down


class GLRRegimeEngine:
    """Build Growth / Liquidity / Risk Appetite composites.

    Usage:
        engine = GLRRegimeEngine()
        result = engine.build(indicators_df)
        # result = {'components': DataFrame, 'composites': DataFrame, 'states': DataFrame}
    """

    def __init__(
        self,
        min_periods_zscore: int = 12,
        state_window: int = 60,
        state_min_periods: int = 12,
    ):
        self.min_periods_zscore = min_periods_zscore
        self.state_window = state_window
        self.state_min_periods = state_min_periods
        self._components: tuple = self._define_components()

    @property
    def components(self) -> tuple:
        """Expose the component registry for drill-down UIs."""
        return self._components

    # ------------------------------------------------------------------
    # Component registry — single source of truth for the GLR spec
    # ------------------------------------------------------------------

    @staticmethod
    def _define_components() -> tuple:
        """Return the 10 + 7 + 6 = 23 GLR components in registry form."""
        return (
            # ── Growth (10) ────────────────────────────────────────
            GLRComponent('growth_INDPRO', ('coincident_INDPRO',),
                         'yoy', +1, 'growth', ('INDPRO',),
                         'Industrial Production YoY'),
            GLRComponent('growth_PAYEMS', ('coincident_PAYEMS',),
                         'yoy', +1, 'growth', ('PAYEMS',),
                         'Nonfarm Payrolls YoY'),
            GLRComponent('growth_RSXFS', ('coincident_RSXFS',),
                         'yoy', +1, 'growth', ('RSXFS',),
                         'Retail Sales YoY'),
            GLRComponent('growth_HOUST', ('leading_HOUST',),
                         'yoy', +1, 'growth', ('HOUST',),
                         'Housing Starts YoY'),
            GLRComponent('growth_PERMIT', ('leading_PERMIT',),
                         'yoy', +1, 'growth', ('PERMIT',),
                         'Building Permits YoY'),
            GLRComponent('growth_UMCSENT', ('leading_UMCSENT',),
                         'yoy', +1, 'growth', ('UMCSENT',),
                         'Consumer Sentiment YoY'),
            GLRComponent('growth_ICSA', ('leading_ICSA',),
                         'yoy', -1, 'growth', ('ICSA',),
                         'Initial Claims YoY (negated)'),
            GLRComponent('growth_CUMFNS', ('glr_CUMFNS',),
                         'yoy', +1, 'growth', ('CUMFNS',),
                         'Capacity Utilization YoY'),
            GLRComponent('growth_DGORDER', ('leading_DGORDER',),
                         'yoy', +1, 'growth', ('DGORDER',),
                         'Durable Goods Orders YoY'),
            GLRComponent('growth_NEWORDER', ('leading_NEWORDER',),
                         'yoy', +1, 'growth', ('NEWORDER',),
                         'New Orders (Consumer Goods) YoY'),

            # ── Liquidity (7) ─────────────────────────────────────
            GLRComponent('liquidity_M2', ('glr_M2SL',),
                         'yoy', +1, 'liquidity', ('M2SL',),
                         'M2 Money Stock YoY'),
            GLRComponent('liquidity_WALCL_3mdiff', ('glr_WALCL',),
                         'mom3', +1, 'liquidity', ('WALCL',),
                         'Fed Balance Sheet 3-month diff'),
            GLRComponent('liquidity_net_liquidity_level',
                         ('glr_WALCL', 'glr_WTREGEN', 'glr_RRPONTSYD'),
                         'level', +1, 'liquidity',
                         ('WALCL', 'WTREGEN', 'RRPONTSYD'),
                         'Net Liquidity Level (WALCL − WTREGEN − RRPONTSYD)'),
            GLRComponent('liquidity_net_liquidity_3mdiff',
                         ('glr_WALCL', 'glr_WTREGEN', 'glr_RRPONTSYD'),
                         'level_mom3', +1, 'liquidity',
                         ('WALCL', 'WTREGEN', 'RRPONTSYD'),
                         'Net Liquidity 3-month diff'),
            GLRComponent('liquidity_RRPONTSYD', ('glr_RRPONTSYD',),
                         'zscore', -1, 'liquidity', ('RRPONTSYD',),
                         'Overnight RRP Level (negated)'),
            GLRComponent('liquidity_WTREGEN_1mdiff', ('glr_WTREGEN',),
                         'mom1', -1, 'liquidity', ('WTREGEN',),
                         'Treasury General Account 1-month diff (negated)'),
            GLRComponent('liquidity_WLRRAL', ('glr_WLRRAL',),
                         'zscore', +1, 'liquidity', ('WLRRAL',),
                         'Bank Reserves at the Fed'),

            # ── Risk Appetite (6) ──────────────────────────────────
            GLRComponent('risk_vix_term_struct', ('glr_VIXCLS', 'glr_VXVCLS'),
                         'termstruct', +1, 'risk', ('VIXCLS', 'VXVCLS'),
                         'VIX Term Structure (1 − VIX/VXV)'),
            GLRComponent('risk_VIXCLS', ('glr_VIXCLS',),
                         'zscore', -1, 'risk', ('VIXCLS',),
                         'VIX Close (negated)'),
            GLRComponent('risk_HY_OAS', ('financial_BAMLH0A0HYM2',),
                         'zscore', -1, 'risk', ('BAMLH0A0HYM2',),
                         'HY OAS (negated)'),
            GLRComponent('risk_NFCI', ('financial_NFCI',),
                         'zscore', -1, 'risk', ('NFCI',),
                         'Chicago Fed NFCI (negated)'),
            GLRComponent('risk_copper_gold_ratio',
                         ('glr_PCOPPUSDM', 'glr_IQ12260'),
                         'ratio', +1, 'risk',
                         ('PCOPPUSDM', 'IQ12260'),
                         'Copper / Gold Ratio (IQ12260 as gold proxy)'),
            GLRComponent('risk_DTWEXBGS', ('glr_DTWEXBGS',),
                         'zscore', -1, 'risk', ('DTWEXBGS',),
                         'Broad Trade-Weighted USD Index (negated)'),
        )

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _apply_transform(
        self,
        component: GLRComponent,
        indicators_df: pd.DataFrame,
    ) -> pd.Series:
        """Dispatch on component.transform and return a series aligned to
        indicators_df.index. Missing source columns produce an all-NaN
        series (caller handles it)."""
        missing = [c for c in component.source_cols if c not in indicators_df.columns]
        if missing:
            logger.warning(
                "GLR component %s missing source columns %s — emitting NaN",
                component.name, missing,
            )
            return pd.Series(np.nan, index=indicators_df.index, dtype=float)

        transform = component.transform
        sources = [indicators_df[c] for c in component.source_cols]

        if transform == 'zscore':
            raw = sources[0]
        elif transform == 'yoy':
            raw = sources[0].pct_change(12)
        elif transform == 'mom3':
            raw = sources[0].diff(3)
        elif transform == 'mom1':
            raw = sources[0].diff(1)
        elif transform == 'level':
            raw = sources[0] - sources[1] - sources[2]
        elif transform == 'level_mom3':
            raw = (sources[0] - sources[1] - sources[2]).diff(3)
        elif transform == 'ratio':
            denom = sources[1].where(sources[1].abs() > 1e-12, np.nan)
            raw = sources[0] / denom
        elif transform == 'termstruct':
            denom = sources[1].where(sources[1].abs() > 1e-12, np.nan)
            raw = 1.0 - (sources[0] / denom)
        else:
            raise ValueError(f"Unknown transform: {transform}")

        raw = raw.replace([np.inf, -np.inf], np.nan)
        return expanding_zscore(raw, min_periods=self.min_periods_zscore)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_components(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """Return a wide frame of per-component z-scores. Columns are
        '<component.name>' with an optional '_neg' suffix when sign=-1."""
        if not isinstance(indicators_df.index, pd.DatetimeIndex):
            raise TypeError("indicators_df must have a DatetimeIndex")

        cols = {}
        for comp in self._components:
            z = self._apply_transform(comp, indicators_df)
            z = z * comp.sign
            col_name = comp.name + ('_neg' if comp.sign == -1 else '')
            cols[col_name] = z
        return pd.DataFrame(cols, index=indicators_df.index)

    def build_composites(self, components_df: pd.DataFrame) -> pd.DataFrame:
        """Equal-weight NaN-skipping mean per composite.

        A month with zero non-NaN components in a composite yields NaN for
        that composite; any non-NaN input lets the composite render.
        """
        out = pd.DataFrame(index=components_df.index)
        for composite_key, out_col in COMPOSITE_COLS.items():
            member_cols = [
                comp.name + ('_neg' if comp.sign == -1 else '')
                for comp in self._components
                if comp.composite == composite_key
            ]
            present = [c for c in member_cols if c in components_df.columns]
            if not present:
                out[out_col] = np.nan
                continue
            out[out_col] = components_df[present].mean(axis=1, skipna=True)
        return out

    def build_states(self, composites_df: pd.DataFrame) -> pd.DataFrame:
        """Assign weak/neutral/strong labels via rolling 33rd/67th-percentile
        bands on each composite."""
        out = pd.DataFrame(index=composites_df.index)
        for composite_key, comp_col in COMPOSITE_COLS.items():
            state_col = STATE_COLS[composite_key]
            if comp_col not in composites_df.columns:
                out[state_col] = np.nan
                continue
            out[state_col] = rolling_tertile_state(
                composites_df[comp_col],
                window=self.state_window,
                min_periods=self.state_min_periods,
            )
        return out

    def build(self, indicators_df: pd.DataFrame) -> dict:
        """Convenience wrapper. Returns dict with 'components', 'composites',
        and 'states' frames, all aligned to indicators_df.index."""
        components = self.build_components(indicators_df)
        composites = self.build_composites(components)
        states = self.build_states(composites)
        return {
            'components': components,
            'composites': composites,
            'states': states,
        }
