"""
Unit tests for the GLR regime engine and shared transforms.
"""

import unittest

import numpy as np
import pandas as pd

from recession_engine.glr_engine import GLRRegimeEngine
from recession_engine.transforms import expanding_zscore, rolling_tertile_state


def _monthly_index(n: int, start: str = '2000-01-01') -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq='ME')


def _all_sources_frame(n: int = 200) -> pd.DataFrame:
    """Build a monthly frame containing every column the engine consults,
    populated with random-but-finite values. Tests that need specific
    dynamics override individual columns."""
    rng = np.random.default_rng(42)
    idx = _monthly_index(n)
    cols = [
        'coincident_INDPRO', 'coincident_PAYEMS', 'coincident_RSXFS',
        'leading_HOUST', 'leading_PERMIT', 'leading_UMCSENT',
        'leading_ICSA', 'leading_DGORDER', 'leading_NEWORDER',
        'glr_CUMFNS', 'glr_M2SL', 'glr_WALCL', 'glr_WTREGEN',
        'glr_RRPONTSYD', 'glr_WLRRAL', 'glr_VIXCLS', 'glr_VXVCLS',
        'glr_DTWEXBGS', 'glr_PCOPPUSDM', 'glr_IQ12260',
        'financial_BAMLH0A0HYM2', 'financial_NFCI',
    ]
    data = {c: 100 + rng.standard_normal(n).cumsum() for c in cols}
    return pd.DataFrame(data, index=idx)


class TestExpandingZscore(unittest.TestCase):

    def test_first_obs_below_min_periods_are_nan(self):
        s = pd.Series(np.arange(24.0))
        z = expanding_zscore(s, min_periods=12)
        self.assertTrue(z.iloc[:11].isna().all())
        self.assertFalse(z.iloc[11:].isna().any())

    def test_no_lookahead(self):
        """z-score at date t must equal the z-score computed from the
        truncated series[: t+1]."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(48))
        z_full = expanding_zscore(s, min_periods=12)
        t = 30
        z_truncated = expanding_zscore(s.iloc[: t + 1], min_periods=12)
        self.assertAlmostEqual(z_full.iloc[t], z_truncated.iloc[t], places=10)

    def test_zero_std_produces_nan_not_inf(self):
        s = pd.Series([5.0] * 20)
        z = expanding_zscore(s, min_periods=12)
        self.assertFalse(np.isinf(z).any())


class TestRollingTertileState(unittest.TestCase):

    def test_min_periods_are_nan(self):
        s = pd.Series(np.arange(30.0))
        labels = rolling_tertile_state(s, window=20, min_periods=12)
        self.assertTrue(labels.iloc[:11].isna().all())

    def test_labels_track_position_vs_bands(self):
        rng = np.random.default_rng(1)
        s = pd.Series(rng.standard_normal(120))
        labels = rolling_tertile_state(s, window=60, min_periods=12,
                                       low_q=0.33, high_q=0.67)
        finite = labels.dropna()
        self.assertTrue(set(finite.unique()).issubset({'weak', 'neutral', 'strong'}))
        counts = finite.value_counts()
        self.assertGreater(counts.get('weak', 0), 0)
        self.assertGreater(counts.get('strong', 0), 0)


class TestGLREngineRegistry(unittest.TestCase):

    def test_twenty_three_components(self):
        engine = GLRRegimeEngine()
        comps = engine.components
        self.assertEqual(len(comps), 23)
        by_comp = {}
        for c in comps:
            by_comp.setdefault(c.composite, []).append(c)
        self.assertEqual(len(by_comp['growth']), 10)
        self.assertEqual(len(by_comp['liquidity']), 7)
        self.assertEqual(len(by_comp['risk']), 6)


class TestGLREngineBuild(unittest.TestCase):

    def test_build_returns_three_frames(self):
        df = _all_sources_frame(200)
        result = GLRRegimeEngine().build(df)
        self.assertIn('components', result)
        self.assertIn('composites', result)
        self.assertIn('states', result)
        self.assertEqual(len(result['components']), len(df))
        self.assertEqual(len(result['composites']), len(df))
        self.assertEqual(len(result['states']), len(df))

    def test_composite_columns_present(self):
        df = _all_sources_frame(200)
        result = GLRRegimeEngine().build(df)
        for col in ('GLR_GROWTH', 'GLR_LIQUIDITY', 'GLR_RISK_APPETITE'):
            self.assertIn(col, result['composites'].columns)
        for col in ('GLR_GROWTH_STATE', 'GLR_LIQUIDITY_STATE', 'GLR_RISK_APPETITE_STATE'):
            self.assertIn(col, result['states'].columns)

    def test_missing_source_column_produces_nan_not_error(self):
        df = _all_sources_frame(200).drop(columns=['glr_VXVCLS'])
        result = GLRRegimeEngine().build(df)
        self.assertTrue(result['components']['risk_vix_term_struct'].isna().all())
        self.assertIn('GLR_RISK_APPETITE', result['composites'].columns)


class TestGLRSignFlip(unittest.TestCase):

    def test_rising_icsa_reduces_growth_contribution(self):
        """ICSA rising monotonically → growth_ICSA_neg z-score trends down."""
        df = _all_sources_frame(200)
        df['leading_ICSA'] = np.linspace(200_000, 500_000, 200)
        result = GLRRegimeEngine().build(df)
        series = result['components']['growth_ICSA_neg'].dropna()
        self.assertGreater(len(series), 100)
        head_mean = series.iloc[:20].mean()
        tail_mean = series.iloc[-20:].mean()
        self.assertGreater(
            head_mean, tail_mean,
            msg="Rising ICSA should produce DECREASING growth_ICSA_neg",
        )


class TestGLRFormulas(unittest.TestCase):

    def test_net_liquidity_level_formula(self):
        """liquidity_net_liquidity_level = zscore(WALCL − WTREGEN − RRPONTSYD)."""
        df = _all_sources_frame(200)
        expected_raw = df['glr_WALCL'] - df['glr_WTREGEN'] - df['glr_RRPONTSYD']
        expected_z = expanding_zscore(expected_raw, min_periods=12)
        engine = GLRRegimeEngine()
        components = engine.build_components(df)
        pd.testing.assert_series_equal(
            components['liquidity_net_liquidity_level'].rename(None),
            expected_z.rename(None),
            check_names=False,
        )

    def test_vix_term_structure_formula(self):
        df = _all_sources_frame(200)
        expected_raw = 1.0 - (df['glr_VIXCLS'] / df['glr_VXVCLS'])
        expected_z = expanding_zscore(expected_raw, min_periods=12)
        engine = GLRRegimeEngine()
        components = engine.build_components(df)
        pd.testing.assert_series_equal(
            components['risk_vix_term_struct'].rename(None),
            expected_z.rename(None),
            check_names=False,
        )

    def test_copper_gold_ratio_formula(self):
        df = _all_sources_frame(200)
        expected_raw = df['glr_PCOPPUSDM'] / df['glr_IQ12260']
        expected_z = expanding_zscore(expected_raw, min_periods=12)
        engine = GLRRegimeEngine()
        components = engine.build_components(df)
        pd.testing.assert_series_equal(
            components['risk_copper_gold_ratio'].rename(None),
            expected_z.rename(None),
            check_names=False,
        )


class TestGLREqualWeightAggregation(unittest.TestCase):

    def test_composite_equals_nan_skipping_mean(self):
        """Composite row = mean of present component z-scores."""
        df = _all_sources_frame(200)
        engine = GLRRegimeEngine()
        components = engine.build_components(df)
        composites = engine.build_composites(components)

        growth_cols = [
            c.name + ('_neg' if c.sign == -1 else '')
            for c in engine.components if c.composite == 'growth'
        ]
        latest_idx = df.index[-1]
        expected = components.loc[latest_idx, growth_cols].mean(skipna=True)
        actual = composites.loc[latest_idx, 'GLR_GROWTH']
        self.assertAlmostEqual(float(actual), float(expected), places=10)


if __name__ == '__main__':
    unittest.main()
