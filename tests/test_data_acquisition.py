"""
Unit tests for data acquisition module
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from recession_engine.data_acquisition import RecessionDataAcquisition


class TestDataAcquisition(unittest.TestCase):
    """Test cases for RecessionDataAcquisition"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.acq = RecessionDataAcquisition(fred_api_key=self.api_key)
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.acq.fred_api_key, self.api_key)
        self.assertIsNotNone(self.acq.fred)
        self.assertIsNotNone(self.acq.indicators)
    
    def test_define_indicators(self):
        """Test indicator definition"""
        indicators = self.acq._define_indicators()
        self.assertIn('leading', indicators)
        self.assertIn('coincident', indicators)
        self.assertIn('lagging', indicators)
        self.assertIn('target', indicators)
    
    @patch('recession_engine.data_acquisition.Fred')
    def test_fetch_data(self, mock_fred):
        """Test data fetching"""
        # Mock FRED API response
        mock_series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2020-01-01', periods=3, freq='ME'))
        mock_fred_instance = Mock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred.return_value = mock_fred_instance
        
        acq = RecessionDataAcquisition(fred_api_key="test")
        df = acq.fetch_data(start_date='2020-01-01', end_date='2020-03-01')
        
        self.assertIsInstance(df, pd.DataFrame)
    
    def test_engineer_features(self):
        """Test feature engineering"""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=12, freq='ME')
        df = pd.DataFrame({
            'leading_T10Y2Y': np.random.randn(12),
            'coincident_UNRATE': np.random.randn(12),
            'RECESSION': [0] * 12
        }, index=dates)
        
        df_eng = self.acq.engineer_features(df)
        
        # Check that features were added
        self.assertGreater(len(df_eng.columns), len(df.columns))
        # Check for MoM feature
        self.assertIn('leading_T10Y2Y_MoM', df_eng.columns)
    
    def test_create_forecast_target(self):
        """Test forecast target creation"""
        dates = pd.date_range('2020-01-01', periods=12, freq='ME')
        df = pd.DataFrame({
            'RECESSION': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        }, index=dates)
        
        df_target = self.acq.create_forecast_target(df, horizon_months=6)

        # Check that target column was created
        self.assertIn('RECESSION_FORWARD_6M', df_target.columns)

    def test_drop_dead_series_filters_stale_columns(self):
        """Dead-series detector drops columns more than threshold months
        stale, keeps fresh columns, and exempts ref_* benchmarks."""
        dates = pd.date_range('2020-01-31', '2026-04-30', freq='ME')
        n = len(dates)
        df = pd.DataFrame({
            # Fresh series — values through the end of the panel.
            'fresh_monthly': pd.Series(range(n), index=dates, dtype=float),
            # Discontinued series — last value 2020-05, 70+ months stale.
            'stale_dead': pd.Series([1.0] * 5 + [np.nan] * (n - 5), index=dates),
            # Reference benchmark with the same staleness — must be exempted.
            'ref_benchmark_old': pd.Series([1.0] * 5 + [np.nan] * (n - 5), index=dates),
            # Column with no observations at all.
            'never_published': pd.Series([np.nan] * n, index=dates),
        })

        result = self.acq._drop_dead_series(df, threshold_months=12)

        self.assertIn('fresh_monthly', result.columns)
        self.assertNotIn('stale_dead', result.columns)
        self.assertIn('ref_benchmark_old', result.columns)
        self.assertNotIn('never_published', result.columns)

    def test_drop_dead_series_handles_empty_frame(self):
        """Detector returns empty frame unchanged without raising."""
        empty = pd.DataFrame()
        result = self.acq._drop_dead_series(empty)
        self.assertTrue(result.empty)

    def test_t10y3m_reconstruction_fills_pre_1982(self):
        """When constituents match published exactly on overlap, pre-1982
        rows are filled, the mask flags them, and overlap rows are unchanged."""
        rng = np.random.default_rng(seed=42)
        idx = pd.date_range('1959-01-31', '2026-04-30', freq='ME')
        gs10 = pd.Series(rng.uniform(2, 8, len(idx)), index=idx)
        tb3ms = pd.Series(rng.uniform(0.5, 5, len(idx)), index=idx)
        published = (gs10 - tb3ms).copy()
        published.loc[idx < '1982-01-31'] = np.nan
        df = pd.DataFrame({
            'leading_GS10': gs10,
            'leading_TB3MS': tb3ms,
            'leading_T10Y3M': published,
        }, index=idx)
        out = self.acq._reconstruct_term_spreads(df)
        overlap = out.index >= pd.Timestamp('1982-01-31')
        pre = out.index < pd.Timestamp('1982-01-31')
        # Overlap rows: published kept as-is; offset is zero by construction.
        self.assertEqual(out['leading_T10Y3M_RECONSTRUCTION_OFFSET'].iloc[0], 0.0)
        np.testing.assert_allclose(
            out.loc[overlap, 'leading_T10Y3M'].values,
            published.loc[overlap].values, atol=1e-9,
        )
        # Pre-1982: filled and mask flagged.
        self.assertTrue(out.loc[pre, 'leading_T10Y3M'].notna().all())
        self.assertTrue((out.loc[pre, 'leading_T10Y3M_RECONSTRUCTED_MASK'] == 1.0).all())
        self.assertTrue((out.loc[overlap, 'leading_T10Y3M_RECONSTRUCTED_MASK'] == 0.0).all())

    def test_t10y3m_reconstruction_applies_constant_offset(self):
        """A systematic basis offset should be detected and corrected so the
        pre-1982 fill aligns with the published-series scale."""
        idx = pd.date_range('1979-01-31', '1985-12-31', freq='ME')
        gs10 = pd.Series(np.linspace(8.0, 12.0, len(idx)), index=idx)
        tb3ms = pd.Series(np.linspace(7.0, 10.0, len(idx)), index=idx)
        # Published has a +0.15pp basis offset over constituents on overlap.
        # Pre-1982 NaN, post-1982 published.
        published = (gs10 - tb3ms) + 0.15
        published.loc[idx < pd.Timestamp('1982-01-31')] = np.nan
        df = pd.DataFrame({
            'leading_GS10': gs10,
            'leading_TB3MS': tb3ms,
            'leading_T10Y3M': published,
        }, index=idx)
        out = self.acq._reconstruct_term_spreads(df)
        # Detected offset should be ~+0.15pp.
        self.assertAlmostEqual(
            out['leading_T10Y3M_RECONSTRUCTION_OFFSET'].iloc[0], 0.15, places=4,
        )
        # Pre-1982 fill = (gs10 - tb3ms) + offset = published-equivalent.
        pre = idx < pd.Timestamp('1982-01-31')
        expected_pre = (gs10 - tb3ms + 0.15).loc[pre]
        np.testing.assert_allclose(
            out.loc[pre, 'leading_T10Y3M'].values, expected_pre.values, atol=1e-9,
        )

    def test_t10y3m_reconstruction_warns_on_high_residual(self):
        """High noise after offset correction (mean abs residual >0.30pp)
        warns but does not raise."""
        rng = np.random.default_rng(seed=7)
        idx = pd.date_range('1982-01-31', '1990-12-31', freq='ME')
        gs10 = pd.Series(np.full(len(idx), 5.0), index=idx)
        tb3ms = pd.Series(np.full(len(idx), 2.0), index=idx)
        # Inject 0.50pp std noise — mean abs residual ~0.40pp, above warn but below fail.
        published = (gs10 - tb3ms) + rng.normal(0.0, 0.50, len(idx))
        df = pd.DataFrame({
            'leading_GS10': gs10,
            'leading_TB3MS': tb3ms,
            'leading_T10Y3M': published,
        }, index=idx)
        with self.assertLogs('recession_engine.data_acquisition', level='WARNING') as ctx:
            self.acq._reconstruct_term_spreads(df)
        self.assertTrue(any('mean abs residual' in m for m in ctx.output))

    def test_t10y3m_reconstruction_raises_on_catastrophic_residual(self):
        """A wrong-series swap (TB3MS sign flipped) should produce huge
        residuals that fail the gate."""
        idx = pd.date_range('1982-01-31', '1985-12-31', freq='ME')
        gs10 = pd.Series(np.full(len(idx), 5.0), index=idx)
        tb3ms = pd.Series(np.full(len(idx), 2.0), index=idx)
        # If we accidentally subtracted the wrong series, residual would be huge.
        published = pd.Series(np.full(len(idx), 7.0), index=idx)  # ~4pp off after offset
        df = pd.DataFrame({
            'leading_GS10': gs10,
            'leading_TB3MS': tb3ms,
            'leading_T10Y3M': published,
        }, index=idx)
        # Mean abs residual after offset will be ~0 actually since it's a constant offset.
        # To trigger the fail, we need residual variance >1.0pp. Inject scaling.
        published = pd.Series(np.linspace(0.0, 10.0, len(idx)), index=idx)
        df['leading_T10Y3M'] = published
        with self.assertRaises(ValueError):
            self.acq._reconstruct_term_spreads(df)

    def test_t10y3m_reconstruction_noop_when_constituents_missing(self):
        """If GS10 isn't fetched, the reconstruction is a no-op."""
        idx = pd.date_range('1982-01-31', '1985-12-31', freq='ME')
        df = pd.DataFrame({
            'leading_TB3MS': pd.Series(np.full(len(idx), 2.0), index=idx),
            'leading_T10Y3M': pd.Series(np.full(len(idx), 3.0), index=idx),
        }, index=idx)
        out = self.acq._reconstruct_term_spreads(df)
        # No mask column added; T10Y3M unchanged.
        self.assertNotIn('leading_T10Y3M_RECONSTRUCTED_MASK', out.columns)
        pd.testing.assert_series_equal(out['leading_T10Y3M'], df['leading_T10Y3M'])


if __name__ == '__main__':
    unittest.main()

