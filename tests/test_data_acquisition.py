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
        mock_series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2020-01-01', periods=3, freq='M'))
        mock_fred_instance = Mock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred.return_value = mock_fred_instance
        
        acq = RecessionDataAcquisition(fred_api_key="test")
        df = acq.fetch_data(start_date='2020-01-01', end_date='2020-03-01')
        
        self.assertIsInstance(df, pd.DataFrame)
    
    def test_engineer_features(self):
        """Test feature engineering"""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=12, freq='M')
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
        dates = pd.date_range('2020-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'RECESSION': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        }, index=dates)
        
        df_target = self.acq.create_forecast_target(df, horizon_months=6)
        
        # Check that target column was created
        self.assertIn('RECESSION_FORWARD_6M', df_target.columns)


if __name__ == '__main__':
    unittest.main()

