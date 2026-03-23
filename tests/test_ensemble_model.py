"""
Unit tests for ensemble model module
"""

import unittest
import pandas as pd
import numpy as np
from recession_engine.ensemble_model import RecessionEnsembleModel


class TestEnsembleModel(unittest.TestCase):
    """Test cases for RecessionEnsembleModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = RecessionEnsembleModel(target_horizon=6)
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.model.target_horizon, 6)
        self.assertEqual(self.model.target_col, 'RECESSION_FORWARD_6M')
        self.assertIsNotNone(self.model.models)
        self.assertIsInstance(self.model.active_models, list)
        self.assertFalse(self.model.is_fitted)
    
    def test_prepare_data(self):
        """Test data preparation"""
        dates = pd.date_range('2020-01-01', periods=100, freq='ME')
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'RECESSION_FORWARD_6M': np.random.randint(0, 2, 100),
            'RECESSION': np.random.randint(0, 2, 100)
        }, index=dates)
        
        # Use default expanding split to avoid tiny train folds that can contain one class
        train_df, test_df = self.model.prepare_data(df)
        
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
    
    def test_select_features(self):
        """Test feature selection"""
        dates = pd.date_range('2020-01-01', periods=200, freq='ME')
        df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200),
            'RECESSION_FORWARD_6M': np.random.randint(0, 2, 200),
            'RECESSION_FORWARD_12M': np.random.randint(0, 2, 200),
            'RECESSION': np.random.randint(0, 2, 200)
        }, index=dates)
        
        features = self.model.select_features(df, max_features=2)
        
        self.assertIsInstance(features, list)
        self.assertLessEqual(len(features), 2)
        self.assertNotIn('RECESSION_FORWARD_12M', features)
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction"""
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=100, freq='ME')
        np.random.seed(42)
        
        # Create correlated features
        feature1 = np.random.randn(100)
        feature2 = np.random.randn(100)
        target = ((feature1 + feature2) > 0).astype(int)
        
        df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'RECESSION_FORWARD_6M': target,
            'RECESSION': target
        }, index=dates)
        
        # Use default expanding split to avoid tiny train folds with one class.
        train_df, test_df = self.model.prepare_data(df)
        
        # Fit model
        self.model.fit(train_df, feature_cols=['feature1', 'feature2'])
        
        self.assertTrue(self.model.is_fitted)
        
        # Make predictions
        predictions = self.model.predict(test_df)
        
        self.assertIn('ensemble', predictions)
        self.assertIn('probit', predictions)
        self.assertEqual(len(predictions['ensemble']), len(test_df))


if __name__ == '__main__':
    unittest.main()

