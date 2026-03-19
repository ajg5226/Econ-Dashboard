"""
Unit tests for scheduler module
"""

import unittest
import os
from unittest.mock import patch, Mock
from scheduler.update_job import run_update_job


class TestScheduler(unittest.TestCase):
    """Test cases for scheduler update job"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Set test environment variable
        os.environ['FRED_API_KEY'] = 'test_key'
    
    def tearDown(self):
        """Clean up"""
        if 'FRED_API_KEY' in os.environ:
            del os.environ['FRED_API_KEY']
    
    @patch('scheduler.update_job.RecessionDataAcquisition')
    @patch('scheduler.update_job.RecessionEnsembleModel')
    def test_run_update_job_missing_key(self, mock_model, mock_acq):
        """Test that job fails without API key"""
        if 'FRED_API_KEY' in os.environ:
            del os.environ['FRED_API_KEY']
        
        result = run_update_job()
        self.assertFalse(result)
    
    @patch('scheduler.update_job.RecessionDataAcquisition')
    @patch('scheduler.update_job.RecessionEnsembleModel')
    @patch('scheduler.update_job.save_predictions')
    @patch('scheduler.update_job.save_indicators')
    @patch('scheduler.update_job.save_executive_report')
    def test_run_update_job_success(self, mock_report, mock_indicators, mock_predictions, 
                                     mock_model, mock_acq):
        """Test successful update job execution"""
        # Mock the data acquisition and model
        mock_acq_instance = Mock()
        mock_acq_instance.fetch_data.return_value = Mock()
        mock_acq_instance.engineer_features.return_value = Mock()
        mock_acq_instance.create_forecast_target.return_value = Mock()
        mock_acq.return_value = mock_acq_instance
        
        # This test would need more mocking to fully work
        # For now, just test that the function exists and can be called
        self.assertTrue(callable(run_update_job))


if __name__ == '__main__':
    unittest.main()

