#!/usr/bin/env python3
"""
Comprehensive API Tests for Flask Application - FIXED VERSION
Tests all API endpoints with unit, integration, and acceptance tests
"""

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class TestFlaskAPI(unittest.TestCase):
    """Test suite for Flask API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        # Import after path setup
        from app import app
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        
        # Create sample data for mocking
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        prices = []
        initial_price = 4000
        for i in range(len(dates)):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            if i == 0:
                prices.append(initial_price)
            else:
                prices.append(prices[-1] * (1 + change))
        
        cls.sample_market_data = pd.DataFrame({
            'Open': np.array(prices) * np.random.uniform(0.995, 1.005, len(prices)),
            'High': np.array(prices) * np.random.uniform(1.005, 1.02, len(prices)),
            'Low': np.array(prices) * np.random.uniform(0.98, 0.995, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=dates)
        
        # Fix OHLC relationships
        cls.sample_market_data['High'] = np.maximum(
            cls.sample_market_data['High'], 
            np.maximum(cls.sample_market_data['Open'], cls.sample_market_data['Close'])
        )
        cls.sample_market_data['Low'] = np.minimum(
            cls.sample_market_data['Low'], 
            np.minimum(cls.sample_market_data['Open'], cls.sample_market_data['Close'])
        )

    def test_index_route(self):
        """Test main index route"""
        response = self.client.get('/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())

    def test_api_status_endpoint(self):
        """Test API status endpoint"""
        response = self.client.get('/api/status')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('status', data)
        
        status = data['status']
        required_keys = [
            'model_trained', 'data_connection', 'features_count',
            'cache_active', 'timestamp'
        ]
        
        for key in required_keys:
            self.assertIn(key, status, f"Missing status key: {key}")

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    @patch('sp500_predictor.EnhancedSP500Predictor.load_model')
    @patch('sp500_predictor.EnhancedSP500Predictor.predict_next_day')
    def test_api_predict_success(self, mock_predict, mock_load, mock_download):
        """Test successful prediction API call"""
        # Mock data download
        mock_download.return_value = self.sample_market_data
        
        # Mock model loading
        mock_load.return_value = True
        
        # Mock prediction result
        mock_predict.return_value = {
            'prediction': 'UP',
            'probability_up': 0.65,
            'probability_down': 0.35,
            'confidence': 0.65,
            'market_regime': 'Bull Market',
            'vix_level': 18.5,
            'trend_strength': 0.3
        }
        
        response = self.client.post('/api/predict')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        prediction_data = data['data']
        required_keys = [
            'prediction', 'probability_up', 'probability_down',
            'confidence', 'market_regime', 'last_price', 'timestamp'
        ]
        
        for key in required_keys:
            self.assertIn(key, prediction_data, f"Missing prediction key: {key}")
        
        # Validate prediction values
        self.assertIn(prediction_data['prediction'], ['UP', 'DOWN'])
        self.assertGreaterEqual(prediction_data['probability_up'], 0)
        self.assertLessEqual(prediction_data['probability_up'], 1)
        self.assertGreaterEqual(prediction_data['probability_down'], 0)
        self.assertLessEqual(prediction_data['probability_down'], 1)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    def test_api_predict_data_failure(self, mock_download):
        """Test prediction API with data download failure"""
        mock_download.return_value = None
        
        response = self.client.post('/api/predict')
        
        self.assertEqual(response.status_code, 500)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    @patch('sp500_predictor.EnhancedSP500Predictor.load_model')
    @patch('sp500_predictor.EnhancedSP500Predictor.create_features')
    def test_api_predict_feature_creation_failure(self, mock_features, mock_load, mock_download):
        """Test prediction API with feature creation failure"""
        mock_download.return_value = self.sample_market_data
        mock_load.return_value = False
        mock_features.return_value = None
        
        response = self.client.post('/api/predict')
        
        self.assertEqual(response.status_code, 500)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    @patch('sp500_predictor.EnhancedSP500Predictor.load_model')
    @patch('sp500_predictor.EnhancedSP500Predictor.backtest_strategy')
    def test_api_backtest_success(self, mock_backtest, mock_load, mock_download):
        """Test successful backtest API call - FIXED VERSION"""
        mock_download.return_value = self.sample_market_data
        mock_load.return_value = True
        
        mock_backtest.return_value = {
            'period': '2022-01-01 to 2023-12-31',
            'total_days': 365,
            'trading_days': 250,
            'accuracy': 0.62,
            'win_rate': 0.55,
            'strategy_return': 0.15,
            'buyhold_return': 0.12,
            'excess_return': 0.03,
            'strategy_sharpe': 1.2,
            'buyhold_sharpe': 1.0,
            'strategy_mdd': -0.08,
            'buyhold_mdd': -0.10,
            'confidence_threshold': 0.6
        }
        
        # FIXED: Use proper JSON content type
        response = self.client.post('/api/backtest',
                                  data=json.dumps({'days_back': 365}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        backtest_data = data['data']
        required_keys = [
            'strategy_return', 'buyhold_return', 'excess_return',
            'accuracy', 'win_rate', 'strategy_sharpe'
        ]
        
        for key in required_keys:
            self.assertIn(key, backtest_data, f"Missing backtest key: {key}")
        
        # Validate backtest values
        self.assertGreaterEqual(backtest_data['accuracy'], 0)
        self.assertLessEqual(backtest_data['accuracy'], 1)
        self.assertGreaterEqual(backtest_data['win_rate'], 0)
        self.assertLessEqual(backtest_data['win_rate'], 1)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    @patch('sp500_predictor.EnhancedSP500Predictor.load_model')
    @patch('sp500_predictor.EnhancedSP500Predictor.backtest_strategy')
    def test_api_backtest_data_failure(self, mock_backtest, mock_load, mock_download):
        """Test backtest API with data download failure"""
        mock_download.return_value = None
        
        response = self.client.post('/api/backtest', 
                                  data=json.dumps({'days_back': 365}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    @patch('sp500_predictor.EnhancedSP500Predictor.create_features')
    @patch('sp500_predictor.EnhancedSP500Predictor.prepare_data')
    @patch('sp500_predictor.EnhancedSP500Predictor.train_model')
    @patch('sp500_predictor.EnhancedSP500Predictor.save_model')
    def test_api_train_success(self, mock_save, mock_train, mock_prepare, mock_features, mock_download):
        """Test successful training API call"""
        mock_download.return_value = self.sample_market_data
        mock_features.return_value = self.sample_market_data  # Simplified for test
        mock_prepare.return_value = (self.sample_market_data[['Close']], pd.Series([1, 0, 1]))
        mock_train.return_value = MagicMock()  # Mock model
        mock_save.return_value = True
        
        response = self.client.post('/api/train')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('features_count', data)
        self.assertIn('data_points', data)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    def test_api_refresh_success(self, mock_download):
        """Test successful data refresh API call"""
        mock_download.return_value = self.sample_market_data
        
        response = self.client.post('/api/refresh')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('data_points', data)
        self.assertIn('latest_date', data)
        self.assertIn('latest_price', data)

    def test_api_history_endpoint(self):
        """Test prediction history endpoint"""
        response = self.client.get('/api/history')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)

    @patch('sp500_predictor.EnhancedSP500Predictor.download_data')
    def test_api_chart_success(self, mock_download):
        """Test successful chart data API call"""
        mock_download.return_value = self.sample_market_data
        
        response = self.client.post('/api/chart',
                                  data=json.dumps({'period': '3mo'}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        chart_data = data['data']
        required_keys = ['prices', 'dates', 'stats', 'period']
        
        for key in required_keys:
            self.assertIn(key, chart_data, f"Missing chart key: {key}")
        
        # Validate data structure
        self.assertIsInstance(chart_data['prices'], list)
        self.assertIsInstance(chart_data['dates'], list)
        self.assertEqual(len(chart_data['prices']), len(chart_data['dates']))


if __name__ == '__main__':
    unittest.main(verbosity=2)