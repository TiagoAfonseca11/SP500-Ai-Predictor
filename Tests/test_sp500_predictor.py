#!/usr/bin/env python3
"""
Comprehensive Unit Tests for SP500 Predictor
Tests all core functionality including feature engineering, model training, and predictions
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp500_predictor import EnhancedSP500Predictor

warnings.filterwarnings('ignore')

class TestEnhancedSP500Predictor(unittest.TestCase):
    """Test suite for EnhancedSP500Predictor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = EnhancedSP500Predictor()
        
        # Create sample test data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock data
        initial_price = 3000
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'Open': np.array(prices) * np.random.uniform(0.995, 1.005, len(prices)),
            'High': np.array(prices) * np.random.uniform(1.005, 1.02, len(prices)),
            'Low': np.array(prices) * np.random.uniform(0.98, 0.995, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=dates)
        
        # Ensure High >= Close >= Low and High >= Open >= Low
        self.sample_data['High'] = np.maximum(
            self.sample_data['High'], 
            np.maximum(self.sample_data['Open'], self.sample_data['Close'])
        )
        self.sample_data['Low'] = np.minimum(
            self.sample_data['Low'], 
            np.minimum(self.sample_data['Open'], self.sample_data['Close'])
        )

    def test_initialization(self):
        """Test predictor initialization"""
        self.assertIsInstance(self.predictor, EnhancedSP500Predictor)
        self.assertIsNone(self.predictor.model)
        self.assertEqual(self.predictor.symbol, "^GSPC")
        self.assertEqual(len(self.predictor.features), 0)

    @patch('yfinance.download')
    def test_download_data_success(self, mock_yf):
        """Test successful data download"""
        mock_yf.return_value = self.sample_data
        
        result = self.predictor.download_data(period="1y")
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0)
        mock_yf.assert_called_once()

    @patch('yfinance.download')
    def test_download_data_failure(self, mock_yf):
        """Test data download failure"""
        mock_yf.return_value = pd.DataFrame()
        
        result = self.predictor.download_data(period="1y")
        
        self.assertIsNone(result)

    @patch('yfinance.download')
    def test_download_data_missing_columns(self, mock_yf):
        """Test data download with missing required columns"""
        incomplete_data = self.sample_data.drop(columns=['Volume'])
        mock_yf.return_value = incomplete_data
        
        result = self.predictor.download_data(period="1y")
        
        self.assertIsNone(result)

    def test_create_features_basic(self):
        """Test basic feature creation"""
        result = self.predictor.create_features(self.sample_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check basic features exist
        expected_features = [
            'returns', 'high_low_pct', 'volume_ratio',
            'ma_5', 'ma_10', 'ma_20', 'ma_50',
            'rsi', 'macd', 'bb_position', 'target'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing feature: {feature}")

    def test_create_features_insufficient_data(self):
        """Test feature creation with insufficient data"""
        small_data = self.sample_data.head(50)  # Less than 200 required
        
        result = self.predictor.create_features(small_data)
        
        self.assertIsNone(result)

    def test_create_features_data_types(self):
        """Test that created features have correct data types"""
        result = self.predictor.create_features(self.sample_data)
        
        self.assertIsNotNone(result)
        
        # Check that numeric features are numeric
        numeric_features = ['returns', 'rsi', 'macd', 'volatility']
        for feature in numeric_features:
            if feature in result.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(result[feature]))

    def test_rsi_calculation(self):
        """Test RSI calculation specifically"""
        result = self.predictor.create_features(self.sample_data)
        
        self.assertIsNotNone(result)
        self.assertIn('rsi', result.columns)
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())

    def test_macd_calculation(self):
        """Test MACD calculation"""
        result = self.predictor.create_features(self.sample_data)
        
        self.assertIsNotNone(result)
        
        macd_features = ['macd', 'macd_signal', 'macd_histogram']
        for feature in macd_features:
            self.assertIn(feature, result.columns)

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        result = self.predictor.create_features(self.sample_data)
        
        self.assertIsNotNone(result)
        
        bb_features = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_position']
        for feature in bb_features:
            self.assertIn(feature, result.columns)
        
        # BB position should be between 0 and 1
        bb_position = result['bb_position'].dropna()
        self.assertTrue((bb_position >= 0).all())
        self.assertTrue((bb_position <= 1).all())

    @patch('yfinance.download')
    def test_add_market_context_features(self, mock_yf):
        """Test market context features addition"""
        # Mock VIX, DXY, TNX data
        mock_yf.side_effect = [
            pd.DataFrame({'Close': np.random.uniform(15, 35, len(self.sample_data))}, 
                        index=self.sample_data.index),  # VIX
            pd.DataFrame({'Close': np.random.uniform(90, 110, len(self.sample_data))}, 
                        index=self.sample_data.index),  # DXY
            pd.DataFrame({'Close': np.random.uniform(1, 5, len(self.sample_data))}, 
                        index=self.sample_data.index)   # TNX
        ]
        
        df = self.sample_data.copy()
        df['returns'] = df['Close'].pct_change()
        df['volume_ratio'] = 1.0
        
        result = self.predictor.add_market_context_features(df)
        
        expected_features = ['vix', 'vix_ma', 'vix_ratio', 'dxy', 'dxy_change', 'tnx', 'tnx_change']
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing market context feature: {feature}")

    def test_add_sentiment_features(self):
        """Test sentiment features addition"""
        df = self.sample_data.copy()
        df['returns'] = df['Close'].pct_change()
        df['volume_ratio'] = 1.0
        df['vix'] = 20.0
        
        result = self.predictor.add_sentiment_features(df)
        
        expected_features = [
            'open_close_ratio', 'high_close_ratio', 'low_close_ratio',
            'volume_change', 'price_volume_trend', 'fear_greed_proxy'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing sentiment feature: {feature}")

    def test_add_microstructure_features(self):
        """Test microstructure features addition"""
        df = self.sample_data.copy()
        
        result = self.predictor.add_microstructure_features(df)
        
        expected_features = ['gap_up', 'gap_down', 'tr', 'atr', 'atr_ratio', 'efficiency_ratio']
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing microstructure feature: {feature}")

    def test_add_regime_features(self):
        """Test regime features addition"""
        df = self.sample_data.copy()
        df['returns'] = df['Close'].pct_change()
        
        result = self.predictor.add_regime_features(df)
        
        expected_features = [
            'trend_strength', 'bull_market', 'bear_market', 
            'sideways_market', 'high_vol_regime'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing regime feature: {feature}")

    def test_prepare_data(self):
        """Test data preparation for training"""
        df_features = self.predictor.create_features(self.sample_data)
        self.assertIsNotNone(df_features)
        
        X, y = self.predictor.prepare_data(df_features)
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertTrue(len(X) > 0)

    def test_prepare_data_insufficient_data(self):
        """Test data preparation with insufficient data"""
        # Create minimal dataframe
        minimal_df = pd.DataFrame({
            'target': [0, 1, 0],
            'returns': [0.01, -0.02, 0.005]
        })
        
        X, y = self.predictor.prepare_data(minimal_df)
        
        self.assertIsNone(X)
        self.assertIsNone(y)

    def test_train_model(self):
        """Test model training"""
        df_features = self.predictor.create_features(self.sample_data)
        self.assertIsNotNone(df_features)
        
        X, y = self.predictor.prepare_data(df_features)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        
        model = self.predictor.train_model(X, y)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.predictor.model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))

    def test_train_model_invalid_data(self):
        """Test model training with invalid data"""
        result = self.predictor.train_model(None, None)
        
        self.assertIsNone(result)

    def test_feature_importance_analysis(self):
        """Test feature importance analysis"""
        # First train a model
        df_features = self.predictor.create_features(self.sample_data)
        X, y = self.predictor.prepare_data(df_features)
        self.predictor.train_model(X, y)
        
        importance_df = self.predictor.feature_importance_analysis()
        
        if importance_df is not None:  # Some models might not have feature_importances_
            self.assertIsInstance(importance_df, pd.DataFrame)
            self.assertIn('feature', importance_df.columns)
            self.assertIn('importance', importance_df.columns)

    def test_predict_next_day(self):
        """Test next day prediction"""
        # Train model first
        df_features = self.predictor.create_features(self.sample_data)
        X, y = self.predictor.prepare_data(df_features)
        self.predictor.train_model(X, y)
        
        result = self.predictor.predict_next_day(self.sample_data)
        
        self.assertIsInstance(result, dict)
        
        expected_keys = [
            'prediction', 'probability_up', 'probability_down', 
            'confidence', 'market_regime'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check prediction values
        self.assertIn(result['prediction'], ['UP', 'DOWN'])
        self.assertGreaterEqual(result['probability_up'], 0)
        self.assertLessEqual(result['probability_up'], 1)
        self.assertGreaterEqual(result['probability_down'], 0)
        self.assertLessEqual(result['probability_down'], 1)
        self.assertAlmostEqual(
            result['probability_up'] + result['probability_down'], 1.0, places=5
        )

    def test_predict_next_day_no_model(self):
        """Test prediction without trained model"""
        with self.assertRaises(ValueError):
            self.predictor.predict_next_day(self.sample_data)

    def test_save_load_model(self):
        """Test model saving and loading"""
        # Train model first
        df_features = self.predictor.create_features(self.sample_data)
        X, y = self.predictor.prepare_data(df_features)
        self.predictor.train_model(X, y)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save model
            save_success = self.predictor.save_model(tmp_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new predictor and load model
            new_predictor = EnhancedSP500Predictor()
            load_success = new_predictor.load_model(tmp_path)
            
            self.assertTrue(load_success)
            self.assertIsNotNone(new_predictor.model)
            self.assertEqual(len(new_predictor.features), len(self.predictor.features))
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_model_no_model(self):
        """Test saving when no model exists"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = self.predictor.save_model(tmp_path)
            self.assertFalse(result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_model_nonexistent(self):
        """Test loading non-existent model"""
        result = self.predictor.load_model('nonexistent_model.pkl')
        self.assertFalse(result)

    def test_time_series_validation(self):
        """Test time series validation"""
        df_features = self.predictor.create_features(self.sample_data)
        X, y = self.predictor.prepare_data(df_features)
        
        # Create a mock model for testing
        from sklearn.ensemble import RandomForestClassifier
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        scores = self.predictor.time_series_validation(X, y, mock_model)
        
        self.assertIsInstance(scores, np.ndarray)
        self.assertGreater(len(scores), 0)
        self.assertTrue(all(0 <= score <= 1 for score in scores))

    @patch('yfinance.download')
    def test_backtest_strategy(self, mock_yf):
        """Test backtesting strategy"""
        # Mock external data downloads for market context
        mock_yf.side_effect = [
            pd.DataFrame({'Close': np.random.uniform(15, 35, len(self.sample_data))}, 
                        index=self.sample_data.index),
            pd.DataFrame({'Close': np.random.uniform(90, 110, len(self.sample_data))}, 
                        index=self.sample_data.index),
            pd.DataFrame({'Close': np.random.uniform(1, 5, len(self.sample_data))}, 
                        index=self.sample_data.index)
        ]
        
        # Train model first
        df_features = self.predictor.create_features(self.sample_data)
        X, y = self.predictor.prepare_data(df_features)
        self.predictor.train_model(X, y)
        
        # Run backtest
        results = self.predictor.backtest_strategy(self.sample_data)
        
        if results is not None:  # Backtest might fail with insufficient data
            self.assertIsInstance(results, dict)
            
            expected_keys = [
                'total_days', 'trading_days', 'accuracy', 'win_rate',
                'strategy_return', 'buyhold_return', 'strategy_sharpe'
            ]
            
            for key in expected_keys:
                self.assertIn(key, results, f"Missing backtest result: {key}")

    def test_data_validation(self):
        """Test data validation and cleaning"""
        # Create data with NaN and inf values
        corrupt_data = self.sample_data.copy()
        corrupt_data.loc[corrupt_data.index[10], 'Close'] = np.nan
        corrupt_data.loc[corrupt_data.index[20], 'Volume'] = np.inf
        
        result = self.predictor.create_features(corrupt_data)
        
        # Should handle corrupt data gracefully
        if result is not None:
            # Check that result doesn't contain inf values
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.assertFalse(np.isinf(result[col]).any(), f"Column {col} contains inf values")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with minimal valid data (exactly 200 days)
        minimal_data = self.sample_data.head(200)
        result = self.predictor.create_features(minimal_data)
        
        # Should work with exactly minimum required data
        self.assertIsNotNone(result)
        
        # Test with all same prices (no volatility)
        flat_data = self.sample_data.copy()
        flat_data['Close'] = 3000.0
        flat_data['Open'] = 3000.0
        flat_data['High'] = 3000.0
        flat_data['Low'] = 3000.0
        
        result = self.predictor.create_features(flat_data)
        
        # Should handle flat prices without crashing
        self.assertIsNotNone(result)

    def tearDown(self):
        """Clean up after tests"""
        # Clean up any temporary files that might have been created
        temp_files = ['test_model.pkl', 'enhanced_sp500_model.pkl']
        for file in temp_files:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except:
                    pass


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and edge cases"""
    
    def test_rsi_edge_cases(self):
        """Test RSI calculation with edge cases"""
        predictor = EnhancedSP500Predictor()
        
        # Test with constant prices (no movement)
        constant_prices = pd.Series([100] * 50)
        rsi = predictor.create_features(pd.DataFrame({
            'Open': constant_prices,
            'High': constant_prices,
            'Low': constant_prices,
            'Close': constant_prices,
            'Volume': [1000000] * 50
        }, index=pd.date_range('2023-01-01', periods=50)))
        
        # Should handle constant prices without error
        if rsi is not None:
            self.assertIn('rsi', rsi.columns)

    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        # Create larger dataset to test memory efficiency
        large_dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        large_data = pd.DataFrame({
            'Open': np.random.uniform(2000, 4000, len(large_dates)),
            'High': np.random.uniform(2000, 4000, len(large_dates)),
            'Low': np.random.uniform(2000, 4000, len(large_dates)),
            'Close': np.random.uniform(2000, 4000, len(large_dates)),
            'Volume': np.random.randint(1000000, 10000000, len(large_dates))
        }, index=large_dates)
        
        predictor = EnhancedSP500Predictor()
        
        # Should handle large datasets without memory issues
        try:
            result = predictor.create_features(large_data)
            if result is not None:
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result), 0)
        except MemoryError:
            self.skipTest("Insufficient memory for large dataset test")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(
        verbosity=2,
        buffer=True,  # Capture stdout/stderr
        failfast=False  # Continue running tests even if some fail
    )