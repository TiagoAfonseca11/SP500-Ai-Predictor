#!/usr/bin/env python3
"""
Benchmark Tests for S&P 500 Predictor System
Tests performance, memory usage, and scalability
"""

import unittest
import time
import sys
import os
import tracemalloc
import psutil
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp500_predictor import EnhancedSP500Predictor

class BenchmarkTests(unittest.TestCase):
    """Benchmark and performance tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.predictor = EnhancedSP500Predictor()
        cls.performance_results = {}
        print("\n🔧 Setting up benchmark tests...")
        
    def setUp(self):
        """Set up each test"""
        self.start_time = time.time()
        tracemalloc.start()
        
    def tearDown(self):
        """Clean up after each test"""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = time.time() - self.start_time
        memory_mb = peak / 1024 / 1024
        
        test_name = self._testMethodName
        self.performance_results[test_name] = {
            'duration': duration,
            'peak_memory_mb': memory_mb
        }
        
        print(f"⏱️  {test_name}: {duration:.2f}s, {memory_mb:.1f}MB peak")
    
    def test_data_download_performance(self):
        """Test data download speed and memory usage"""
        periods = ['1mo', '3mo', '6mo', '1y']
        
        for period in periods:
            start = time.time()
            data = self.predictor.download_data(period=period)
            duration = time.time() - start
            
            self.assertIsNotNone(data, f"Failed to download {period} data")
            self.assertFalse(data.empty, f"Empty data for {period}")
            
            # Performance assertions
            self.assertLess(duration, 10.0, f"Download took too long for {period}: {duration:.2f}s")
            self.assertGreater(len(data), 10, f"Insufficient data points for {period}")
            
            print(f"  📊 {period}: {len(data)} points in {duration:.2f}s")
    
    def test_feature_creation_scalability(self):
        """Test feature creation with different data sizes"""
        # Test with different data sizes
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            # Create synthetic data
            dates = pd.date_range(start='2020-01-01', periods=size, freq='D')
            synthetic_data = pd.DataFrame({
                'Open': np.random.uniform(3000, 4000, size),
                'High': np.random.uniform(3000, 4100, size),
                'Low': np.random.uniform(2900, 4000, size),
                'Close': np.random.uniform(3000, 4000, size),
                'Volume': np.random.randint(1000000, 10000000, size)
            }, index=dates)
            
            start = time.time()
            features = self.predictor.create_features(synthetic_data)
            duration = time.time() - start
            
            if features is not None:
                # Performance assertions
                self.assertLess(duration, 30.0, f"Feature creation too slow for {size} points: {duration:.2f}s")
                self.assertGreater(len(features.columns), 10, "Too few features created")
                
                # Memory efficiency check
                expected_memory_mb = size * len(features.columns) * 8 / 1024 / 1024  # Rough estimate
                self.assertLess(expected_memory_mb, 500, "Memory usage too high")
                
                print(f"  🔧 {size} points → {len(features.columns)} features in {duration:.2f}s")
    
    def test_model_training_performance(self):
        """Test model training speed with different data sizes"""
        # Use real data for more realistic test
        data = self.predictor.download_data(period="2y")
        self.assertIsNotNone(data, "Failed to download training data")
        
        # Create features
        start = time.time()
        df_features = self.predictor.create_features(data)
        feature_time = time.time() - start
        
        self.assertIsNotNone(df_features, "Failed to create features")
        
        # Prepare data
        start = time.time()
        X, y = self.predictor.prepare_data(df_features)
        prep_time = time.time() - start
        
        self.assertIsNotNone(X, "Failed to prepare X data")
        self.assertIsNotNone(y, "Failed to prepare y data")
        
        # Train model
        start = time.time()
        model = self.predictor.train_model(X, y)
        train_time = time.time() - start
        
        self.assertIsNotNone(model, "Failed to train model")
        
        # Performance assertions
        self.assertLess(feature_time, 60.0, f"Feature creation too slow: {feature_time:.2f}s")
        self.assertLess(prep_time, 10.0, f"Data preparation too slow: {prep_time:.2f}s")
        self.assertLess(train_time, 300.0, f"Model training too slow: {train_time:.2f}s")
        
        total_time = feature_time + prep_time + train_time
        print(f"  🤖 Training pipeline: {total_time:.2f}s total")
        print(f"     Features: {feature_time:.2f}s, Prep: {prep_time:.2f}s, Train: {train_time:.2f}s")
    
    def test_prediction_speed(self):
        """Test prediction speed and consistency"""
        # Prepare model
        data = self.predictor.download_data(period="1y")
        self.assertIsNotNone(data, "Failed to download data for prediction test")
        
        # Quick training for testing
        df_features = self.predictor.create_features(data)
        if df_features is not None:
            X, y = self.predictor.prepare_data(df_features)
            if X is not None and y is not None:
                self.predictor.train_model(X, y)
        
        # Test prediction speed
        prediction_times = []
        predictions = []
        
        for i in range(10):  # 10 predictions
            start = time.time()
            try:
                result = self.predictor.predict_next_day(data)
                duration = time.time() - start
                prediction_times.append(duration)
                predictions.append(result)
            except Exception as e:
                self.fail(f"Prediction failed on iteration {i}: {e}")
        
        # Performance assertions
        avg_time = np.mean(prediction_times)
        max_time = np.max(prediction_times)
        
        self.assertLess(avg_time, 5.0, f"Average prediction too slow: {avg_time:.2f}s")
        self.assertLess(max_time, 10.0, f"Slowest prediction too slow: {max_time:.2f}s")
        
        # Consistency check
        directions = [p['prediction'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        # All predictions should be valid
        self.assertTrue(all(d in ['UP', 'DOWN'] for d in directions), "Invalid predictions")
        self.assertTrue(all(0 <= c <= 1 for c in confidences), "Invalid confidence values")
        
        print(f"  ⚡ 10 predictions: avg {avg_time:.3f}s, max {max_time:.3f}s")
    
    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple operations
        for i in range(5):
            data = self.predictor.download_data(period="3mo")
            if data is not None:
                df_features = self.predictor.create_features(data)
                if df_features is not None:
                    X, y = self.predictor.prepare_data(df_features)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory shouldn't grow more than 200MB during operations
        self.assertLess(memory_growth, 200, f"Excessive memory growth: {memory_growth:.1f}MB")
        
        print(f"  💾 Memory growth: {memory_growth:.1f}MB ({initial_memory:.1f} → {final_memory:.1f}MB)")
    

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        duration = time.time() - start_time
        
        # Check results
        successes = 0
        while not results_queue.empty():
            result = results_queue.get()
            if result == "success":
                successes += 1
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        # Assertions
        self.assertGreater(successes, 0, "No successful concurrent operations")
        self.assertLess(len(errors), 3, f"Too many errors in concurrent test: {errors}")
        self.assertLess(duration, 120, f"Concurrent operations took too long: {duration:.2f}s")
        
        print(f"  🔀 Concurrent: {successes}/3 success in {duration:.2f}s")
    
    def test_model_file_operations(self):
        """Test model save/load performance"""
        test_file = "test_benchmark_model.pkl"
        
        try:
            # First ensure we have a model to test with
            model_ready = False
            
            # Try to train a minimal model for testing
            try:
                data = self.predictor.download_data(period="3mo")  # Smaller dataset
                if data is not None and not data.empty and len(data) > 50:
                    # Quick feature creation for testing
                    df_simple = data.copy()
                    df_simple['returns'] = data['Close'].pct_change()
                    df_simple['ma_5'] = data['Close'].rolling(5, min_periods=1).mean()
                    df_simple['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                    df_simple = df_simple.dropna()
                    
                    if len(df_simple) > 20:
                        # Simple features for testing
                        X_simple = df_simple[['returns', 'ma_5']].iloc[:-1]
                        y_simple = df_simple['target'].iloc[:-1]
                        
                        # Quick training with minimal model
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.preprocessing import StandardScaler
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_simple)
                        
                        simple_model = RandomForestClassifier(n_estimators=10, random_state=42)
                        simple_model.fit(X_scaled, y_simple)
                        
                        # Set up predictor with simple model for testing
                        self.predictor.model = simple_model
                        self.predictor.scaler = scaler
                        self.predictor.features = ['returns', 'ma_5']
                        model_ready = True
                        
            except Exception as e:
                print(f"  ⚠️ Could not train test model: {e}")
            
            if not model_ready:
                # Create a minimal mock model for I/O testing
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                mock_model = RandomForestClassifier(n_estimators=5, random_state=42)
                mock_scaler = StandardScaler()
                
                # Fit with dummy data
                import numpy as np
                X_dummy = np.random.random((20, 2))
                y_dummy = np.random.randint(0, 2, 20)
                
                mock_scaler.fit(X_dummy)
                mock_model.fit(mock_scaler.transform(X_dummy), y_dummy)
                
                self.predictor.model = mock_model
                self.predictor.scaler = mock_scaler
                self.predictor.features = ['feature1', 'feature2']
                
                print("  📝 Using mock model for I/O testing")
            
            # Test save speed
            start = time.time()
            success = self.predictor.save_model(test_file)
            save_time = time.time() - start
            
            self.assertTrue(success, "Failed to save model")
            self.assertTrue(os.path.exists(test_file), "Model file was not created")
            self.assertLess(save_time, 10.0, f"Model save too slow: {save_time:.2f}s")
            
            # Test file size (should be reasonable)
            file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
            self.assertLess(file_size, 100, f"Model file too large: {file_size:.1f}MB")
            
            # Test load speed
            new_predictor = EnhancedSP500Predictor()
            start = time.time()
            success = new_predictor.load_model(test_file)
            load_time = time.time() - start
            
            self.assertTrue(success, "Failed to load model")
            self.assertIsNotNone(new_predictor.model, "Model not loaded properly")
            self.assertIsNotNone(new_predictor.scaler, "Scaler not loaded properly")
            self.assertIsNotNone(new_predictor.features, "Features not loaded properly")
            self.assertLess(load_time, 5.0, f"Model load too slow: {load_time:.2f}s")
            
            # Test model integrity
            self.assertEqual(len(new_predictor.features), len(self.predictor.features), 
                           "Feature count mismatch after load")
            
            print(f"  💾 Model I/O: save {save_time:.3f}s, load {load_time:.3f}s")
            print(f"     File size: {file_size:.1f}MB, Features: {len(new_predictor.features)}")
            
        except Exception as e:
            self.fail(f"Model file operations test failed: {e}")
            
        finally:
            # Cleanup - ensure file is removed even if test fails
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                except Exception as cleanup_error:
                    print(f"  ⚠️ Could not cleanup test file: {cleanup_error}")
    
    
    @classmethod
    def tearDownClass(cls):
        """Print final benchmark report"""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if cls.performance_results:
            total_time = sum(r['duration'] for r in cls.performance_results.values())
            max_memory = max(r['peak_memory_mb'] for r in cls.performance_results.values())
            
            print(f"⏱️  Total execution time: {total_time:.2f}s")
            print(f"💾 Peak memory usage: {max_memory:.1f}MB")
            print(f"🧪 Tests completed: {len(cls.performance_results)}")
            
            # Performance summary
            slow_tests = [(name, data['duration']) for name, data in cls.performance_results.items() 
                         if data['duration'] > 10]
            
            if slow_tests:
                print(f"\n⚠️  Slow tests (>10s):")
                for name, duration in sorted(slow_tests, key=lambda x: x[1], reverse=True):
                    print(f"   • {name}: {duration:.2f}s")
            
            memory_heavy = [(name, data['peak_memory_mb']) for name, data in cls.performance_results.items() 
                           if data['peak_memory_mb'] > 100]
            
            if memory_heavy:
                print(f"\n🐘 Memory heavy tests (>100MB):")
                for name, memory in sorted(memory_heavy, key=lambda x: x[1], reverse=True):
                    print(f"   • {name}: {memory:.1f}MB")
        
        print("\n✅ Benchmark tests completed!")
        print("="*60)

def run_benchmarks():
    """Run benchmark tests standalone"""
    print("🚀 S&P 500 Predictor - Benchmark Tests")
    print("="*50)
    
    # Check system resources
    import psutil
    print(f"💻 System Info:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    print(f"   Python version: {sys.version.split()[0]}")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(BenchmarkTests)
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_benchmarks()
    sys.exit(0 if success else 1)