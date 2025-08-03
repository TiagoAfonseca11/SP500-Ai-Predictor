from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timedelta
import traceback
import threading
import time
import pandas as pd

# Import your existing predictor
from sp500_predictor import EnhancedSP500Predictor

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = EnhancedSP500Predictor()
prediction_cache = {}
cache_timestamp = None

def load_html_template():
    """Load the HTML template"""
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to embedded template
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>S&P 500 Predictor</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>
            <h1>S&P 500 AI Predictor</h1>
            <p>Please check if templates/index.html exists</p>
        </body>
        </html>
        """

@app.route('/')
def index():
    """Serve the main page"""
    return load_html_template()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    global prediction_cache, cache_timestamp
    
    try:
        # Check cache (5-minute cache)
        now = datetime.now()
        if cache_timestamp and (now - cache_timestamp).seconds < 300 and prediction_cache:
            return jsonify({
                'success': True,
                'data': prediction_cache,
                'cached': True,
                'timestamp': cache_timestamp.isoformat()
            })
        
        print("Running new prediction...")
        
        # Download fresh data
        data = predictor.download_data(period="3y")
        if data is None or data.empty:
            return jsonify({
                'success': False,
                'error': 'Failed to download market data'
            }), 500
        
        # Load or train model
        model_loaded = False
        if os.path.exists('enhanced_sp500_model.pkl'):
            try:
                model_loaded = predictor.load_model()
                print(f"Model loaded: {model_loaded}")
            except Exception as e:
                print(f"Error loading model: {e}")
                model_loaded = False
        
        if not model_loaded:
            print("Training new model...")
            df_features = predictor.create_features(data)
            if df_features is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create features'
                }), 500
            
            X, y = predictor.prepare_data(df_features)
            if X is None or y is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to prepare data'
                }), 500
            
            model = predictor.train_model(X, y)
            if model is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to train model'
                }), 500
            
            predictor.save_model()
        
        # Make prediction
        result = predictor.predict_next_day(data)
        
        # Get additional context
        last_close = float(data['Close'].iloc[-1])
        last_volume = int(data['Volume'].iloc[-1])
        
        # Get feature importance
        importance_df = predictor.feature_importance_analysis()
        feature_importance = []
        if importance_df is not None:
            feature_importance = [
                {
                    'name': str(row['feature']),
                    'importance': float(row['importance'])
                }
                for _, row in importance_df.head(10).iterrows()
            ]
        
        # Prepare response with better error handling
        response_data = {
            'prediction': str(result.get('prediction', 'UNKNOWN')),
            'probability_up': float(result.get('probability_up', 0.5)),
            'probability_down': float(result.get('probability_down', 0.5)),
            'confidence': float(result.get('confidence', 0.5)),
            'market_regime': str(result.get('market_regime', 'Unknown')),
            'vix_level': float(result.get('vix_level', 0)) if result.get('vix_level') else None,
            'trend_strength': float(result.get('trend_strength', 0)) if result.get('trend_strength') is not None else None,
            'last_price': last_close,
            'last_volume': last_volume,
            'feature_importance': feature_importance,
            'model_features': len(predictor.features) if predictor.features else 0,
            'timestamp': now.isoformat()
        }
        
        # Cache the result
        prediction_cache = response_data
        cache_timestamp = now
        
        print(f"Prediction successful: {response_data['prediction']} with {response_data['confidence']:.2%} confidence")
        
        return jsonify({
            'success': True,
            'data': response_data,
            'cached': False
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """API endpoint for backtesting"""
    try:
        # Get parameters
        request_data = request.get_json() or {}
        days_back = request_data.get('days_back', 730)  # Default 2 years
        
        print(f"Running backtest for last {days_back} days...")
        
        # Download data
        market_data = predictor.download_data(period="5y")
        if market_data is None or market_data.empty:
            return jsonify({
                'success': False,
                'error': 'Failed to download market data'
            }), 500
        
        # Ensure model is loaded
        model_loaded = False
        if os.path.exists('enhanced_sp500_model.pkl'):
            try:
                model_loaded = predictor.load_model()
            except Exception as e:
                print(f"Error loading model for backtest: {e}")
        
        if not model_loaded:
            # Train model if not available
            print("Training model for backtest...")
            df_features = predictor.create_features(market_data)
            if df_features is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create features for training'
                }), 500
            
            X, y = predictor.prepare_data(df_features)
            if X is None or y is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to prepare data for training'
                }), 500
            
            model = predictor.train_model(X, y)
            if model is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to train model for backtest'
                }), 500
            
            predictor.save_model()
        
        # Run backtest
        start_date = datetime.now() - timedelta(days=days_back)
        backtest_results = predictor.backtest_strategy(
            market_data, 
            start_date=start_date
        )
        
        if backtest_results is None:
            return jsonify({
                'success': False,
                'error': 'Backtest failed to execute'
            }), 500
        
        # Convert all values to safe types for JSON
        safe_results = {}
        for key, value in backtest_results.items():
            if isinstance(value, (int, float)):
                if key in ['strategy_return', 'buyhold_return', 'excess_return', 
                          'strategy_mdd', 'buyhold_mdd', 'win_rate', 'accuracy']:
                    safe_results[key] = float(value)
                else:
                    safe_results[key] = value
            else:
                safe_results[key] = str(value)
        
        print(f"Backtest completed successfully. Strategy return: {safe_results.get('strategy_return', 0):.2%}")
        
        return jsonify({
            'success': True,
            'data': safe_results
        })
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Backtest failed: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint for training model"""
    try:
        print("Training new model...")
        
        # Download data
        data = predictor.download_data(period="5y")
        if data is None or data.empty:
            return jsonify({
                'success': False,
                'error': 'Failed to download market data'
            }), 500
        
        # Create features
        df_features = predictor.create_features(data)
        if df_features is None:
            return jsonify({
                'success': False,
                'error': 'Failed to create features'
            }), 500
        
        # Prepare data
        X, y = predictor.prepare_data(df_features)
        if X is None or y is None:
            return jsonify({
                'success': False,
                'error': 'Failed to prepare data'
            }), 500
        
        # Train model
        model = predictor.train_model(X, y)
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Failed to train model'
            }), 500
        
        # Save model
        predictor.save_model()
        
        # Clear cache to force new prediction
        global prediction_cache, cache_timestamp
        prediction_cache = {}
        cache_timestamp = None
        
        print(f"Model trained successfully with {len(predictor.features)} features and {len(X)} data points")
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'features_count': len(predictor.features),
            'data_points': len(X)
        })
        
    except Exception as e:
        print(f"Error in training: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """API endpoint for refreshing data"""
    try:
        print("Refreshing market data...")
        
        # Clear cache
        global prediction_cache, cache_timestamp
        prediction_cache = {}
        cache_timestamp = None
        
        # Test data download
        data = predictor.download_data(period="1y")
        if data is None or data.empty:
            return jsonify({
                'success': False,
                'error': 'Failed to download fresh market data'
            }), 500
        
        print(f"Data refreshed successfully: {len(data)} data points, latest: {data.index[-1]}")
        
        return jsonify({
            'success': True,
            'message': 'Data refreshed successfully',
            'data_points': len(data),
            'latest_date': data.index[-1].strftime('%Y-%m-%d'),
            'latest_price': float(data['Close'].iloc[-1])
        })
        
    except Exception as e:
        print(f"Error in refresh: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Data refresh failed: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint for system status"""
    try:
        model_exists = os.path.exists('enhanced_sp500_model.pkl')
        
        # Test data connection
        try:
            test_data = predictor.download_data(period="5d")
            data_connection = test_data is not None and not test_data.empty
            latest_price = float(test_data['Close'].iloc[-1]) if data_connection else None
        except:
            data_connection = False
            latest_price = None
        
        return jsonify({
            'success': True,
            'status': {
                'model_trained': model_exists,
                'data_connection': data_connection,
                'features_count': len(predictor.features) if predictor.features else 0,
                'cache_active': cache_timestamp is not None,
                'cache_age_minutes': (datetime.now() - cache_timestamp).seconds // 60 if cache_timestamp else None,
                'latest_price': latest_price,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"Error in status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """API endpoint for prediction history"""
    try:
        history_file = 'enhanced_daily_predictions.csv'
        
        if not os.path.exists(history_file):
            return jsonify({
                'success': True,
                'data': [],
                'message': 'No prediction history found'
            })
        
        df = pd.read_csv(history_file)
        
        # Get last 30 predictions
        recent_predictions = df.tail(30).to_dict('records')
        
        # Calculate accuracy for recent predictions
        if len(recent_predictions) > 1:
            total_predictions = len(recent_predictions)
            avg_confidence = sum(float(p.get('confianca', 0)) for p in recent_predictions) / total_predictions
        else:
            total_predictions = 0
            avg_confidence = 0
        
        return jsonify({
            'success': True,
            'data': recent_predictions,
            'summary': {
                'total_predictions': total_predictions,
                'avg_confidence': avg_confidence,
                'latest_date': recent_predictions[-1].get('data') if recent_predictions else None
            }
        })
        
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
# Add this endpoint in app.py, along with the other existing endpoints

@app.route('/api/chart', methods=['POST'])
def api_chart():
    """API endpoint for chart data"""
    try:
        # Get parameters
        request_data = request.get_json() or {}
        period = request_data.get('period', '3mo')
        
        print(f"Loading chart data for period: {period}")
        
        # Download data based on period
        period_mapping = {
            '1mo': '1mo',
            '3mo': '3mo', 
            '6mo': '6mo',
            '1y': '1y',
            '2y': '2y'
        }
        
        actual_period = period_mapping.get(period, '3mo')
        data = predictor.download_data(period=actual_period)
        
        if data is None or data.empty:
            return jsonify({
                'success': False,
                'error': 'Failed to download chart data'
            }), 500
        
        # Prepare chart data
        prices = data['Close'].tolist()
        dates = [date.strftime('%Y-%m-%d') for date in data.index]
        
        # Calculate statistics
        first_price = float(data['Close'].iloc[0])
        last_price = float(data['Close'].iloc[-1])
        period_return = (last_price - first_price) / first_price
        
        # Calculate volatility (annualized)
        returns = data['Close'].pct_change().dropna()
        volatility = float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else 0
        
        max_price = float(data['Close'].max())
        min_price = float(data['Close'].min())
        
        # Prepare response
        chart_data = {
            'prices': prices,
            'dates': dates,
            'stats': {
                'period_return': period_return,
                'volatility': volatility,
                'max_price': max_price,
                'min_price': min_price,
                'first_price': first_price,
                'last_price': last_price,
                'data_points': len(prices)
            },
            'period': period,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Chart data prepared: {len(prices)} points, return: {period_return:.2%}")
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
        
    except Exception as e:
        print(f"Error in chart endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Chart data failed: {str(e)}'
        }), 500

def run_background_tasks():
    """Background tasks runner"""
    while True:
        try:
            # Auto-refresh model daily at 6 AM
            now = datetime.now()
            if now.hour == 6 and now.minute < 10:
                print("Running daily model refresh...")
                try:
                    predictor.run_daily_prediction()
                except Exception as e:
                    print(f"Daily prediction error: {e}")
                time.sleep(600)  # Sleep 10 minutes to avoid multiple runs
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Background task error: {e}")
            time.sleep(300)  # Sleep 5 minutes on error

if __name__ == '__main__':
    print("🚀 Starting S&P 500 Predictor Server...")
    
    # Start background tasks in a separate thread
    background_thread = threading.Thread(target=run_background_tasks, daemon=True)
    background_thread.start()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)