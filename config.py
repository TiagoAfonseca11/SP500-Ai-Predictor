"""
Configuration file for S&P 500 Predictor Web Application
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'sp500-predictor-secret-key-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server Configuration
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Prediction Configuration
    CACHE_TIMEOUT = timedelta(minutes=5)  # Cache predictions for 5 minutes
    AUTO_REFRESH_HOURS = [6, 12, 18]  # Auto refresh at these hours
    
    # Data Configuration
    DEFAULT_PERIOD = "3y"  # Default data period for predictions
    BACKTEST_PERIOD = "2y"  # Default backtest period
    MIN_DATA_POINTS = 100  # Minimum data points required
    
    # Model Configuration
    MODEL_FILE = "enhanced_sp500_model.pkl"
    PREDICTIONS_FILE = "enhanced_daily_predictions.csv"
    
    # API Configuration
    API_RATE_LIMIT = "100/hour"  # Rate limiting for API endpoints
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5000"]
    
    # Feature Configuration
    MAX_FEATURES_DISPLAY = 10  # Max features to show in importance chart
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for trading signals
    
    # UI Configuration
    THEME_COLORS = {
        'primary': '#1a1d29',
        'secondary': '#2d3748',
        'accent': '#00d4aa',
        'success': '#48bb78',
        'warning': '#ed8936',
        'error': '#f56565'
    }
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = 'sp500_predictor.log'
    
    # Market Data Configuration
    MARKET_SYMBOLS = {
        'SP500': '^GSPC',
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB',
        'TNX': '^TNX'
    }
    
    # Alert Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    LOW_CONFIDENCE_THRESHOLD = 0.50
    
    # Backtest Configuration
    BACKTEST_CONFIDENCE_THRESHOLD = 0.6
    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Security settings for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MODEL_FILE = "test_model.pkl"
    PREDICTIONS_FILE = "test_predictions.csv"

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}