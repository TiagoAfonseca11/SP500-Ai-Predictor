import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime, timedelta
import warnings
import csv
import os
warnings.filterwarnings('ignore')

class EnhancedSP500Predictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        self.symbol = "^GSPC"  # S&P 500 ticker
        self.results_df = None
        
    def download_data(self, period="5y"):
        """Download dados históricos do S&P 500"""
        try:
            print(f"Baixando dados do S&P 500 ({period})...")
            data = yf.download(self.symbol, period=period, interval="1d")
            
            if data.empty:
                print("Erro: Nenhum dado foi baixado")
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"Erro: Colunas ausentes: {missing_cols}")
                return None
            
            print(f"Dados baixados: {len(data)} dias")
            return data
            
        except Exception as e:
            print(f"Erro ao baixar dados: {e}")
            return None
    
    def add_market_context_features(self, df):
        """Adicionar contexto de mercado mais amplo"""
        try:
            print("Adicionando features de contexto de mercado...")
            
            # VIX - Índice de volatilidade
            try:
                vix = yf.download("^VIX", start=df.index[0], end=df.index[-1], progress=False)['Close']
                df['vix'] = vix.reindex(df.index, method='ffill')
                df['vix_ma'] = df['vix'].rolling(20, min_periods=1).mean()
                df['vix_ratio'] = df['vix'] / df['vix_ma']
                df['vix_ratio'] = df['vix_ratio'].fillna(1.0)
                print("✓ VIX features adicionadas")
            except:
                print("⚠ VIX não disponível, usando volatilidade local")
                df['vix'] = df['Close'].rolling(20).std() * np.sqrt(252) * 100
                df['vix_ma'] = df['vix'].rolling(20, min_periods=1).mean()
                df['vix_ratio'] = df['vix'] / df['vix_ma']
                df['vix_ratio'] = df['vix_ratio'].fillna(1.0)
            
            # Dollar Index (DXY) - proxy usando USD/EUR se não disponível
            try:
                dxy = yf.download("DX-Y.NYB", start=df.index[0], end=df.index[-1], progress=False)['Close']
                df['dxy'] = dxy.reindex(df.index, method='ffill')
                df['dxy_change'] = df['dxy'].pct_change().fillna(0)
                print("✓ DXY features adicionadas")
            except:
                print("⚠ DXY não disponível, usando proxy")
                df['dxy'] = 100  # Valor neutro
                df['dxy_change'] = 0
            
            # Treasury yields (^TNX)
            try:
                tnx = yf.download("^TNX", start=df.index[0], end=df.index[-1], progress=False)['Close']
                df['tnx'] = tnx.reindex(df.index, method='ffill')
                df['tnx_change'] = df['tnx'].pct_change().fillna(0)
                print("✓ Treasury yields features adicionadas")
            except:
                print("⚠ TNX não disponível, usando proxy")
                df['tnx'] = 2.0  # Valor neutro
                df['tnx_change'] = 0
                
        except Exception as e:
            print(f"Erro ao adicionar features de mercado: {e}")
            
        return df
    
    def add_sentiment_features(self, df):
        """Features baseadas em sentiment"""
        try:
            print("Adicionando features de sentiment...")
            
            # Intraday sentiment
            df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']
            df['high_close_ratio'] = (df['High'] - df['Close']) / df['Close']
            df['low_close_ratio'] = (df['Close'] - df['Low']) / df['Close']
            
            # Volume-based sentiment
            df['volume_change'] = df['Volume'].pct_change().fillna(0)
            df['price_volume_trend'] = df['returns'] * df['volume_change']
            
            # Fear & Greed proxy
            returns_ma = df['returns'].rolling(10, min_periods=1).mean()
            vix_inv = 1 / (df['vix'] + 1) if 'vix' in df.columns else 0.5
            volume_component = df['volume_ratio'] if 'volume_ratio' in df.columns else 1.0
            
            df['fear_greed_proxy'] = (
                returns_ma * 0.3 +
                vix_inv * 0.3 +
                volume_component * 0.4
            )
            
            # Preencher NaN
            sentiment_cols = ['open_close_ratio', 'high_close_ratio', 'low_close_ratio', 
                            'price_volume_trend', 'fear_greed_proxy']
            for col in sentiment_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            print("✓ Features de sentiment adicionadas")
            
        except Exception as e:
            print(f"Erro ao adicionar features de sentiment: {e}")
            
        return df
    
    def add_microstructure_features(self, df):
        """Features de microestrutura do mercado"""
        try:
            print("Adicionando features de microestrutura...")
            
            # Gaps
            prev_close = df['Close'].shift(1)
            df['gap_up'] = ((df['Open'] - prev_close) / prev_close).clip(lower=0).fillna(0)
            df['gap_down'] = ((prev_close - df['Open']) / prev_close).clip(lower=0).fillna(0)
            
            # True Range e Average True Range
            high_low = df['High'] - df['Low']
            high_prev_close = abs(df['High'] - prev_close)
            low_prev_close = abs(df['Low'] - prev_close)
            
            df['tr'] = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
            df['atr'] = df['tr'].rolling(14, min_periods=1).mean()
            df['atr_ratio'] = df['tr'] / df['atr']
            df['atr_ratio'] = df['atr_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
            
            # Price efficiency
            close_change_10 = abs(df['Close'] - df['Close'].shift(10))
            tr_sum_10 = df['tr'].rolling(10, min_periods=1).sum()
            df['efficiency_ratio'] = close_change_10 / tr_sum_10
            df['efficiency_ratio'] = df['efficiency_ratio'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
            
            print("✓ Features de microestrutura adicionadas")
            
        except Exception as e:
            print(f"Erro ao adicionar features de microestrutura: {e}")
            
        return df
    
    def add_regime_features(self, df):
        """Detectar regimes de mercado"""
        try:
            print("Adicionando features de regime de mercado...")
            
            # Trend strength
            def trend_strength(prices):
                if len(prices) < 2:
                    return 0
                x = np.arange(len(prices))
                try:
                    correlation = np.corrcoef(x, prices)[0, 1]
                    return correlation if not np.isnan(correlation) else 0
                except:
                    return 0
            
            df['trend_strength'] = df['Close'].rolling(20, min_periods=2).apply(trend_strength)
            df['trend_strength'] = df['trend_strength'].fillna(0)
            
            # Market regime classification
            sma_50 = df['Close'].rolling(50, min_periods=1).mean()
            sma_200 = df['Close'].rolling(200, min_periods=1).mean()
            
            df['bull_market'] = ((df['Close'] > sma_50) & (sma_50 > sma_200)).astype(int)
            df['bear_market'] = ((df['Close'] < sma_50) & (sma_50 < sma_200)).astype(int)
            df['sideways_market'] = (~((df['Close'] > sma_50) & (sma_50 > sma_200)) & 
                                   ~((df['Close'] < sma_50) & (sma_50 < sma_200))).astype(int)
            
            # Volatility regime
            vol_20 = df['returns'].rolling(20, min_periods=1).std()
            vol_60 = df['returns'].rolling(60, min_periods=1).std()
            df['high_vol_regime'] = (vol_20 > vol_60 * 1.5).astype(int)
            df['high_vol_regime'] = df['high_vol_regime'].fillna(0)
            
            print("✓ Features de regime de mercado adicionadas")
            
        except Exception as e:
            print(f"Erro ao adicionar features de regime: {e}")
            
        return df
    
    def create_features(self, data):
        """Criar features técnicas avançadas para o modelo"""
        df = data.copy()
        
        if len(df) < 200:
            print("Erro: Dados insuficientes para criar features avançadas")
            return None
        
        try:
            print("Criando features básicas...")
            
            # Features básicas
            df['returns'] = df['Close'].pct_change()
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            
            # Volume features
            df['volume_ma'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # Médias móveis básicas
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
                df[f'ma_{window}_ratio'] = df['Close'] / df[f'ma_{window}']
                df[f'ma_{window}_ratio'] = df[f'ma_{window}_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # RSI
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = loss.replace(0, 1e-10)
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50)
            
            df['rsi'] = calculate_rsi(df['Close'])
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            bb_range = df['bb_upper'] - df['bb_lower']
            bb_range = bb_range.replace(0, 1e-10)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / bb_range
            df['bb_position'] = df['bb_position'].clip(0, 1)
            
            # Volatilidade
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
            df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
            
            # Features de momentum
            for window in [3, 7, 14]:
                df[f'momentum_{window}'] = df['Close'].pct_change(window).fillna(0)
            
            print("✓ Features básicas criadas")
            
            # Adicionar features avançadas
            df = self.add_market_context_features(df)
            df = self.add_sentiment_features(df)
            df = self.add_microstructure_features(df)
            df = self.add_regime_features(df)
            
            # Variável target
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Limpar dados
            df = df.dropna()
            
            if len(df) < 100:
                print("Erro: Dados insuficientes após limpeza")
                return None
            
            print(f"✅ Features criadas: {len(df)} amostras com {len(df.columns)} colunas")
            return df
            
        except Exception as e:
            print(f"Erro ao criar features: {e}")
            return None
    
    def prepare_data(self, df):
        """Preparar dados para treinamento"""
        if df is None:
            return None, None
        
        # Lista expandida de features
        feature_cols = [
            # Features básicas
            'returns', 'high_low_pct', 'volume_ratio',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility',
            'momentum_3', 'momentum_7', 'momentum_14',
            
            # Features de mercado
            'vix_ratio', 'dxy_change', 'tnx_change',
            
            # Features de sentiment
            'open_close_ratio', 'high_close_ratio', 'low_close_ratio',
            'price_volume_trend', 'fear_greed_proxy',
            
            # Features de microestrutura
            'gap_up', 'gap_down', 'atr_ratio', 'efficiency_ratio',
            
            # Features de regime
            'trend_strength', 'bull_market', 'bear_market', 'sideways_market', 'high_vol_regime'
        ]
        
        # Filtrar apenas features que existem
        available_features = [col for col in feature_cols if col in df.columns]
        self.features = available_features
        
        print(f"Features disponíveis: {len(available_features)}")
        
        X = df[available_features]
        y = df['target']
        
        # Remover última linha (não tem target)
        X = X[:-1]
        y = y[:-1]
        
        if len(X) < 50:
            print("Erro: Dados insuficientes para treinamento")
            return None, None
        
        return X, y
    
    def time_series_validation(self, X, y, model):
        """Validação respeitando série temporal"""
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                scores.append(score)
            
            return np.array(scores)
        except Exception as e:
            print(f"Erro na validação temporal: {e}")
            return np.array([0.5])
    
    def train_model(self, X, y):
        """Treinar modelo avançado com ensemble"""
        if X is None or y is None:
            print("Erro: Dados inválidos para treinamento")
            return None
        
        print(f"Treinando modelo avançado com {len(X)} amostras e {len(self.features)} features...")
        
        try:
            

            # Dividir dados
            test_size = min(0.2, max(0.1, 50 / len(X)))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Corrigir valores inválidos
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Modelos individuais com hiperparâmetros otimizados
            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                random_state=42
            )
            
            # Ensemble voting
            ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model)
                ],
                voting='soft'
            )
            
            # Treinar modelos
            print("Treinando Random Forest...")
            rf_model.fit(X_train_scaled, y_train)
            
            print("Treinando Gradient Boosting...")
            gb_model.fit(X_train_scaled, y_train)
            
            print("Treinando Ensemble...")
            ensemble_model.fit(X_train_scaled, y_train)
            
            # Avaliar modelos
            rf_score = accuracy_score(y_test, rf_model.predict(X_test_scaled))
            gb_score = accuracy_score(y_test, gb_model.predict(X_test_scaled))
            ensemble_score = accuracy_score(y_test, ensemble_model.predict(X_test_scaled))
            
            print(f"Random Forest Accuracy: {rf_score:.4f}")
            print(f"Gradient Boosting Accuracy: {gb_score:.4f}")
            print(f"Ensemble Accuracy: {ensemble_score:.4f}")
            
            # Escolher melhor modelo
            if ensemble_score >= max(rf_score, gb_score):
                self.model = ensemble_model
                print("✅ Modelo escolhido: Ensemble")
                best_score = ensemble_score
            elif rf_score > gb_score:
                self.model = rf_model
                print("✅ Modelo escolhido: Random Forest")
                best_score = rf_score
            else:
                self.model = gb_model
                print("✅ Modelo escolhido: Gradient Boosting")
                best_score = gb_score
            
            # Validação temporal
            if len(X_train) >= 100:
                print("Executando validação temporal...")
                cv_scores = self.time_series_validation(X_train, y_train, self.model)
                print(f"Time Series CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Relatório detalhado
            y_pred = self.model.predict(X_test_scaled)
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            return self.model
            
        except Exception as e:
            print(f"Erro durante treinamento: {e}")
            return None
    
    def feature_importance_analysis(self):
        """Analisar importância das features"""
        if self.model is None:
            return None
        
        try:
            # Para ensemble, usar o primeiro estimador como referência
            if hasattr(self.model, 'estimators_'):
                base_model = self.model.estimators_[0]
            else:
                base_model = self.model
            
            if hasattr(base_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.features,
                    'importance': base_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
        except Exception as e:
            print(f"Erro na análise de importância: {e}")
        
        return None
    
    def predict_next_day(self, data):
        """Fazer previsão avançada para o próximo dia"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        try:
            # Preparar dados mais recentes
            df_features = self.create_features(data)
            if df_features is None:
                raise ValueError("Erro ao criar features para previsão")
            
            # Pegar últimas features
            latest_features = df_features[self.features].iloc[-1:].values
            
            # Verificar e corrigir valores inválidos
            if np.any(np.isnan(latest_features)) or np.any(np.isinf(latest_features)):
                print("⚠ Valores inválidos detectados, corrigindo...")
                latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalizar
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Fazer previsão
            prediction = self.model.predict(latest_features_scaled)[0]
            probability = self.model.predict_proba(latest_features_scaled)[0]
            
            # Probabilidades
            prob_down = probability[0]
            prob_up = probability[1]
            confidence = max(prob_up, prob_down)
            
            # Análise adicional de regime
            regime_info = ""
            if 'bull_market' in df_features.columns:
                if df_features['bull_market'].iloc[-1] == 1:
                    regime_info = "Bull Market"
                elif df_features['bear_market'].iloc[-1] == 1:
                    regime_info = "Bear Market"
                else:
                    regime_info = "Sideways Market"
            
            return {
                'prediction': 'SUBIR' if prediction == 1 else 'DESCER',
                'probability_up': prob_up,
                'probability_down': prob_down,
                'confidence': confidence,
                'market_regime': regime_info,
                'vix_level': df_features['vix'].iloc[-1] if 'vix' in df_features.columns else None,
                'trend_strength': df_features['trend_strength'].iloc[-1] if 'trend_strength' in df_features.columns else None
            }
            
        except Exception as e:
            print(f"Erro na previsão: {e}")
            raise
    
    def generate_enhanced_report(self):
        """Relatório com análises avançadas"""
        print("\n🚀 RELATÓRIO AVANÇADO SP500 PREDICTOR")
        print("=" * 60)
        
        # Feature importance
        importance = self.feature_importance_analysis()
        if importance is not None:
            print("\n📊 TOP 10 FEATURES MAIS IMPORTANTES:")
            for _, row in importance.head(10).iterrows():
                print(f"   • {row['feature']}: {row['importance']:.4f}")
        
        # Estatísticas do modelo
        if hasattr(self.model, 'estimators_'):
            print(f"\n🤖 MODELO: Ensemble com {len(self.model.estimators_)} estimadores")
        else:
            print(f"\n🤖 MODELO: {type(self.model).__name__}")
        
        print(f"📈 FEATURES TOTAIS: {len(self.features)}")
    
    def save_model(self, filepath='enhanced_sp500_model.pkl'):
        """Salvar modelo treinado"""
        if self.model is None:
            print("Erro: Nenhum modelo para salvar")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'model_type': 'enhanced'
            }
            joblib.dump(model_data, filepath)
            print(f"✅ Modelo avançado salvo em: {filepath}")
            return True
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
            return False
    
    def load_model(self, filepath='enhanced_sp500_model.pkl'):
        """Carregar modelo treinado"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            print(f"✅ Modelo avançado carregado de: {filepath}")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def run_daily_prediction(self):
        """Executar previsão diária avançada"""
        print("🎯 === PREVISÃO DIÁRIA S&P 500 AVANÇADA ===")
        print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Baixar dados mais recentes
        data = self.download_data(period="3y")
        if data is None:
            print("❌ Erro: Não foi possível baixar dados")
            return
        
        # Tentar carregar modelo existente
        if not self.load_model():
            print("🔄 Treinando novo modelo avançado...")
            df_features = self.create_features(data)
            if df_features is None:
                print("❌ Erro: Não foi possível criar features")
                return
            
            X, y = self.prepare_data(df_features)
            if X is None or y is None:
                print("❌ Erro: Não foi possível preparar dados")
                return
            
            model = self.train_model(X, y)
            if model is None:
                print("❌ Erro: Não foi possível treinar modelo")
                return
            
            self.save_model()
        
        # Fazer previsão
        try:
            result = self.predict_next_day(data)
            
            print(f"\n🎯 PREVISÃO PARA AMANHÃ:")
            print(f"📊 Direção: {result['prediction']}")
            print(f"📈 Prob. SUBIR: {result['probability_up']:.1%}")
            print(f"📉 Prob. DESCER: {result['probability_down']:.1%}")
            print(f"🎯 Confiança: {result['confidence']:.1%}")
            
            if result['market_regime']:
                print(f"🏛️  Regime: {result['market_regime']}")
            
            if result['vix_level']:
                print(f"😰 VIX Level: {result['vix_level']:.1f}")
            
            if result['trend_strength'] is not None:
                trend = "Forte" if abs(result['trend_strength']) > 0.7 else "Moderada" if abs(result['trend_strength']) > 0.3 else "Fraca"
                direction = "Alta" if result['trend_strength'] > 0 else "Baixa"
                print(f"📈 Tendência: {trend} {direction}")
            
            # Info adicional
            last_close = data['Close'].iloc[-1]
            last_volume = data['Volume'].iloc[-1]
            print(f"\n💰 Último fechamento: ${last_close:.2f}")
            print(f"📊 Market Cap: {last_volume:,.0f}")
            
            # Interpretar resultado
            if result['confidence'] > 0.65:
                confidence_level = "🟢 ALTA"
            elif result['confidence'] > 0.58:
                confidence_level = "🟡 MÉDIA"
            else:
                confidence_level = "🔴 BAIXA"
            
            print(f"🎯 Nível de confiança: {confidence_level}")
            
            # Gerar relatório avançado
            self.generate_enhanced_report()
            
            # Salvar previsão
            prediction_data = {
                "data": datetime.now().strftime("%Y-%m-%d"),
                "direcao": result['prediction'],
                "prob_subir": round(result['probability_up'], 4),
                "prob_descer": round(result['probability_down'], 4),
                "confianca": round(result['confidence'], 4),
                "preco_fechamento": round(last_close, 2),
                "Market Cap": int(last_volume),
                "regime_mercado": result.get('market_regime', ''),
                "vix_level": round(result['vix_level'], 2) if result['vix_level'] else None,
                "trend_strength": round(result['trend_strength'], 4) if result['trend_strength'] is not None else None,
                "num_features": len(self.features),
                "modelo_tipo": "Enhanced"
            }

            csv_file = "enhanced_daily_predictions.csv"
            file_exists = os.path.isfile(csv_file)

            try:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=prediction_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(prediction_data)
                print(f"\n✅ Previsão avançada guardada em {csv_file}")
            except Exception as e:
                print(f"❌ Erro ao guardar previsão: {e}")
                
        except Exception as e:
            print(f"❌ Erro na previsão: {e}")
    
    def backtest_strategy(self, data, start_date=None, end_date=None):
        """Backtest da estratégia com métricas avançadas"""
        if self.model is None:
            print("❌ Modelo não treinado")
            return None
        
        try:
            print("🔄 Executando backtest...")
            
            # Preparar dados para backtest
            df_features = self.create_features(data)
            if df_features is None:
                return None
            
            # Filtrar período se especificado
            if start_date:
                df_features = df_features[df_features.index >= start_date]
            if end_date:
                df_features = df_features[df_features.index <= end_date]
            
            if len(df_features) < 50:
                print("❌ Dados insuficientes para backtest")
                return None
            
            # Fazer previsões para todo o período
            X = df_features[self.features]
            y_true = df_features['target']
            
            # Normalizar
            X_scaled = self.scaler.transform(X)
            
            # Previsões
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)
            
            # Calcular retornos
            returns = df_features['returns'].shift(-1)  # Próximo dia
            
            # Estratégia: seguir previsões com alta confiança
            confidence_threshold = 0.6
            max_proba = np.max(y_proba, axis=1)
            high_confidence_mask = max_proba >= confidence_threshold
            
            # Calcular performance
            strategy_returns = []
            buy_hold_returns = []
            positions = []
            
            for i in range(len(y_pred) - 1):  # -1 porque precisamos do retorno do próximo dia
                actual_return = returns.iloc[i]
                
                if high_confidence_mask[i]:
                    if y_pred[i] == 1:  # Previsão de subida
                        strategy_return = actual_return
                        position = 'LONG'
                    else:  # Previsão de descida
                        strategy_return = -actual_return
                        position = 'SHORT'
                else:
                    strategy_return = 0  # Ficar fora do mercado
                    position = 'HOLD'
                
                strategy_returns.append(strategy_return)
                buy_hold_returns.append(actual_return)  # Sempre long
                positions.append(position)
            
            # Métricas de performance
            strategy_returns = np.array(strategy_returns)
            buy_hold_returns = np.array(buy_hold_returns)
            
            # Retornos cumulativos
            strategy_cumret = np.cumprod(1 + strategy_returns) - 1
            buyhold_cumret = np.cumprod(1 + buy_hold_returns) - 1
            
            # Métricas
            total_strategy_return = strategy_cumret[-1]
            total_buyhold_return = buyhold_cumret[-1]
            
            # Sharpe ratio (assumindo 252 dias de trading)
            strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
            buyhold_sharpe = np.mean(buy_hold_returns) / np.std(buy_hold_returns) * np.sqrt(252) if np.std(buy_hold_returns) > 0 else 0
            
            # Max drawdown
            def max_drawdown(returns):
                cumret = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumret)
                drawdown = (cumret - running_max) / running_max
                return np.min(drawdown)
            
            strategy_mdd = max_drawdown(strategy_returns)
            buyhold_mdd = max_drawdown(buy_hold_returns)
            
            # Win rate
            winning_trades = np.sum(strategy_returns > 0)
            total_trades = np.sum(high_confidence_mask)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Accuracy geral
            accuracy = accuracy_score(y_true[:-1], y_pred[:-1])
            
            results = {
                'period': f"{df_features.index[0].strftime('%Y-%m-%d')} to {df_features.index[-1].strftime('%Y-%m-%d')}",
                'total_days': len(strategy_returns),
                'trading_days': total_trades,
                'accuracy': accuracy,
                'win_rate': win_rate,
                'strategy_return': total_strategy_return,
                'buyhold_return': total_buyhold_return,
                'excess_return': total_strategy_return - total_buyhold_return,
                'strategy_sharpe': strategy_sharpe,
                'buyhold_sharpe': buyhold_sharpe,
                'strategy_mdd': strategy_mdd,
                'buyhold_mdd': buyhold_mdd,
                'confidence_threshold': confidence_threshold
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no backtest: {e}")
            return None
    
    def print_backtest_results(self, results):
        """Imprimir resultados do backtest de forma organizada"""
        if results is None:
            return
        
        print("\n📊 === RESULTADOS DO BACKTEST ===")
        print(f"📅 Período: {results['period']}")
        print(f"📈 Total de dias: {results['total_days']}")
        print(f"💼 Dias de trading: {results['trading_days']}")
        print(f"📊 Threshold de confiança: {results['confidence_threshold']:.1%}")
        
        print(f"\n🎯 PERFORMANCE:")
        print(f"   Accuracy: {results['accuracy']:.1%}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        
        print(f"\n💰 RETORNOS:")
        print(f"   Estratégia: {results['strategy_return']:.1%}")
        print(f"   Buy & Hold: {results['buyhold_return']:.1%}")
        print(f"   Excesso: {results['excess_return']:.1%}")
        
        print(f"\n📊 MÉTRICAS DE RISCO:")
        print(f"   Sharpe Estratégia: {results['strategy_sharpe']:.2f}")
        print(f"   Sharpe Buy&Hold: {results['buyhold_sharpe']:.2f}")
        print(f"   Max DD Estratégia: {results['strategy_mdd']:.1%}")
        print(f"   Max DD Buy&Hold: {results['buyhold_mdd']:.1%}")
        
        # Interpretação
        if results['excess_return'] > 0:
            print(f"\n✅ A estratégia superou o Buy & Hold em {results['excess_return']:.1%}")
        else:
            print(f"\n❌ A estratégia ficou atrás do Buy & Hold em {abs(results['excess_return']):.1%}")
    
    def run_comprehensive_analysis(self):
        """Executar análise completa com backtest"""
        print("🔍 === ANÁLISE COMPLETA S&P 500 ===")
        
        # Download dados
        data = self.download_data(period="5y")
        if data is None:
            return
        
        # Treinar modelo se necessário
        if not self.load_model():
            print("🔄 Treinando modelo...")
            df_features = self.create_features(data)
            if df_features is None:
                return
            
            X, y = self.prepare_data(df_features)
            if X is None or y is None:
                return
            
            self.train_model(X, y)
            self.save_model()
        
        # Fazer previsão atual
        print("\n1️⃣ PREVISÃO ATUAL:")
        try:
            result = self.predict_next_day(data)
            print(f"   Direção: {result['prediction']}")
            print(f"   Confiança: {result['confidence']:.1%}")
            print(f"   Regime: {result.get('market_regime', 'N/A')}")
        except Exception as e:
            print(f"   Erro: {e}")
        
        # Backtest
        print("\n2️⃣ BACKTEST (últimos 2 anos):")
        two_years_ago = datetime.now() - timedelta(days=730)
        backtest_results = self.backtest_strategy(data, start_date=two_years_ago)
        self.print_backtest_results(backtest_results)
        
        # Relatório avançado
        print("\n3️⃣ ANÁLISE DO MODELO:")
        self.generate_enhanced_report()

# Função principal melhorada
def main():
    """Função principal com opções"""
    try:
        predictor = EnhancedSP500Predictor()
        
        print("🚀 Enhanced SP500 Predictor")
        print("Escolha uma opção:")
        print("1 - Previsão diária")
        print("2 - Análise completa com backtest")
        print("3 - Apenas treinar modelo")
        
        try:
            choice = input("Opção (1-3): ").strip()
        except:
            choice = "1"  # Default
        
        if choice == "2":
            predictor.run_comprehensive_analysis()
        elif choice == "3":
            data = predictor.download_data("3y")
            if data is not None and not data.empty:
                df_features = predictor.create_features(data)
                if df_features is not None and not df_features.empty:
                    X, y = predictor.prepare_data(df_features)
                    if X is not None and y is not None:
                        predictor.train_model(X, y)
                        predictor.save_model()
                        print("✅ Modelo treinado e salvo!")
        else:  # Default: previsão diária
            predictor.run_daily_prediction()
            
    except KeyboardInterrupt:
        print("\n⏹️ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro geral: {e}")

if __name__ == "__main__":
    main()