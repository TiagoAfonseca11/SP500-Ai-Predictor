let isLoading = false;
        let debugMode = false; // Set to true for debugging

        function log(message, data = null) {
            if (debugMode) {
                console.log(message, data);
            }
        }

        function showLoading() {
            isLoading = true;
            document.getElementById('loading').style.display = 'block';
            document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
        }

        function hideLoading() {
            isLoading = false;
            document.getElementById('loading').style.display = 'none';
            document.querySelectorAll('.btn').forEach(btn => btn.disabled = false);
        }

        function showAlert(message, type = 'success') {
            const alertsDiv = document.getElementById('alerts');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : 'times-circle'}"></i> ${message}`;
            
            alertsDiv.appendChild(alert);
            
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 5000);
        }

        async function makeApiCall(endpoint, method = 'POST', data = null) {
            try {
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                    }
                };
                
                if (data && method !== 'GET') {
                    options.body = JSON.stringify(data);
                }
                
                log(`Making API call to ${endpoint}`, data);
                
                const response = await fetch(endpoint, options);
                const responseData = await response.json();
                
                log(`API response from ${endpoint}:`, responseData);
                
                if (!response.ok) {
                    throw new Error(responseData.error || `HTTP error! status: ${response.status}`);
                }
                
                return responseData;
                
            } catch (error) {
                log(`API error for ${endpoint}:`, error);
                throw error;
            }
        }

        async function runPrediction() {
            if (isLoading) return;
            
            showLoading();
            try {
                const data = await makeApiCall('/api/predict');
                
                if (data.success && data.data) {
                    updatePredictionDisplay(data.data);
                    showAlert('Prediction completed successfully!');
                } else {
                    throw new Error(data.error || 'Invalid response data');
                }
                
            } catch (error) {
                console.error('Prediction error:', error);
                showAlert(`Failed to run prediction: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }

        async function runBacktest() {
            if (isLoading) return;
            
            showLoading();
            try {
                const data = await makeApiCall('/api/backtest', 'POST', { days_back: 730 });
                
                if (data.success && data.data) {
                    updateBacktestDisplay(data.data);
                    showAlert('Backtest completed successfully!');
                } else {
                    throw new Error(data.error || 'Invalid backtest response');
                }
                
            } catch (error) {
                console.error('Backtest error:', error);
                showAlert(`Failed to run backtest: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }

        async function trainModel() {
            if (isLoading) return;
            
            showLoading();
            try {
                const data = await makeApiCall('/api/train');
                
                if (data.success) {
                    showAlert(`Model trained successfully! Features: ${data.features_count}, Data points: ${data.data_points}`);
                } else {
                    throw new Error(data.error || 'Training failed');
                }
                
            } catch (error) {
                console.error('Training error:', error);
                showAlert(`Failed to train model: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }

        async function refreshData() {
            if (isLoading) return;
            
            showLoading();
            try {
                const data = await makeApiCall('/api/refresh');
                
                if (data.success) {
                    showAlert(`Data refreshed successfully! Latest: ${data.latest_date} - ${data.latest_price?.toFixed(2)}`);
                } else {
                    throw new Error(data.error || 'Refresh failed');
                }
                
            } catch (error) {
                console.error('Refresh error:', error);
                showAlert(`Failed to refresh data: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }

        function updatePredictionDisplay(data) {
            log('Updating prediction display:', data);
            
            try {
                // Update prediction direction
                const directionEl = document.getElementById('predictionDirection');
                const prediction = data.prediction || 'UNKNOWN';
                directionEl.textContent = prediction;
                directionEl.className = `prediction-direction ${(prediction === 'SUBIR') ? 'up' : 'down'}`;
                
                // Update confidence
                const confidence = Math.round((data.confidence || 0) * 100);
                document.getElementById('confidenceValue').textContent = `${confidence}%`;
                document.getElementById('confidenceFill').style.width = `${confidence}%`;
                
                // Update probabilities
                document.getElementById('probUp').textContent = `${Math.round((data.probability_up || 0) * 100)}%`;
                document.getElementById('probDown').textContent = `${Math.round((data.probability_down || 0) * 100)}%`;
                
                // Update market context
                document.getElementById('lastPrice').textContent = data.last_price ? `${data.last_price.toFixed(2)}` : 'N/A';
                document.getElementById('marketRegime').textContent = data.market_regime || 'N/A';
                document.getElementById('vixLevel').textContent = data.vix_level ? data.vix_level.toFixed(1) : 'N/A';
                
                // Update trend strength
                const trendStrength = data.trend_strength;
                let trendText = 'N/A';
                if (trendStrength !== null && trendStrength !== undefined) {
                    const strength = Math.abs(trendStrength);
                    const direction = trendStrength > 0 ? 'Up' : 'Down';
                    const level = strength > 0.7 ? 'Strong' : strength > 0.3 ? 'Moderate' : 'Weak';
                    trendText = `${level} ${direction}`;
                }
                document.getElementById('trendStrength').textContent = trendText;
                
                // Update feature importance if available
                if (data.feature_importance && data.feature_importance.length > 0) {
                    updateFeatureImportance(data.feature_importance);
                }
                
            } catch (error) {
                console.error('Error updating prediction display:', error);
                showAlert('Error updating display', 'warning');
            }
        }

        function updateFeatureImportance(features) {
            log('Updating feature importance:', features);
            
            try {
                const container = document.getElementById('featureImportance');
                container.innerHTML = '';
                
                if (!features || features.length === 0) {
                    container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No feature importance data available</p>';
                    return;
                }
                
                features.slice(0, 10).forEach(feature => {
                    const item = document.createElement('div');
                    item.className = 'feature-item';
                    const importance = (feature.importance || 0) * 100;
                    item.innerHTML = `
                        <div class="feature-name">${feature.name || 'Unknown'}</div>
                        <div class="feature-bar-container">
                            <div class="feature-bar">
                                <div class="feature-bar-fill" style="width: ${importance}%"></div>
                            </div>
                            <div class="feature-value">${importance.toFixed(1)}%</div>
                        </div>
                    `;
                    container.appendChild(item);
                });
                
            } catch (error) {
                console.error('Error updating feature importance:', error);
            }
        }

        function updateBacktestDisplay(data) {
            log('Updating backtest display:', data);
            
            try {
                document.getElementById('winRate').textContent = data.win_rate ? `${Math.round(data.win_rate * 100)}%` : 'N/A';
                document.getElementById('strategyReturn').textContent = data.strategy_return ? `${Math.round(data.strategy_return * 100)}%` : 'N/A';
                document.getElementById('sharpeRatio').textContent = data.strategy_sharpe ? data.strategy_sharpe.toFixed(2) : 'N/A';
                document.getElementById('maxDrawdown').textContent = data.strategy_mdd ? `${Math.round(Math.abs(data.strategy_mdd) * 100)}%` : 'N/A';
                
            } catch (error) {
                console.error('Error updating backtest display:', error);
                showAlert('Error updating backtest display', 'warning');
            }
        }

        // Initialize with status check
        document.addEventListener('DOMContentLoaded', async function() {
            showAlert('S&P 500 AI Predictor loaded successfully!', 'success');
            
            // Check system status
            try {
                const status = await makeApiCall('/api/status', 'GET');
                if (status.success) {
                    const statusInfo = status.status;
                    if (statusInfo.latest_price) {
                        document.getElementById('lastPrice').textContent = `${statusInfo.latest_price.toFixed(2)}`;
                    }
                    
                    if (statusInfo.model_trained) {
                        showAlert('Model ready for predictions', 'success');
                    } else {
                        showAlert('No trained model found. Please train a model first.', 'warning');
                    }
                }
            } catch (error) {
                console.error('Status check failed:', error);
                showAlert('System status check failed', 'warning');
            }
        });

        // Auto-refresh every 5 minutes (disabled by default to prevent spam)
        // setInterval(() => {
        //     if (!isLoading) {
        //         refreshData();
        //     }
        // }, 300000);