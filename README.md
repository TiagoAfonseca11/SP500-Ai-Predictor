# 🚀 Guia Completo - Como Usar o SP500 Predictor

## 📋 Pré-requisitos

### 1. Instalação do Python
Python 3.8+

### 2. Instalar Dependências
Cria um ficheiro `requirements.txt`:
```txt
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
```

Instala as dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Estrutura do Projeto

```
sp500_predictor/
├── sp500_predictor.py      # Classe principal do predictor
├── analysis.py             # Análise de resultados
├── automation.py           # Versão corrigida do analysis.py
├── run_prediction.py       # Script para execução única
├── requirements.txt        # Dependências
├── daily_predictions.csv   # Histórico de previsões (criado automaticamente)
├── sp500_model.pkl        # Modelo treinado (criado automaticamente)
└── prediction_analysis.png # Gráficos de análise (criado automaticamente)
```

## 🏃‍♂️ Como Executar

### **Opção 1: Execução Única (Recomendado para Início)**

```bash
python run_prediction.py
```

**O que acontece:**
- ✅ Baixa dados históricos do S&P 500 (últimos 3 anos)
- ✅ Cria features técnicas (RSI, MACD, Bollinger Bands, etc.)
- ✅ Treina modelo de machine learning (se não existir)
- ✅ Faz previsão para o próximo dia
- ✅ Guarda resultado em `daily_predictions.csv`
- ✅ Salva modelo treinado em `sp500_model.pkl`

### **Opção 2: Execução Direta da Classe**

```python
from sp500_predictor import SP500Predictor

# Criar instância
predictor = SP500Predictor()

# Executar previsão
predictor.run_daily_prediction()
```

### **Opção 3: Análise de Resultados**

Depois de teres algumas previsões guardadas:

```bash
python automation.py  # Este é o analysis.py corrigido
```

## 📊 Interpretando os Resultados

### **Saída da Previsão:**
```
=== PREVISÃO DIÁRIA S&P 500 ===
Data: 2025-01-31 14:30:00

📊 PREVISÃO PARA AMANHÃ:
Direção: SUBIR
Probabilidade de SUBIR: 67.3%
Probabilidade de DESCER: 32.7%
Confiança: 67.3%

Último fechamento: $4,567.89
Volume: 3,456,789,012
Nível de confiança: ALTA
```

### **Níveis de Confiança:**
- 🟢 **ALTA (>60%)**: Previsão mais confiável
- 🟡 **MÉDIA (55-60%)**: Confiança moderada  
- 🔴 **BAIXA (<55%)**: Menos confiável

## 📈 Análise de Performance

### **Executar Análise:**
```bash
python automation.py
```

### **Métricas Principais:**
- **Accuracy Geral**: % de previsões corretas
- **Accuracy Alta Confiança**: % corretas apenas com confiança >60%
- **Win Rate**: Taxa de acerto na simulação de trading
- **Excess Return**: Retorno vs Buy & Hold

### **Ficheiros Gerados:**
- `prediction_analysis.png`: Gráficos de performance
- `detailed_results.csv`: Resultados detalhados

## 🔄 Uso Avançado

### **1. Treinar Novo Modelo**
```bash
# Deletar modelo existente
rm sp500_model.pkl

# Executar novamente (vai treinar novo modelo)
python run_prediction.py
```

### **2. Usar Modelo Existente**
```python
from sp500_predictor import SP500Predictor

predictor = SP500Predictor()

# Carregar modelo existente
if predictor.load_model():
    # Baixar dados recentes
    data = predictor.download_data("1y")
    
    # Fazer previsão
    result = predictor.predict_next_day(data)
    print(f"Previsão: {result['prediction']}")
    print(f"Confiança: {result['confidence']:.1%}")
```

### **3. Backtesting Manual**
```python
# Análise personalizada
from automation import PredictionAnalyzer

analyzer = PredictionAnalyzer('daily_predictions.csv')
analyzer.load_predictions()
analyzer.get_actual_results()

# Calcular métricas
metrics = analyzer.calculate_accuracy()
print(f"Accuracy: {metrics['overall_accuracy']:.1%}")

# Simulação de trading
trading_results = analyzer.calculate_trading_simulation(
    initial_capital=10000
)
print(f"Retorno: {trading_results['total_return']:.1%}")
```

## 🛠️ Personalização

### **Alterar Parâmetros do Modelo**
Editar `sp500_predictor.py`:
```python
# Linha ~185 - Alterar parâmetros Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,  # Aumentar árvores
    max_depth=15,      # Maior profundidade
    random_state=42
)
```

### **Adicionar Novas Features**
```python
# No método create_features(), adicionar:
def create_features(self, data):
    # ... código existente ...
    
    # Nova feature personalizada
    df['custom_indicator'] = df['Close'].rolling(30).mean() / df['Close']
    
    return df
```

### **Alterar Critérios de Confiança**
```python
# Alterar limite na simulação de trading
if row['confidence'] > 0.7:  # Era 0.6, agora 0.7
    # Fazer trade
```

## 🚨 Troubleshooting

### **Erro: "Nenhum dado foi baixado"**
```bash
# Testar conexão yfinance
python -c "import yfinance as yf; print(yf.download('^GSPC', period='5d'))"
```

### **Erro: "Dados insuficientes"**
- Aumenta o período de dados: `period="5y"` em vez de `period="3y"`
- Verifica conexão à internet

### **Erro: "Module not found"**
```bash
# Reinstalar dependências
pip install --upgrade yfinance pandas scikit-learn matplotlib seaborn
```

### **Modelo com Performance Baixa**
- Apaga o modelo: `rm sp500_model.pkl`
- Executa novamente para retreinar
- Aumenta período de dados para treino

## 🎯 Workflow Recomendado

### **Para Utilizador Novo:**
1. Executa `python run_prediction.py` uma vez
2. Espera alguns dias para acumular previsões
3. Executa `python automation.py` para análise
4. Ajusta parâmetros se necessário

### **Para Uso Diário:**
```bash
# Executar todos os dias
python run_prediction.py

# Análise semanal
python automation.py
```

### **Para Desenvolvimento:**
1. Modifica `sp500_predictor.py`
2. Testa com `python run_prediction.py`
3. Analisa resultados com `automation.py`
4. Itera e melhora

## 📝 Notas Importantes

⚠️ **Aviso Legal**: Este é um projeto educacional. Não usar para decisões financeiras reais sem análise profissional.

✅ **Boas Práticas**:
- Executar diariamente na mesma hora
- Manter histórico de previsões
- Analisar performance regularmente
- Não confiar apenas numa previsão

🔄 **Manutenção**:
- Retreinar modelo mensalmente
- Monitorizar accuracy
- Ajustar parâmetros conforme necessário

## 🆘 Suporte

Se tiveres problemas:
1. Verifica se todas as dependências estão instaladas
2. Confirma conexão à internet
3. Verifica se os ficheiros não estão corrompidos
4. Tenta apagar modelo e retreinar

**Ficheiros a verificar em caso de erro:**
- `daily_predictions.csv` (deve ter formato correto)
- `sp500_model.pkl` (pode estar corrompido)
- Logs de erro no terminal