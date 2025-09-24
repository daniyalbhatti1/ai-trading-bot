# Advanced AI Trading Bot

A sophisticated, production-ready AI trading bot that combines cutting-edge machine learning models with advanced trading strategies. Built with Lumibot framework for institutional-grade paper trading and backtesting capabilities.

## 🚀 Key Features

### 📊 Multi-Asset Trading Support
- **🪙 Cryptocurrency Trading**: Bitcoin, Ethereum, and 50+ altcoins via Binance
- **📈 Stock Trading**: Major stocks, ETFs, and indices via Alpaca
- **🔄 Easy Asset Switching**: Switch between crypto and stocks with one command
- **⏰ Flexible Trading Hours**: 24/7 crypto trading or market hours for stocks

### 🤖 Optimized AI/ML Models
- **Fast Super Ensemble**: Lightweight XGBoost + LightGBM combination
- **Smart Feature Engineering**: 50+ technical indicators optimized for performance
- **Lazy Loading**: Models load only when needed for faster startup
- **Intelligent Caching**: Compressed serialization and parallel processing

### 📈 Mean Reversion Strategy
- **Statistical Analysis**: Z-scores, Bollinger Bands, and RSI indicators
- **Risk Management**: Stop-loss, take-profit, and position sizing
- **Confidence Filtering**: Only trades with high signal strength and confidence
- **Adaptive Parameters**: Optimized for both crypto and stock markets

### 🛡️ Enterprise-Grade Risk Management
- **VaR/CVaR**: Value at Risk and Conditional Value at Risk calculations
- **Monte Carlo Simulations**: 10,000+ scenario stress testing
- **Portfolio Optimization**: Modern Portfolio Theory with risk constraints
- **Dynamic Position Sizing**: Kelly Criterion and volatility-adjusted sizing

### 🔧 Production-Ready Infrastructure
- **Lumibot Integration**: Professional backtesting and paper trading framework
- **Real-time Data**: WebSocket connections for live market data
- **Advanced Configuration**: Environment-based config with validation
- **Comprehensive Logging**: Structured logging with performance metrics
- **Monitoring & Alerts**: Real-time performance tracking and notifications

## 📁 Project Architecture

```
ai-trading-bot/
├── src/
│   ├── models/                    # ML Model Implementations
│   │   ├── base_model.py         # Abstract base class
│   │   ├── lstm_model.py         # LSTM with attention
│   │   ├── transformer_model.py  # Transformer architecture
│   │   ├── xgboost_model.py      # XGBoost with feature engineering
│   │   ├── ensemble_model.py     # Model ensemble framework
│   │   ├── feature_engineering.py # Advanced feature creation
│   │   └── model_factory.py      # Model creation and management
│   ├── strategies/               # Trading Strategy Implementations
│   │   ├── base_strategy.py      # Abstract strategy base
│   │   ├── mean_reversion_strategy.py # Statistical arbitrage
│   │   ├── momentum_strategy.py  # Trend following
│   │   ├── arbitrage_strategy.py # Multi-type arbitrage
│   │   └── strategy_factory.py   # Strategy management
│   ├── config/                   # Configuration Management
│   │   ├── config.py            # Advanced config system
│   │   └── __init__.py
│   ├── utils/                    # Utility Functions
│   │   ├── logger.py            # Logging setup
│   │   ├── data_manager.py      # Data handling
│   │   ├── risk_manager.py      # Risk management
│   │   └── performance_tracker.py # Performance analytics
│   └── main.py                  # Main application entry point
├── config/
│   └── config.yaml              # Main configuration file
├── requirements.txt             # Python dependencies
├── env.example                  # Environment variables template
└── README.md                   # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd ai-trading-bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp env.example .env
# Edit .env with your API keys and settings
```

### 4. Choose Your Asset Class
```bash
# For cryptocurrency trading (24/7)
python switch_asset_class.py crypto

# For stock trading (market hours)
python switch_asset_class.py stocks

# Check current configuration
python switch_asset_class.py status
```

### 5. Configure Trading Parameters
Edit `config/config.yaml` to customize:
- Model parameters and hyperparameters
- Trading strategy settings
- Risk management thresholds
- Data sources and update frequencies

## 🚀 Quick Start

### Choose Your Asset Class
```bash
# For cryptocurrency trading (24/7)
python switch_asset_class.py crypto

# For stock trading (market hours)  
python switch_asset_class.py stocks

# Check current configuration
python switch_asset_class.py status
```

### Paper Trading (Recommended)
```bash
python src/main.py paper
```

### Backtesting
```bash
python src/main.py backtest
```

### Demo Asset Switching
```bash
python demo_asset_switching.py
```

## 🤖 AI Models Deep Dive

### LSTM Model
- **Architecture**: Multi-layer LSTM with attention mechanism
- **Features**: Dropout, batch normalization, regularization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Hyperparameter Tuning**: Optuna-based optimization

### Transformer Model
- **Architecture**: Multi-head attention with positional encoding
- **Features**: Self-attention, feed-forward networks, layer normalization
- **Optimization**: Advanced attention mechanisms for time series
- **Interpretability**: Attention weight visualization

### Ensemble Model
- **Combination**: Weighted average, stacking, and voting methods
- **Optimization**: Dynamic weight adjustment based on performance
- **Robustness**: Multiple model types for diverse predictions
- **Performance**: Superior to individual models

## 📈 Trading Strategies Explained

### Mean Reversion Strategy
- **Z-Score Analysis**: Statistical deviation from mean
- **Bollinger Bands**: Price channel analysis
- **RSI Divergence**: Momentum confirmation
- **Cointegration**: Pairs trading opportunities
- **Half-Life**: Mean reversion speed estimation

### Momentum Strategy
- **MACD Signals**: Trend change detection
- **ADX Filter**: Trend strength confirmation
- **Breakout Detection**: Volume-confirmed breakouts
- **Multi-Timeframe**: Trend alignment across timeframes
- **Risk Management**: Dynamic stop-loss and take-profit

### Arbitrage Strategy
- **Exchange Arbitrage**: Cross-exchange price differences
- **Triangular Arbitrage**: Three-currency profit opportunities
- **Statistical Arbitrage**: Cointegrated pairs trading
- **Latency Optimization**: High-frequency execution
- **Risk Controls**: Maximum exposure limits

## 🔧 Advanced Configuration

### Model Configuration
```yaml
models:
  - name: "ensemble_model"
    type: "ensemble"
    base_models: ["lstm", "xgboost", "catboost"]
    ensemble_method: "weighted_average"
    optimize_weights: true
    hyperparameter_optimization: true
```

### Strategy Configuration
```yaml
strategies:
  - name: "mean_reversion"
    type: "mean_reversion"
    zscore_threshold: 2.0
    lookback_period: 20
    use_cointegration: true
    use_half_life: true
```

### Risk Management
```yaml
risk_management:
  max_position_size: 0.1
  stop_loss: 0.02
  take_profit: 0.05
  max_drawdown: 0.15
  var_confidence_level: 0.95
  monte_carlo_simulations: 10000
```

## 📊 Performance Metrics

### Trading Metrics
- **Total Return**: Portfolio performance over time
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annual return / Maximum drawdown

### Model Metrics
- **RMSE/MAE**: Prediction accuracy
- **Directional Accuracy**: Correct trend prediction
- **R² Score**: Model fit quality
- **Cross-Validation**: Out-of-sample performance
- **Feature Importance**: Model interpretability

## 🔒 Security & Risk Management

### API Security
- Encrypted API key storage
- Rate limiting and request throttling
- Secure credential management
- Session timeout controls

### Trading Risk Controls
- Position size limits
- Maximum daily trades
- Stop-loss and take-profit orders
- Portfolio-level risk monitoring
- Real-time risk alerts

### Data Security
- Encrypted data storage
- Secure data transmission
- Access control and authentication
- Audit logging

## 🧪 Testing & Validation

### Unit Testing
```bash
pytest tests/unit/ -v
```

### Integration Testing
```bash
pytest tests/integration/ -v
```

### Backtesting
```bash
python scripts/backtest.py --strategy ensemble --start-date 2020-01-01 --end-date 2023-12-31
```

### Performance Testing
```bash
python scripts/performance_test.py --models all --strategies all
```

## 📊 Monitoring & Analytics

### Real-time Monitoring
- Portfolio performance tracking
- Model prediction accuracy
- Strategy signal generation
- Risk metric monitoring
- System health checks

### Performance Analytics
- Detailed performance reports
- Risk-adjusted return analysis
- Drawdown analysis
- Trade analysis and statistics
- Model performance comparison

### Alerting System
- Performance threshold alerts
- Risk limit breaches
- System error notifications
- Model degradation warnings
- Market condition changes

## 🚀 Deployment Options

### Local Development
- Paper trading with Alpaca
- Backtesting with historical data
- Model training and validation
- Strategy optimization

### Cloud Deployment
- AWS/GCP/Azure deployment
- Containerized with Docker
- Kubernetes orchestration
- Auto-scaling capabilities

### Production Trading
- Live trading with proper risk controls
- Multi-exchange connectivity
- High-frequency execution
- Institutional-grade infrastructure

## 📚 Advanced Features

### Feature Engineering
- 100+ technical indicators
- Statistical features (skewness, kurtosis)
- Fourier transform features
- Wavelet analysis
- Volume profile analysis
- Sentiment analysis integration

### Model Optimization
- Hyperparameter tuning with Optuna
- Cross-validation strategies
- Feature selection algorithms
- Model ensemble optimization
- Online learning capabilities

### Risk Management
- Value at Risk (VaR) calculations
- Conditional VaR (CVaR)
- Monte Carlo stress testing
- Scenario analysis
- Portfolio optimization
- Dynamic hedging strategies

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone <repo-url>
cd ai-trading-bot
pip install -r requirements-dev.txt
pre-commit install
```

### Code Quality
- Black code formatting
- Flake8 linting
- MyPy type checking
- Pytest testing
- Pre-commit hooks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimers

**This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.**

**The authors and contributors are not responsible for any financial losses incurred through the use of this software. Use at your own risk.**

