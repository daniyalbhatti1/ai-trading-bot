# 📈 AI-Powered Algorithmic Trading Bot

A fully automated algorithmic trading bot that trades SPY, QQQ, GLD, and USO using technical analysis combined with machine learning. Built with Python, LightGBM, Alpaca API for paper trading, and Streamlit for real-time monitoring.

## 🎯 Features

- **ML-Enhanced Signal Generation**: LightGBM model validates trades with 60%+ confidence threshold
- **Technical Analysis Foundation**: RSI, EMA, MACD indicators for signal generation
- **Mean Reversion + Trend Following**: Combined strategy for high-probability setups
- **Adaptive Learning System**: Trade journal automatically improves model over time
- **Paper Trading**: Safe testing with Alpaca's paper trading API
- **Real-time Dashboard**: Streamlit dashboard for monitoring performance
- **Automated Execution**: Stop-loss and take-profit management
- **Backtesting Engine**: Test strategies on historical data
- **Multi-Symbol Support**: Trade multiple symbols simultaneously

## 📊 Trading Strategy

The bot combines rule-based technical analysis with machine learning for signal validation:

### **Signal Generation (Technical Analysis)**

**1. Mean Reversion (Primary)**
- **Extreme RSI**: Trades RSI < 25 (oversold) and RSI > 75 (overbought)
- **Rule Confidence**: 75%+ for extreme mean reversion setups

**2. Trend-Aligned Mean Reversion**
- **Pullback Entries**: Buys pullbacks in uptrends, sells rallies in downtrends
- **EMA Confirmation**: 9/21 EMA crossovers define trend direction
- **Rule Confidence**: 65-85% based on trend strength

**3. Trend Following with Momentum**
- **MACD + EMA**: Both must align for trend continuation
- **Not Overbought/Oversold**: Avoids extremes (RSI 40-60 range)
- **Rule Confidence**: 60-80%

### **ML Validation (LightGBM)**

Every signal is validated by a trained LightGBM model that:
- Analyzes 20+ technical features (RSI, EMA, MACD, volume, volatility, etc.)
- Predicts probability of profitable trade outcome
- **Requires 60%+ ML confidence** to execute trades
- Continuously learns from trade outcomes via trade journal

### **Risk Management**
- **Stop Loss**: 1.5% automatic stop loss on all positions
- **Take Profit**: 3% first target (2:1 R/R), 6% second target (4:1 R/R)
- **Partial Profits**: 50% position closed at first target, stop moved to breakeven
- **Position Sizing**: Fixed quantity per trade (configurable)
- **Max Positions**: 4 concurrent positions maximum

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11+
- Alpaca Paper Trading Account (free)

### **Installation**

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-trading-bot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Configuration**

1. **Create `.env` file** in the project root:
```env
ALPACA_API_KEY_ID=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get your free Alpaca API keys from https://alpaca.markets

2. **Adjust `config.yaml`** (optional):
```yaml
universe:
  - SPY
  - QQQ
  - GLD
  - USO

risk:
  max_positions: 4
  stop_loss_pct: 0.015      # 1.5%
  take_profit_pct: 0.03     # 3%
  take_profit_2_pct: 0.06   # 6%

strategy:
  confidence_min: 0.50      # Minimum rule confidence
  ml_confidence_min: 0.60   # Minimum ML model confidence
```

### **Usage**

1. **Initialize Database**
```bash
python3 scripts/init_db.py
```

2. **Backfill Historical Data**
```bash
python3 scripts/backfill.py --symbols SPY QQQ GLD USO --period 7d
```

3. **Train ML Model** (or use pre-trained model in `trained_models/lgbm.pkl`)
```bash
python3 scripts/train_lgbm.py
```

4. **Run Backtest** (optional - test strategy first)
```bash
python3 scripts/quick_backtest.py --symbols SPY QQQ GLD USO
```

5. **Start Paper Trading**
```bash
python3 main.py
# or use the full system with API server:
python3 start_trading.py
```

6. **Launch Dashboard** (in a separate terminal)
```bash
streamlit run app/dashboard/app.py
```
Dashboard opens at `http://localhost:8501`

## 📱 Dashboard Features

- 📊 **Equity Curve**: Track account equity over time
- 💼 **Open Positions**: Monitor current positions and P&L
- 📋 **Order History**: View all executed orders
- 🎯 **Trading Signals**: Recent buy/sell signals
- 📈 **Performance Metrics**: Win rate, profit factor, Sharpe ratio
- ⏯️ **Trading Controls**: Start/pause/resume/stop trading

## 🔧 Project Structure

```
ai-trading-bot/
├── app/
│   ├── api/              # FastAPI server
│   ├── backtest/         # Backtesting engine
│   ├── core/             # Settings, logger, utils
│   ├── dashboard/        # Streamlit dashboard
│   ├── data/             # Database management
│   ├── ingestion/        # Data ingestion (Alpaca, yfinance)
│   ├── jobs/             # Scheduled tasks
│   ├── learning/         # Trade journal & ML learning system
│   ├── models/           # LightGBM ML model
│   ├── signals/          # Technical indicators & trading rules
│   └── trading/          # Broker integration & execution
├── scripts/              # Utility scripts (backfill, train, backtest)
├── trained_models/       # Trained ML models
├── config.yaml           # Trading configuration
├── main.py               # Main entry point
└── requirements.txt      # Dependencies
```

## 📈 Performance Metrics

- **Equity Curve**: Real-time account value
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline

## 🤖 Machine Learning Model

**LightGBM Classifier**
- **Input Features**: 20+ technical indicators and price patterns
- **Training Data**: Historical trades with known outcomes
- **Validation**: Cross-validation with time series splits
- **Retraining**: Automatic retraining from trade journal entries
- **Confidence Threshold**: 60% minimum for trade execution

## 📚 Technical Features

The ML model analyzes these features for every trade signal:

- **RSI (14)**: Relative Strength Index for overbought/oversold levels
- **EMA (9/21)**: Exponential Moving Averages for trend identification
- **MACD (12/26/9)**: Moving Average Convergence Divergence for momentum
- **ATR (14)**: Average True Range for volatility measurement
- **Volume Indicators**: Volume trends and anomalies
- **Price Patterns**: Support/resistance, highs/lows, price momentum
- **Cross-symbol Correlation**: Market-wide signals from SPY, QQQ


## 🛡️ Risk Disclaimer

**⚠️ FOR EDUCATIONAL AND PAPER TRADING PURPOSES ONLY**

- Never risk money you can't afford to lose
- Past performance does not guarantee future results
- Always test thoroughly with paper trading first
- Trading involves substantial risk of loss

## 📄 License

MIT License - see LICENSE file for details

---
