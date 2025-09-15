# AI Trading Bot

An intelligent cryptocurrency trading bot powered by machine learning and advanced algorithms.

## 🚀 Features

- **AI-Powered Predictions**: Uses LSTM, Transformer, and XGBoost models for price prediction
- **Risk Management**: Advanced risk assessment with VaR, CVaR, and Monte Carlo simulations
- **Multi-Exchange Support**: Compatible with Binance, Coinbase, Kraken, and more
- **Real-time Data**: Live market data processing and analysis
- **Backtesting**: Comprehensive backtesting framework for strategy validation
- **Paper Trading**: Safe testing environment before live trading
- **Portfolio Management**: Automated portfolio optimization and rebalancing
- **Monitoring**: Real-time performance metrics and alerts

## 📁 Project Structure

```
ai-trading-bot/
├── src/                    # Source code
│   ├── models/            # AI/ML models
│   ├── strategies/        # Trading strategies
│   ├── data/              # Data processing modules
│   ├── utils/             # Utility functions
│   └── config/            # Configuration management
├── tests/                 # Test files
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── data/                  # Data storage
├── logs/                  # Log files
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-trading-bot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the database**
   ```bash
   python scripts/setup_database.py
   ```

## ⚙️ Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

- **API Keys**: Binance, Alpha Vantage, etc.
- **Database**: PostgreSQL connection string
- **Trading**: Mode (paper/live), initial balance, etc.
- **Security**: Encryption keys and JWT secrets

### Configuration File

Edit `config/config.yaml` to customize:

- Trading parameters (stop loss, take profit, position sizes)
- AI model settings (lookback windows, prediction horizons)
- Data sources and update intervals
- Risk management parameters

## 🚀 Quick Start

1. **Paper Trading Mode** (Recommended for beginners)
   ```bash
   python src/main.py
   ```

2. **Backtesting**
   ```bash
   python scripts/backtest.py --strategy lstm --start-date 2023-01-01 --end-date 2023-12-31
   ```

3. **Model Training**
   ```bash
   python scripts/train_model.py --model lstm --symbol BTCUSDT --period 1y
   ```

## 📊 Supported Exchanges

- **Binance** (Primary)
- **Coinbase Pro**
- **Kraken**
- **Alpha Vantage** (Market data)

## 🤖 AI Models

### Prediction Models
- **LSTM**: Long Short-Term Memory networks for time series
- **Transformer**: Attention-based models for sequence prediction
- **XGBoost**: Gradient boosting for feature-based predictions

### Risk Models
- **VaR**: Value at Risk calculations
- **CVaR**: Conditional Value at Risk
- **Monte Carlo**: Monte Carlo simulations for risk assessment

## 📈 Trading Strategies

- **Trend Following**: Momentum-based strategies
- **Mean Reversion**: Contrarian approaches
- **Arbitrage**: Cross-exchange opportunities
- **Market Making**: Liquidity provision strategies

## 🔒 Security

- API key encryption
- Secure credential storage
- Rate limiting and request throttling
- Input validation and sanitization

## 📝 Logging

Comprehensive logging system with:
- Real-time console output
- File-based logging with rotation
- Structured logging for analysis
- Performance metrics tracking

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

## 📊 Monitoring

The bot includes built-in monitoring with:
- Real-time performance metrics
- Health checks and alerts
- Prometheus metrics export
- Web dashboard (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.**

## 📞 Support

- Create an issue for bug reports or feature requests
- Check the [documentation](docs/) for detailed guides
- Join our community discussions

## 🗺️ Roadmap

- [ ] Web-based dashboard
- [ ] Mobile app integration
- [ ] Advanced portfolio optimization
- [ ] Social trading features
- [ ] Multi-asset support (stocks, forex)
- [ ] Cloud deployment options

---

**Happy Trading! 🚀📈**
