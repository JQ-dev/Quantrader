# Bitcoin Quantitative Trader

A fully functional, production-ready Bitcoin quantitative trading system built with Python. This system implements multiple proven trading strategies, comprehensive risk management, and supports both backtesting and live trading on cryptocurrency exchanges.

## Features

### Trading Strategies
- **Trend Following**: Golden/Death Cross detection with ADX confirmation
- **Momentum**: RSI and MACD-based momentum trading
- **Mean Reversion**: Bollinger Bands-based reversal trading
- **Machine Learning**: Random Forest classifier with technical indicators
- **Ensemble Trading**: Combine multiple strategies with weighted signals

### Risk Management
- Position sizing (Fixed Fractional, Kelly Criterion, Volatility-based, Risk Parity)
- Stop loss and take profit automation
- Daily loss limits
- Maximum position limits
- Real-time risk monitoring

### Backtesting Engine
- Historical data analysis
- Comprehensive performance metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Trade-by-trade reporting
- Multi-strategy ensemble testing

### Exchange Integration
- CCXT library support (100+ exchanges)
- Market and limit order execution
- Stop loss order management
- Paper trading mode (dry run)
- Real-time balance tracking

### Data Management
- Automated OHLCV data fetching
- Data caching for performance
- CSV import/export
- Technical indicator calculation (20+ indicators)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Quantrader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your exchange API credentials
```

4. Configure trading parameters:
```bash
# Edit config/config.yaml with your preferences
```

## Configuration

### Environment Variables (.env)

```env
# Exchange Configuration
EXCHANGE=binance
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here

# Trading Settings
TRADING_MODE=paper  # paper or live
SYMBOL=BTC/USDT
CAPITAL=10000

# Risk Management
MAX_POSITION_SIZE=0.1
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
```

### Strategy Configuration (config/config.yaml)

```yaml
strategies:
  active:
    - trend_following
    - momentum
    - mean_reversion

  weights:
    trend_following: 0.4
    momentum: 0.3
    mean_reversion: 0.3
```

## Usage

### Running Backtests

```bash
# Run backtest with default configuration
python -m src.bot --mode backtest

# Run with custom config
python -m src.bot --mode backtest --config config/config.yaml
```

### Live Trading

```bash
# Paper trading (recommended for testing)
python -m src.bot --mode live --interval 3600

# Live trading (use with caution!)
# Set TRADING_MODE=live in .env
python -m src.bot --mode live --interval 3600
```

### Python API

```python
from src.bot import QuantTrader

# Initialize bot
bot = QuantTrader(config_path='config/config.yaml')

# Run backtest
results = bot.run_backtest(
    start_date='2023-01-01',
    end_date='2024-12-31'
)

# Analyze results
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

## Project Structure

```
Quantrader/
├── src/
│   ├── data/              # Data fetching and storage
│   ├── strategies/        # Trading strategies
│   ├── backtest/          # Backtesting engine
│   ├── risk/             # Risk management
│   ├── execution/        # Order execution
│   ├── utils/            # Utilities
│   └── bot.py            # Main bot orchestrator
├── config/
│   └── config.yaml       # Configuration
├── tests/                # Unit tests
├── data/                 # Data storage
├── logs/                 # Log files
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Strategies Explained

### 1. Trend Following
Identifies major trend changes using moving average crossovers and ADX strength confirmation.

**Indicators**: SMA(50), SMA(200), ADX
**Entry**: Golden Cross + Strong ADX
**Exit**: Death Cross or stop loss

### 2. Momentum
Captures price momentum using RSI and MACD indicators.

**Indicators**: RSI(14), MACD(12,26,9)
**Entry**: RSI oversold + MACD bullish crossover
**Exit**: RSI overbought + MACD bearish crossover

### 3. Mean Reversion
Trades price bounces from Bollinger Bands extremes.

**Indicators**: Bollinger Bands(20,2)
**Entry**: Price touches lower band + reversal
**Exit**: Price touches upper band or mean

### 4. Machine Learning
Uses Random Forest to predict price movements based on technical indicators.

**Features**: 15+ technical indicators + lagged features
**Model**: Random Forest Classifier
**Training**: Historical data with 5-period forward returns

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Returns**: Total, annualized, risk-adjusted
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios
- **Drawdown**: Maximum, average drawdown
- **Trade Stats**: Win rate, profit factor, avg win/loss
- **Volatility**: Daily and annualized

## Risk Management

### Position Sizing Methods
1. **Fixed Fractional**: Fixed percentage of capital
2. **Kelly Criterion**: Optimal sizing based on win rate
3. **Volatility-based**: Size inversely proportional to volatility
4. **Risk Parity**: Based on stop loss distance

### Protection Mechanisms
- Automatic stop loss orders
- Take profit targets
- Daily loss limits
- Maximum position limits
- Real-time P&L tracking

## Supported Exchanges

Via CCXT library, the bot supports 100+ exchanges including:
- Binance
- Coinbase Pro
- Kraken
- Bitfinex
- Bybit
- And many more...

## Safety Features

- **Paper Trading Mode**: Test strategies without risking real money
- **Dry Run**: Simulates orders without actual execution
- **Risk Limits**: Multiple safety mechanisms
- **Logging**: Comprehensive trade and error logging
- **Sandbox Mode**: Use exchange testnet/sandbox environments

## Example Backtest Output

```
==================================================
PERFORMANCE METRICS
==================================================

Returns:
  Total Return:          45.23%
  Annualized Return:     23.45%
  Final Equity:      $14,523.00

Risk-Adjusted Metrics:
  Sharpe Ratio:            1.85
  Sortino Ratio:           2.34
  Calmar Ratio:            3.12

Drawdown:
  Max Drawdown:          -12.34%
  Avg Drawdown:           -3.45%

Trade Statistics:
  Total Trades:              156
  Winning Trades:             92
  Losing Trades:              64
  Win Rate:               58.97%
  Avg Win:            $234.56
  Avg Loss:           $123.45
  Profit Factor:          1.89
```

## Advanced Usage

### Custom Strategy Development

```python
from src.strategies.base_strategy import BaseStrategy, Signal

class MyCustomStrategy(BaseStrategy):
    def __init__(self, params=None):
        super().__init__('MyStrategy', params)

    def calculate_indicators(self, df):
        # Add your indicators
        return df

    def generate_signal(self, df):
        # Your signal logic
        if condition:
            return Signal.BUY
        elif other_condition:
            return Signal.SELL
        return Signal.HOLD
```

### Multi-Strategy Backtesting

```python
from src.strategies import (
    TrendFollowingStrategy,
    MomentumStrategy,
    MeanReversionStrategy
)

strategies = [
    TrendFollowingStrategy(),
    MomentumStrategy(),
    MeanReversionStrategy()
]

weights = [0.4, 0.3, 0.3]  # Must sum to 1.0

results = bot.backtest_engine.run_multi_strategy(
    df=historical_data,
    strategies=strategies,
    weights=weights
)
```

## Warning

**IMPORTANT**: Cryptocurrency trading involves substantial risk of loss. This software is provided for educational purposes. Always test thoroughly in paper trading mode before risking real capital. Past performance does not guarantee future results.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Research based on quantitative trading best practices for 2025
- Technical analysis via `ta` library
- Exchange integration via `ccxt` library
- Machine learning via `scikit-learn`

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Review documentation in `/docs`
- Check example scripts in `/examples`

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

---

**Happy Trading! 🚀📈**
