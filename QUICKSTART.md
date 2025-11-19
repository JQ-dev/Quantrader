# Quick Start Guide

Get started with Bitcoin Quant Trader in 5 minutes!

## Step 1: Installation

```bash
# Clone and install dependencies
git clone <repository-url>
cd Quantrader
pip install -r requirements.txt
```

## Step 2: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env (use paper mode for testing)
# Set TRADING_MODE=paper
```

## Step 3: Run Your First Backtest

```bash
# Simple backtest using default configuration
python -m src.bot --mode backtest
```

This will:
- Fetch BTC/USDT historical data
- Run multiple strategies (Trend Following, Momentum, Mean Reversion)
- Calculate comprehensive performance metrics
- Save results to `data/` directory

## Step 4: Analyze Results

The backtest will print metrics like:

```
Total Return: 45.23%
Sharpe Ratio: 1.85
Max Drawdown: -12.34%
Win Rate: 58.97%
```

## Step 5: Try Paper Trading (Optional)

```bash
# Run in paper trading mode (simulated trades)
python -m src.bot --mode live --interval 3600
```

This will:
- Check markets every hour (3600 seconds)
- Generate trading signals
- Simulate order execution (no real money)
- Track performance in real-time

## Understanding the Strategies

### Trend Following
- Detects major trend changes
- Uses Golden Cross (50/200 MA)
- Best for strong trends

### Momentum
- Captures price momentum
- Uses RSI + MACD
- Good for volatile markets

### Mean Reversion
- Trades extremes
- Uses Bollinger Bands
- Works in ranging markets

## Customization

### Change Trading Pair
Edit `config/config.yaml`:
```yaml
trading:
  symbol: ETH/USDT  # Change to any pair
```

### Adjust Risk
Edit `config/config.yaml`:
```yaml
risk_management:
  max_position_size: 0.05  # 5% instead of 10%
  stop_loss_pct: 0.01      # 1% stop loss
```

### Select Strategies
Edit `config/config.yaml`:
```yaml
strategies:
  active:
    - trend_following  # Only use trend following
```

## Example Scripts

Run pre-made examples:

```bash
# Compare all strategies
cd examples
python strategy_comparison.py

# Run detailed backtest
python run_backtest.py

# Paper trading demo
python run_live_trading.py
```

## Next Steps

1. **Optimize Parameters**: Adjust strategy parameters in `config/config.yaml`
2. **Add Strategies**: Create custom strategies in `src/strategies/`
3. **Longer Backtests**: Test on more historical data
4. **Multi-Asset**: Configure multiple trading pairs
5. **Live Trading**: When ready, switch to `TRADING_MODE=live` (use caution!)

## Common Issues

### "No module named 'src'"
Run from project root directory

### "Insufficient data"
Increase `lookback_periods` in config or use longer timeframe

### "API Error"
Check API credentials in `.env` and ensure exchange is accessible

## Safety Tips

✅ **Always start with paper trading**
✅ **Test thoroughly on historical data**
✅ **Start with small capital**
✅ **Use stop losses**
✅ **Monitor daily loss limits**

❌ **Never invest more than you can lose**
❌ **Don't use live mode without testing**
❌ **Don't disable risk management**

## Getting Help

- Check README.md for detailed documentation
- Review example scripts in `examples/`
- Check logs in `logs/trader.log`

---

**Ready to trade? Start with backtesting and paper trading first!** 🚀
