# Agent-Based Model (ABM) for Bitcoin Price Prediction

## Overview

This implementation adds a sophisticated **Agent-Based Model (ABM)** to the Bitcoin quantitative trading system. The ABM simulates a market populated with heterogeneous investor agents, each with unique characteristics, investment strategies, and capital levels. By observing how these agents interact and respond to market conditions, the system can predict Bitcoin price movements based on collective behavior.

## Key Features

### 🎯 Multi-Agent Simulation
- **93 agents** by default (configurable)
- 6 different agent types with unique behaviors
- Realistic capital distribution across agent types
- Agent-to-agent interactions and sentiment propagation

### 🧠 Intelligent Agent Types

#### 1. **Retail Investors** (50 agents)
- **Characteristics**: Small capital, emotional decision-making
- **Capital Range**: $1,000 - $50,000
- **Behavior**:
  - High social influence (follows the crowd)
  - FOMO buying on upward momentum
  - Panic selling on downward trends
  - Simple moving average strategies
  - Shorter memory (10 periods)

#### 2. **Institutional Investors** (10 agents)
- **Characteristics**: Large capital, sophisticated analysis
- **Capital Range**: $1M - $10M
- **Behavior**:
  - Data-driven decisions
  - Portfolio rebalancing strategies
  - Risk management focused
  - Lower social influence
  - Long memory (50 periods)

#### 3. **Whales** (3 agents)
- **Characteristics**: Extremely large capital, strategic trading
- **Capital Range**: $50M - $500M
- **Behavior**:
  - Accumulate during fear, distribute during greed
  - Contrarian strategies
  - Very low social influence
  - Patient, long-term focus
  - Very long memory (100 periods)

#### 4. **Algorithmic Traders** (15 agents)
- **Characteristics**: Quantitative, unemotional
- **Capital Range**: $100K - $5M
- **Behavior**:
  - Multiple technical indicator signals
  - Kelly Criterion position sizing
  - No social influence
  - Purely data-driven

#### 5. **Momentum Traders** (10 agents)
- **Characteristics**: Trend followers
- **Capital Range**: $50K - $500K
- **Behavior**:
  - Follow strong price momentum
  - Quick to enter/exit
  - Higher risk tolerance
  - Moderate social influence

#### 6. **Contrarian Traders** (5 agents)
- **Characteristics**: Counter-trend traders
- **Capital Range**: $50K - $500K
- **Behavior**:
  - Buy during extreme fear
  - Sell during extreme greed
  - Negative social influence (go against crowd)
  - Patient, opportunistic

## How It Works

### 1. Market Simulation Loop

```
For each time step:
├── Calculate market sentiment from price action
├── Each agent observes:
│   ├── Market data (OHLCV)
│   ├── Market sentiment
│   └── Other agents' actions
├── Agents decide actions (BUY/SELL/HOLD)
├── Agents update sentiment based on:
│   ├── Personal analysis
│   ├── Market sentiment
│   └── Other agents' sentiment
├── Agents execute trades
└── Aggregate predictions from agent behavior
```

### 2. Agent Decision Making

Each agent type uses its own decision-making logic:

- **Technical Analysis**: Moving averages, RSI, Bollinger Bands, volume
- **Sentiment Analysis**: Market fear/greed indicators
- **Social Influence**: Herding behavior, crowd psychology
- **Portfolio Management**: Rebalancing, diversification
- **Risk Management**: Position sizing, capital preservation

### 3. Price Prediction

The ABM predicts price movements by:

1. **Weighted Voting**: Each agent's action is weighted by:
   - Agent type (whales > institutions > algorithmic > retail)
   - Portfolio size
   - Agent confidence

2. **Net Pressure Calculation**:
   ```
   Net Pressure = (Buy Pressure - Sell Pressure) / Total Pressure
   ```

3. **Volatility Adjustment**:
   ```
   Predicted Change = Net Pressure × Market Volatility × 2
   ```

4. **Final Prediction**:
   ```
   Predicted Price = Current Price × (1 + Predicted Change)
   ```

## Configuration

Edit `config/config.yaml` to customize the ABM:

```yaml
# Agent-Based Model Configuration
abm:
  # Number of agents by type
  num_retail: 50           # Retail investors
  num_institutional: 10    # Institutional investors
  num_whales: 3            # Whale investors
  num_algorithmic: 15      # Algorithmic traders
  num_momentum: 10         # Momentum traders
  num_contrarian: 5        # Contrarian traders

  # Capital distribution
  capital_distribution: realistic  # 'realistic' or 'equal'

  # Signal parameters
  signal_threshold: 0.6    # Minimum confidence (0-1) to generate signal

  # Reproducibility
  random_seed: 42          # Set to null for random initialization
```

## Strategy Integration

The ABM is integrated as a trading strategy in the ensemble:

```yaml
strategies:
  active:
    - trend_following
    - momentum
    - mean_reversion
    - abm  # Agent-based model

  weights:
    trend_following: 0.25
    momentum: 0.25
    mean_reversion: 0.2
    abm: 0.3  # Highest weight due to multi-agent sophistication
```

## Usage Examples

### 1. Run the Test Suite

```bash
python test_abm_simple.py
```

This will:
- Create agents of each type
- Run simulation on sample data
- Display agent statistics and predictions

### 2. Use ABM in Backtesting

```python
from src.bot import QuantTrader

# Initialize bot with ABM
bot = QuantTrader(config_path='config/config.yaml')

# Run backtest
results = bot.run_backtest(
    start_date='2023-01-01',
    end_date='2024-12-31'
)
```

### 3. Access ABM Strategy Directly

```python
from src.strategies.abm_strategy import ABMStrategy

# Create ABM strategy
abm = ABMStrategy(params={
    'num_retail': 30,
    'num_institutional': 8,
    'signal_threshold': 0.7,
})

# Generate signal
signal = abm.generate_signal(market_data)
confidence = abm.get_signal_strength(market_data)

# Get agent statistics
stats = abm.get_agent_statistics()
print(stats.groupby('agent_type').agg({
    'total_trades': 'mean',
    'win_rate': 'mean',
}))
```

### 4. Analyze Market Sentiment

```python
# Get current market sentiment
sentiment = abm.get_market_sentiment()
print(f"Market Sentiment: {sentiment:.2f}")  # -1 (bearish) to +1 (bullish)

# Get agent action distribution
actions = abm.get_agent_actions_distribution()
print(f"BUY: {actions['BUY']}, SELL: {actions['SELL']}, HOLD: {actions['HOLD']}")
```

## Architecture

```
src/agents/
├── base_agent.py           # Base agent class with core functionality
├── agent_types.py          # Implementation of 6 agent types
├── abm_simulator.py        # Market simulator managing all agents
└── __init__.py

src/strategies/
└── abm_strategy.py         # ABM strategy for trading bot integration
```

## Agent Attributes

Each agent tracks:

- **Portfolio**:
  - Current capital (cash)
  - BTC holdings
  - Other investments (stocks, bonds, etc.)

- **Personality**:
  - Risk tolerance (0.0 - 1.0)
  - Social influence (how much they follow others)
  - Confidence (in own decisions)

- **Memory**:
  - Price history
  - Action history
  - Trade history

- **Performance**:
  - Total trades
  - Win rate
  - Total P&L

## Advanced Features

### 1. Realistic Capital Distribution

When `capital_distribution: realistic`:
- Whales: $50M - $500M (0.6% of total portfolio in BTC)
- Institutions: $1M - $10M (5-10% in BTC)
- Retail: $1K - $50K (30-70% in BTC)

This mirrors real-world wealth distribution.

### 2. Diversified Portfolios

Agents hold other investments:
- **Retail**: Stocks (60%), Savings (30%), Other Crypto (10%)
- **Institutional**: Stocks (50%), Bonds (30%), Real Estate (15%), Crypto (5%)
- **Whales**: Stocks (40%), Real Estate (30%), Bonds (20%), Crypto (10%)

### 3. Social Dynamics

- **Herding Behavior**: Retail investors follow the crowd
- **Contrarian Behavior**: Contrarians go against the crowd
- **Sentiment Propagation**: Agent sentiment influences others

### 4. Learning and Adaptation

Agents track their performance:
- Win rate
- Profit/Loss
- Confidence adjustment based on success

## Performance Metrics

The ABM provides rich analytics:

```python
# Get detailed agent statistics
stats = simulator.get_agent_statistics()

# Metrics include:
# - total_trades: Number of trades executed
# - winning_trades: Number of profitable trades
# - win_rate: Percentage of winning trades
# - total_pnl: Total profit/loss
# - sentiment: Current sentiment (-1 to +1)
# - confidence: Decision confidence (0 to 1)
# - current_capital: Available cash
# - btc_holdings: BTC amount held
```

## Advantages of ABM

1. **Captures Market Psychology**: Models fear, greed, FOMO, panic
2. **Heterogeneous Agents**: Different strategies, not single model
3. **Emergent Behavior**: Complex patterns from simple agent rules
4. **Realistic Simulation**: Mirrors actual market participants
5. **Interpretable**: Can analyze which agent types drive predictions
6. **Robust**: Multiple perspectives reduce overfitting

## Tuning Tips

### Increase Prediction Accuracy
- Increase `num_algorithmic` for more quantitative signals
- Increase `num_institutional` for better risk management
- Lower `signal_threshold` for more frequent signals

### Reduce False Signals
- Increase `signal_threshold` (e.g., 0.7 or 0.8)
- Reduce `num_retail` (less emotional noise)
- Increase `num_contrarian` for balance

### Faster Execution
- Use `capital_distribution: equal` (simpler calculations)
- Reduce total number of agents
- Set `random_seed` for reproducible results

## Future Enhancements

Potential improvements:

1. **Adaptive Learning**: Agents learn from their mistakes
2. **Network Effects**: Agents form social networks
3. **Order Book Simulation**: Model supply/demand curves
4. **External Factors**: News sentiment, on-chain metrics
5. **Multi-Asset**: Extend to other cryptocurrencies
6. **Real-time Calibration**: Adjust agent parameters from market data

## References

This implementation draws inspiration from:

- **Heterogeneous Agent Models (HAM)** in economics
- **Behavioral Finance** principles
- **Multi-Agent Systems** in AI
- **Market Microstructure** theory

## License

Part of the Bitcoin Quantitative Trading System.

---

**Created**: 2025-01-19
**Author**: Quantrader Development Team
**Version**: 1.0.0
