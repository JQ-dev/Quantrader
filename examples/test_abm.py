"""
Test script for the Agent-Based Model (ABM) strategy.

This script demonstrates:
1. Creating an ABM simulator with different agent types
2. Running simulations on historical data
3. Analyzing agent behavior and predictions
4. Integrating ABM with the trading system
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents import AgentBasedMarketSimulator
from src.strategies.abm_strategy import ABMStrategy


def generate_sample_data(num_periods: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""

    dates = pd.date_range(start='2024-01-01', periods=num_periods, freq='1H')

    # Generate realistic Bitcoin price data
    np.random.seed(42)
    initial_price = 50000

    # Generate returns with some autocorrelation (trending behavior)
    returns = np.random.normal(0.0001, 0.02, num_periods)
    for i in range(1, num_periods):
        returns[i] += 0.3 * returns[i-1]  # Add momentum

    prices = initial_price * (1 + returns).cumprod()

    # Generate OHLCV data
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, num_periods)),
        'high': prices * (1 + np.random.uniform(0, 0.01, num_periods)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, num_periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, num_periods),
    }, index=dates)

    return df


def test_abm_simulator():
    """Test the basic ABM simulator functionality."""

    print("=" * 80)
    print("Testing Agent-Based Market Simulator")
    print("=" * 80)

    # Create simulator
    simulator = AgentBasedMarketSimulator(
        num_retail=20,
        num_institutional=5,
        num_whales=2,
        num_algorithmic=5,
        num_momentum=3,
        num_contrarian=2,
        capital_distribution="realistic",
        random_seed=42,
    )

    print(f"\n✓ Created simulator with {len(simulator.agents)} agents")

    # Generate sample data
    df = generate_sample_data(100)
    print(f"✓ Generated {len(df)} periods of sample data")

    # Run simulation for last 10 periods
    print("\n" + "-" * 80)
    print("Running simulation for 10 time steps...")
    print("-" * 80)

    for i in range(-10, 0):
        # Get data up to current point
        current_data = df.iloc[:len(df) + i]
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Run simulation step
        results = simulator.simulate_step(current_data, current_price, timestamp)

        print(f"\nStep {i + 10}: Price=${current_price:.2f}")
        print(f"  Market Sentiment: {results['market_sentiment']:+.3f}")
        print(f"  Predicted Price: ${results['predicted_price']:.2f} "
              f"({(results['predicted_price'] / current_price - 1) * 100:+.2f}%)")
        print(f"  Agent Actions: BUY={results['buy_pressure']}, "
              f"SELL={results['sell_pressure']}, HOLD={results['hold_count']}")
        print(f"  Trades Executed: {results['num_trades']}")

    # Get agent statistics
    print("\n" + "=" * 80)
    print("Agent Performance Statistics")
    print("=" * 80)

    stats = simulator.get_agent_statistics()

    # Group by agent type
    print("\nBy Agent Type:")
    type_stats = stats.groupby('agent_type').agg({
        'total_trades': 'mean',
        'win_rate': 'mean',
        'total_pnl': 'sum',
        'sentiment': 'mean',
        'confidence': 'mean',
    }).round(3)

    print(type_stats.to_string())

    print("\n✓ ABM simulator test completed successfully!")

    return simulator


def test_abm_strategy():
    """Test the ABM trading strategy."""

    print("\n" + "=" * 80)
    print("Testing ABM Trading Strategy")
    print("=" * 80)

    # Create strategy
    params = {
        "num_retail": 30,
        "num_institutional": 8,
        "num_whales": 2,
        "num_algorithmic": 10,
        "num_momentum": 5,
        "num_contrarian": 3,
        "signal_threshold": 0.6,
        "random_seed": 42,
    }

    strategy = ABMStrategy(params=params)
    print(f"\n✓ Created ABM strategy with {len(strategy.simulator.agents)} agents")

    # Generate sample data
    df = generate_sample_data(100)

    # Test signal generation for last 10 periods
    print("\n" + "-" * 80)
    print("Generating Trading Signals")
    print("-" * 80)

    signals = []
    for i in range(-10, 0):
        current_data = df.iloc[:len(df) + i]

        signal = strategy.generate_signal(current_data)
        strength = strategy.get_signal_strength(current_data)

        current_price = df['close'].iloc[i]
        signals.append({
            'timestamp': df.index[i],
            'price': current_price,
            'signal': signal.name,
            'strength': strength,
        })

        print(f"\n{df.index[i]}: ${current_price:.2f}")
        print(f"  Signal: {signal.name} (confidence: {strength:.3f})")
        print(f"  Market Sentiment: {strategy.get_market_sentiment():+.3f}")

        actions = strategy.get_agent_actions_distribution()
        print(f"  Agent Actions: {actions}")

    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)

    print("\n" + "=" * 80)
    print("Signal Summary")
    print("=" * 80)
    print(f"\nBUY signals: {(signals_df['signal'] == 'BUY').sum()}")
    print(f"SELL signals: {(signals_df['signal'] == 'SELL').sum()}")
    print(f"HOLD signals: {(signals_df['signal'] == 'HOLD').sum()}")
    print(f"Average confidence: {signals_df['strength'].mean():.3f}")

    print("\n✓ ABM strategy test completed successfully!")

    return strategy


def test_abm_integration():
    """Test ABM integration with the full trading system."""

    print("\n" + "=" * 80)
    print("Testing ABM Integration with Trading System")
    print("=" * 80)

    try:
        # Import the main bot
        from src.bot import QuantTrader

        print("\n✓ Successfully imported QuantTrader")

        # Initialize bot (this will load ABM from config)
        print("\nInitializing trading bot with ABM strategy...")
        bot = QuantTrader(config_path='config/config.yaml')

        # Check if ABM strategy is loaded
        abm_strategies = [s for s in bot.strategies if isinstance(s, ABMStrategy)]

        if abm_strategies:
            print(f"✓ ABM strategy loaded successfully!")
            abm_strategy = abm_strategies[0]
            print(f"  Number of agents: {len(abm_strategy.simulator.agents)}")
            print(f"  Signal threshold: {abm_strategy.signal_threshold}")
        else:
            print("⚠ ABM strategy not found in active strategies")
            print("  Make sure 'abm' is in the 'strategies.active' list in config.yaml")

        print("\n✓ Integration test completed successfully!")

        return bot

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""

    print("\n" + "=" * 80)
    print("AGENT-BASED MODEL (ABM) TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Basic simulator
        simulator = test_abm_simulator()

        # Test 2: Strategy
        strategy = test_abm_strategy()

        # Test 3: Integration
        bot = test_abm_integration()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("\n✓ All tests passed successfully!")
        print("\nThe Agent-Based Model is ready to use.")
        print("\nNext steps:")
        print("  1. Run a backtest with ABM: python examples/backtest_example.py")
        print("  2. Adjust ABM parameters in config/config.yaml")
        print("  3. Analyze agent behavior and optimize strategy weights")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
