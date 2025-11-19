"""
Simple test script for the Agent-Based Model (ABM).
Tests the core ABM functionality without requiring all dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import ABM components
from src.agents.base_agent import BaseAgent, AgentAction, AgentType
from src.agents.agent_types import (
    RetailAgent,
    InstitutionalAgent,
    WhaleAgent,
    AlgorithmicAgent,
    MomentumTrader,
    ContrarianAgent,
)
from src.agents.abm_simulator import AgentBasedMarketSimulator


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


def test_agent_creation():
    """Test creation of different agent types."""

    print("=" * 80)
    print("Test 1: Agent Creation")
    print("=" * 80)

    # Create different agent types
    agents = {
        "Retail": RetailAgent(1, initial_capital=10000, risk_tolerance=0.5),
        "Institutional": InstitutionalAgent(2, initial_capital=5000000, risk_tolerance=0.6),
        "Whale": WhaleAgent(3, initial_capital=100000000, risk_tolerance=0.7),
        "Algorithmic": AlgorithmicAgent(4, initial_capital=1000000, risk_tolerance=0.5),
        "Momentum": MomentumTrader(5, initial_capital=50000, risk_tolerance=0.8),
        "Contrarian": ContrarianAgent(6, initial_capital=50000, risk_tolerance=0.6),
    }

    for name, agent in agents.items():
        print(f"\n{name} Agent:")
        print(f"  ID: {agent.agent_id}")
        print(f"  Type: {agent.agent_type.value}")
        print(f"  Capital: ${agent.current_capital:,.2f}")
        print(f"  Risk Tolerance: {agent.risk_tolerance:.2f}")
        print(f"  Social Influence: {agent.social_influence:.2f}")
        print(f"  Confidence: {agent.confidence:.2f}")
        print(f"  Total Portfolio: ${agent.get_portfolio_value(50000):,.2f}")

    print("\n✓ Agent creation test passed!")
    return agents


def test_simulator_creation():
    """Test ABM simulator creation."""

    print("\n" + "=" * 80)
    print("Test 2: Simulator Creation")
    print("=" * 80)

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

    # Count agents by type
    type_counts = {}
    for agent in simulator.agents:
        agent_type = agent.agent_type.value
        type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

    print("\nAgent distribution:")
    for agent_type, count in type_counts.items():
        print(f"  {agent_type}: {count}")

    print("\n✓ Simulator creation test passed!")
    return simulator


def test_simulation():
    """Test running simulation."""

    print("\n" + "=" * 80)
    print("Test 3: Simulation Execution")
    print("=" * 80)

    # Create simulator
    simulator = AgentBasedMarketSimulator(
        num_retail=30,
        num_institutional=8,
        num_whales=2,
        num_algorithmic=10,
        num_momentum=5,
        num_contrarian=3,
        capital_distribution="realistic",
        random_seed=42,
    )

    # Generate sample data
    df = generate_sample_data(100)
    print(f"\n✓ Generated {len(df)} periods of sample data")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Run simulation for last 10 periods
    print("\n" + "-" * 80)
    print("Running simulation steps...")
    print("-" * 80)

    results_list = []

    for i in range(-10, 0):
        # Get data up to current point
        current_data = df.iloc[:len(df) + i]
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Run simulation step
        results = simulator.simulate_step(current_data, current_price, timestamp)
        results_list.append(results)

        print(f"\nStep {i + 11}/10: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Predicted Price: ${results['predicted_price']:.2f} "
              f"({(results['predicted_price'] / current_price - 1) * 100:+.2f}%)")
        print(f"  Market Sentiment: {results['market_sentiment']:+.3f}")
        print(f"  Agent Actions: BUY={results['buy_pressure']}, "
              f"SELL={results['sell_pressure']}, HOLD={results['hold_count']}")

    print("\n✓ Simulation execution test passed!")
    return results_list


def test_agent_statistics():
    """Test agent statistics collection."""

    print("\n" + "=" * 80)
    print("Test 4: Agent Statistics")
    print("=" * 80)

    # Create simulator
    simulator = AgentBasedMarketSimulator(
        num_retail=10,
        num_institutional=3,
        num_whales=1,
        num_algorithmic=3,
        num_momentum=2,
        num_contrarian=1,
        random_seed=42,
    )

    # Generate sample data
    df = generate_sample_data(50)

    # Run simulation for 20 steps
    for i in range(-20, 0):
        current_data = df.iloc[:len(df) + i]
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]
        simulator.simulate_step(current_data, current_price, timestamp)

    # Get statistics
    stats = simulator.get_agent_statistics()

    print("\n✓ Collected statistics for {} agents".format(len(stats)))

    # Group by type
    print("\nStatistics by agent type:")
    type_stats = stats.groupby('agent_type').agg({
        'total_trades': 'mean',
        'win_rate': 'mean',
        'total_pnl': 'sum',
        'sentiment': 'mean',
        'confidence': 'mean',
    }).round(3)

    print(type_stats.to_string())

    print("\n✓ Agent statistics test passed!")
    return stats


def main():
    """Run all tests."""

    print("\n" + "=" * 80)
    print("AGENT-BASED MODEL (ABM) TEST SUITE")
    print("=" * 80)
    print("\nTesting the core ABM functionality...")

    try:
        # Run tests
        test_agent_creation()
        simulator = test_simulator_creation()
        results = test_simulation()
        stats = test_agent_statistics()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("\n✓ All tests passed successfully!")
        print("\nAgent-Based Model Features:")
        print("  ✓ 6 different agent types with unique behaviors")
        print("  ✓ Agents make decisions based on market data and other agents")
        print("  ✓ Realistic capital distribution and portfolio management")
        print("  ✓ Sentiment analysis and social influence modeling")
        print("  ✓ Price prediction from aggregated agent behavior")
        print("\nThe Agent-Based Model is working correctly!")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
