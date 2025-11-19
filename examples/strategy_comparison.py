"""Compare performance of different strategies"""

import sys
sys.path.append('..')

from src.bot import QuantTrader
from src.strategies import TrendFollowingStrategy, MomentumStrategy, MeanReversionStrategy
from src.backtest.engine import BacktestEngine
from src.data.data_fetcher import DataFetcher
import pandas as pd


def main():
    """Compare individual strategy performances"""

    print("Strategy Comparison Backtest\n")

    # Fetch historical data
    print("Fetching historical data...")
    fetcher = DataFetcher(exchange_name='binance', sandbox=True)

    df = fetcher.fetch_historical_data(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date='2023-01-01',
        end_date='2024-12-31'
    )

    print(f"Loaded {len(df)} candles\n")

    # Define strategies
    strategies = [
        TrendFollowingStrategy(),
        MomentumStrategy(),
        MeanReversionStrategy(),
    ]

    # Backtest each strategy
    results = {}

    for strategy in strategies:
        print(f"\nBacktesting {strategy.name}...")
        print("-" * 50)

        engine = BacktestEngine(
            initial_capital=10000,
            commission=0.001,
            risk_config={
                'max_position_size': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'max_daily_loss': 0.05,
                'max_open_positions': 1,
            }
        )

        result = engine.run(df, strategy)
        results[strategy.name] = result

        # Print key metrics
        metrics = result['metrics']
        print(f"Total Return: {metrics['total_return']:>15.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:>16.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:>15.2%}")
        print(f"Win Rate: {metrics['win_rate']:>20.2%}")
        print(f"Total Trades: {metrics['total_trades']:>17}")

    # Comparison table
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    comparison_data = []
    for name, result in results.items():
        m = result['metrics']
        comparison_data.append({
            'Strategy': name,
            'Return': f"{m['total_return']:.2%}",
            'Sharpe': f"{m['sharpe_ratio']:.2f}",
            'Max DD': f"{m['max_drawdown']:.2%}",
            'Win Rate': f"{m['win_rate']:.2%}",
            'Trades': m['total_trades'],
            'Profit Factor': f"{m['profit_factor']:.2f}",
        })

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    # Determine best strategy by Sharpe ratio
    best_strategy = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    print(f"\n🏆 Best Strategy (by Sharpe): {best_strategy[0]}")
    print(f"   Sharpe Ratio: {best_strategy[1]['metrics']['sharpe_ratio']:.2f}")

    # Save comparison
    df_comparison.to_csv('strategy_comparison.csv', index=False)
    print("\nComparison saved to strategy_comparison.csv")


if __name__ == '__main__':
    main()
