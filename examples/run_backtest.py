"""Example script for running backtests"""

import sys
sys.path.append('..')

from src.bot import QuantTrader
from src.backtest.metrics import PerformanceMetrics
import matplotlib.pyplot as plt


def main():
    """Run a comprehensive backtest"""

    # Initialize trader
    print("Initializing Bitcoin Quant Trader...")
    bot = QuantTrader(config_path='../config/config.yaml')

    # Run backtest
    print("\nRunning backtest...")
    results = bot.run_backtest(
        start_date='2023-01-01',
        end_date='2024-12-31',
        save_results=True
    )

    # Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    metrics = results['metrics']

    print(f"\nInitial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")

    print(f"\nRisk Metrics:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    print(f"\nTrade Statistics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    # Plot equity curve
    print("\nGenerating equity curve plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'].index, results['equity_curve']['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curve.png')
    print("Equity curve saved to equity_curve.png")

    # Print detailed metrics
    PerformanceMetrics.print_metrics(metrics)

    # Trade analysis
    if not results['trades'].empty:
        print("\nTrade Analysis:")
        print(f"Average trade duration: {metrics['avg_trade_duration']}")
        print(f"Best trade: ${metrics['max_win']:.2f}")
        print(f"Worst trade: ${metrics['max_loss']:.2f}")

        # Save trades to CSV
        results['trades'].to_csv('trades.csv', index=False)
        print("\nTrades saved to trades.csv")


if __name__ == '__main__':
    main()
