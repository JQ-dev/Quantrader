"""Example script for running live trading (paper mode)"""

import sys
sys.path.append('..')

from src.bot import QuantTrader


def main():
    """Run live trading in paper mode"""

    print("="*60)
    print("BITCOIN QUANT TRADER - LIVE MODE")
    print("="*60)

    # Initialize trader
    print("\nInitializing trader...")
    bot = QuantTrader(config_path='../config/config.yaml')

    # Verify paper trading mode
    if bot.config['trading']['mode'] != 'paper':
        print("\n⚠️  WARNING: Not in paper trading mode!")
        print("Please set TRADING_MODE=paper in .env for safety")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return

    print(f"\nTrading Mode: {bot.config['trading']['mode'].upper()}")
    print(f"Symbol: {bot.config['trading']['symbol']}")
    print(f"Timeframe: {bot.config['trading']['timeframe']}")
    print(f"Capital: ${bot.config['trading']['capital']:,.2f}")

    print("\nActive Strategies:")
    for strategy in bot.strategies:
        weight = bot.config['strategies']['weights'].get(
            strategy.name.lower().replace(' ', '_'),
            1.0 / len(bot.strategies)
        )
        print(f"  - {strategy.name} (weight: {weight:.2f})")

    print("\nRisk Management:")
    print(f"  Max Position Size: {bot.config['risk_management']['max_position_size']:.1%}")
    print(f"  Stop Loss: {bot.config['risk_management']['stop_loss_pct']:.1%}")
    print(f"  Take Profit: {bot.config['risk_management']['take_profit_pct']:.1%}")
    print(f"  Max Daily Loss: {bot.config['risk_management']['max_daily_loss']:.1%}")

    print("\n" + "="*60)
    print("Starting live trading...")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Run live trading
    # Check every hour (3600 seconds)
    bot.run_live(interval_seconds=3600)


if __name__ == '__main__':
    main()
