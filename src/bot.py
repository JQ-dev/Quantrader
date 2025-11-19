"""Main trading bot orchestrator"""

import yaml
import os
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime
from dotenv import load_dotenv

from .data.data_fetcher import DataFetcher
from .data.data_storage import DataStorage
from .strategies import TrendFollowingStrategy, MomentumStrategy, MeanReversionStrategy
from .strategies.ml_strategy import MLStrategy
from .strategies.abm_strategy import ABMStrategy
from .strategies.base_strategy import BaseStrategy, Signal
from .backtest.engine import BacktestEngine
from .backtest.metrics import PerformanceMetrics
from .risk.risk_manager import RiskManager
from .risk.position_sizing import PositionSizer
from .execution.exchange import ExchangeInterface
from .execution.order_manager import OrderManager, OrderType
from .utils.logger import setup_logger


class QuantTrader:
    """Main Bitcoin Quantitative Trading Bot"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the trading bot.

        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv()

        # Setup logger
        self.logger = setup_logger('QuantTrader')
        self.logger.info("Initializing Bitcoin Quant Trader...")

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self._initialize_components()

        self.logger.info("Quantitative Trader initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override with environment variables if present
            if os.getenv('EXCHANGE'):
                config['exchange']['name'] = os.getenv('EXCHANGE')
            if os.getenv('SYMBOL'):
                config['trading']['symbol'] = os.getenv('SYMBOL')
            if os.getenv('TRADING_MODE'):
                config['trading']['mode'] = os.getenv('TRADING_MODE')

            self.logger.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

    def _initialize_components(self):
        """Initialize all trading components"""
        # Data components
        self.data_fetcher = DataFetcher(
            exchange_name=self.config['exchange']['name'],
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            sandbox=self.config['exchange'].get('sandbox', True)
        )

        self.data_storage = DataStorage(
            cache_dir=self.config['data'].get('cache_dir', './data')
        )

        # Exchange and order execution
        self.exchange = ExchangeInterface(
            exchange_name=self.config['exchange']['name'],
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            sandbox=self.config['exchange'].get('sandbox', True)
        )

        # Dry run mode for paper trading
        dry_run = self.config['trading']['mode'] == 'paper'
        self.order_manager = OrderManager(self.exchange, dry_run=dry_run)

        # Risk management
        self.risk_manager = RiskManager(self.config['risk_management'])
        self.position_sizer = PositionSizer(
            capital=self.config['trading']['capital'],
            max_position_size=self.config['risk_management']['max_position_size']
        )

        # Initialize strategies
        self.strategies = self._initialize_strategies()

        self.logger.info(f"Initialized {len(self.strategies)} strategies")

    def _initialize_strategies(self) -> List[BaseStrategy]:
        """Initialize trading strategies based on configuration"""
        strategies = []
        active_strategies = self.config['strategies']['active']

        if 'trend_following' in active_strategies:
            strategies.append(TrendFollowingStrategy())

        if 'momentum' in active_strategies:
            strategies.append(MomentumStrategy())

        if 'mean_reversion' in active_strategies:
            strategies.append(MeanReversionStrategy())

        if 'ml_strategy' in active_strategies:
            strategies.append(MLStrategy())

        if 'abm' in active_strategies:
            # Get ABM configuration
            abm_params = self.config.get('abm', {})
            strategies.append(ABMStrategy(params=abm_params))

        return strategies

    def run_backtest(self, start_date: str = None, end_date: str = None,
                    save_results: bool = True) -> Dict:
        """
        Run backtest on historical data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_results: Save results to file

        Returns:
            Backtest results
        """
        self.logger.info("Starting backtest...")

        # Use config dates if not provided
        start_date = start_date or self.config['backtesting']['start_date']
        end_date = end_date or self.config['backtesting']['end_date']

        # Fetch historical data
        symbol = self.config['trading']['symbol']
        timeframe = self.config['trading']['timeframe']

        self.logger.info(f"Fetching historical data for {symbol} ({start_date} to {end_date})")

        df = self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            self.logger.error("No historical data available")
            return {}

        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=self.config['backtesting']['initial_capital'],
            commission=self.config['backtesting']['commission'],
            risk_config=self.config['risk_management']
        )

        # Get strategy weights
        weights = []
        for strategy in self.strategies:
            weight = self.config['strategies']['weights'].get(
                strategy.name.lower().replace(' ', '_'),
                1.0 / len(self.strategies)
            )
            weights.append(weight)

        # Run backtest with multiple strategies
        if len(self.strategies) > 1:
            results = engine.run_multi_strategy(df, self.strategies, weights)
        else:
            results = engine.run(df, self.strategies[0])

        # Print metrics
        PerformanceMetrics.print_metrics(results['metrics'])

        # Save results
        if save_results:
            self._save_backtest_results(results)

        return results

    def _save_backtest_results(self, results: Dict):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save equity curve
        equity_path = f"data/backtest_equity_{timestamp}.csv"
        results['equity_curve'].to_csv(equity_path)
        self.logger.info(f"Equity curve saved to {equity_path}")

        # Save trades
        if not results['trades'].empty:
            trades_path = f"data/backtest_trades_{timestamp}.csv"
            results['trades'].to_csv(trades_path, index=False)
            self.logger.info(f"Trades saved to {trades_path}")

    def run_live(self, interval_seconds: int = 3600):
        """
        Run bot in live trading mode.

        Args:
            interval_seconds: Time between trading iterations (default: 1 hour)
        """
        self.logger.info("Starting live trading mode...")
        self.logger.info(f"Trading {self.config['trading']['symbol']} every {interval_seconds}s")

        symbol = self.config['trading']['symbol']
        timeframe = self.config['trading']['timeframe']

        # Main trading loop
        while True:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Trading iteration: {datetime.now()}")
                self.logger.info(f"{'='*50}")

                # Fetch latest market data
                df = self.data_fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=self.config['data']['lookback_periods']
                )

                # Get current price
                current_price = df.iloc[-1]['close']
                self.logger.info(f"Current {symbol} price: ${current_price:.2f}")

                # Check balance
                balance = self.exchange.get_balance('USDT')
                self.logger.info(f"Available balance: ${balance.get('free', 0):.2f} USDT")

                # Generate signals from all strategies
                signals = []
                strengths = []
                weights = []

                for strategy in self.strategies:
                    try:
                        signal = strategy.generate_signal(df)
                        strength = strategy.get_signal_strength(df)

                        signals.append(signal.value)
                        strengths.append(strength)

                        weight = self.config['strategies']['weights'].get(
                            strategy.name.lower().replace(' ', '_'),
                            1.0 / len(self.strategies)
                        )
                        weights.append(weight)

                        self.logger.info(
                            f"{strategy.name}: {signal.name} (strength: {strength:.2f})"
                        )

                    except Exception as e:
                        self.logger.error(f"Error in {strategy.name}: {e}")
                        signals.append(0)
                        strengths.append(0)
                        weights.append(0)

                # Ensemble signal
                weighted_signal = sum(s * w * st for s, w, st in zip(signals, weights, strengths))
                avg_strength = sum(st * w for st, w in zip(strengths, weights))

                self.logger.info(f"Ensemble signal: {weighted_signal:.2f} (strength: {avg_strength:.2f})")

                # Check for existing positions
                current_position = len(self.risk_manager.open_positions) > 0

                # Execute trades based on ensemble signal
                if weighted_signal > 0.3 and not current_position:
                    # BUY signal
                    if self.risk_manager.check_daily_loss_limit(balance.get('total', 0)):
                        self._execute_buy(symbol, current_price, df, avg_strength)

                elif weighted_signal < -0.3 and current_position:
                    # SELL signal - close position
                    self._execute_sell(symbol, current_price)

                # Check stop loss and take profit for open positions
                if current_position:
                    self._check_exits(current_price)

                # Log status
                stats = self.risk_manager.get_statistics()
                self.logger.info(f"Daily PnL: ${stats['daily_pnl']:.2f}")
                self.logger.info(f"Open positions: {stats['open_positions']}")
                self.logger.info(f"Total trades: {stats['total_trades']}, Win rate: {stats['win_rate']:.2%}")

                # Wait for next iteration
                self.logger.info(f"Sleeping for {interval_seconds} seconds...")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                self.logger.info("Stopping trading bot...")
                break

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _execute_buy(self, symbol: str, price: float, df, signal_strength: float):
        """Execute buy order"""
        try:
            # Calculate position size
            volatility = df['volatility'].iloc[-1] if 'volatility' in df else 0.02

            position_size = self.position_sizer.calculate_optimal_size({
                'signal_strength': signal_strength,
                'volatility': volatility,
                'price': price,
                'stop_loss_pct': self.config['risk_management']['stop_loss_pct']
            })

            # Calculate amount in BTC
            amount = position_size / price

            # Execute order
            order = self.order_manager.execute_buy(
                symbol=symbol,
                amount=amount,
                order_type=OrderType.MARKET
            )

            if order:
                # Add to risk manager
                position = {
                    'type': 'long',
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'size': amount,
                    'cost': position_size,
                    'stop_loss': self.risk_manager.calculate_stop_loss(price, 'long'),
                    'take_profit': self.risk_manager.calculate_take_profit(price, 'long'),
                }

                self.risk_manager.add_position(position)

                self.logger.info(
                    f"Bought {amount:.6f} BTC @ ${price:.2f}, "
                    f"SL: ${position['stop_loss']:.2f}, TP: ${position['take_profit']:.2f}"
                )

        except Exception as e:
            self.logger.error(f"Error executing buy: {e}")

    def _execute_sell(self, symbol: str, price: float):
        """Execute sell order"""
        try:
            if not self.risk_manager.open_positions:
                return

            position = self.risk_manager.open_positions[0]
            amount = position['size']

            # Execute order
            order = self.order_manager.execute_sell(
                symbol=symbol,
                amount=amount,
                order_type=OrderType.MARKET
            )

            if order:
                self.risk_manager.close_position(position, price)
                self.logger.info(f"Sold {amount:.6f} BTC @ ${price:.2f}")

        except Exception as e:
            self.logger.error(f"Error executing sell: {e}")

    def _check_exits(self, current_price: float):
        """Check stop loss and take profit exits"""
        # Check stop loss
        stop_positions = self.risk_manager.check_stop_loss(current_price)
        for position in stop_positions:
            self._execute_sell(self.config['trading']['symbol'], current_price)

        # Check take profit
        tp_positions = self.risk_manager.check_take_profit(current_price)
        for position in tp_positions:
            self._execute_sell(self.config['trading']['symbol'], current_price)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Bitcoin Quantitative Trader')
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest',
                       help='Trading mode (backtest or live)')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Trading interval in seconds (for live mode)')

    args = parser.parse_args()

    # Initialize bot
    bot = QuantTrader(config_path=args.config)

    # Run in specified mode
    if args.mode == 'backtest':
        bot.run_backtest()
    else:
        bot.run_live(interval_seconds=args.interval)


if __name__ == '__main__':
    main()
