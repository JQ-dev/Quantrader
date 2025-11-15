"""Backtesting engine for strategy evaluation"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from ..strategies.base_strategy import BaseStrategy, Signal
from ..risk.risk_manager import RiskManager
from ..risk.position_sizing import PositionSizer
from .metrics import PerformanceMetrics
from ..utils.logger import setup_logger


class BacktestEngine:
    """Backtesting engine for trading strategies"""

    def __init__(self, initial_capital: float = 10000,
                 commission: float = 0.001,
                 risk_config: Dict = None):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Trading commission (0.001 = 0.1%)
            risk_config: Risk management configuration
        """
        self.logger = setup_logger(__name__)
        self.initial_capital = initial_capital
        self.commission = commission

        # Risk management
        risk_config = risk_config or {
            'max_position_size': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_daily_loss': 0.05,
            'max_open_positions': 3,
        }
        self.risk_manager = RiskManager(risk_config)
        self.position_sizer = PositionSizer(initial_capital, risk_config['max_position_size'])

        # Performance tracking
        self.equity_curve = []
        self.trades = []

        self.logger.info(
            f"Backtest engine initialized with ${initial_capital:.2f} capital, "
            f"{commission*100:.2f}% commission"
        )

    def run(self, df: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            strategy: Trading strategy to test

        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Running backtest for {strategy.name}")
        self.logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
        self.logger.info(f"Total candles: {len(df)}")

        # Reset state
        capital = self.initial_capital
        position = None  # Current position
        equity = capital

        self.equity_curve = []
        self.trades = []

        # Calculate indicators once
        df_with_indicators = strategy.calculate_indicators(df.copy())

        # Iterate through data
        for i in range(len(df_with_indicators)):
            current_data = df_with_indicators.iloc[:i+1]

            # Skip if insufficient data
            if len(current_data) < 50:
                continue

            current_price = current_data.iloc[-1]['close']
            current_time = current_data.index[-1]

            # Check existing position for stop loss / take profit
            if position:
                # Check stop loss
                stop_positions = self.risk_manager.check_stop_loss(current_price)
                if stop_positions:
                    pnl = self._close_position(position, current_price, current_time, 'stop_loss')
                    capital += pnl
                    equity = capital
                    position = None
                    continue

                # Check take profit
                tp_positions = self.risk_manager.check_take_profit(current_price)
                if tp_positions:
                    pnl = self._close_position(position, current_price, current_time, 'take_profit')
                    capital += pnl
                    equity = capital
                    position = None
                    continue

            # Generate signal
            try:
                signal = strategy.generate_signal(current_data)
                signal_strength = strategy.get_signal_strength(current_data)
            except Exception as e:
                self.logger.error(f"Error generating signal at {current_time}: {e}")
                continue

            # Execute trades based on signals
            if signal == Signal.BUY and position is None:
                # Check risk limits
                if not self.risk_manager.check_daily_loss_limit(capital):
                    continue

                if not self.risk_manager.check_position_limit():
                    continue

                # Calculate position size
                volatility = current_data['volatility'].iloc[-1] if 'volatility' in current_data else 0.02
                position_size = self.position_sizer.calculate_optimal_size({
                    'signal_strength': signal_strength,
                    'volatility': volatility,
                    'price': current_price,
                    'stop_loss_pct': self.risk_manager.config['stop_loss_pct']
                })

                # Ensure we don't exceed capital
                position_size = min(position_size, capital * 0.95)  # Keep some cash

                # Calculate units (BTC)
                units = position_size / current_price
                cost = units * current_price
                commission_cost = cost * self.commission

                if cost + commission_cost <= capital:
                    # Open long position
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'size': units,
                        'cost': cost + commission_cost,
                        'stop_loss': self.risk_manager.calculate_stop_loss(current_price, 'long'),
                        'take_profit': self.risk_manager.calculate_take_profit(current_price, 'long'),
                    }

                    capital -= (cost + commission_cost)
                    self.risk_manager.add_position(position)

                    self.logger.debug(
                        f"{current_time}: LONG {units:.6f} BTC @ ${current_price:.2f}, "
                        f"SL: ${position['stop_loss']:.2f}, TP: ${position['take_profit']:.2f}"
                    )

            elif signal == Signal.SELL and position is not None and position['type'] == 'long':
                # Close long position
                pnl = self._close_position(position, current_price, current_time, 'signal')
                capital += pnl
                position = None

            # Update equity (capital + unrealized PnL)
            if position:
                if position['type'] == 'long':
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    equity = capital + position['cost'] + unrealized_pnl
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                    equity = capital + position['cost'] + unrealized_pnl
            else:
                equity = capital

            # Track equity curve
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'capital': capital,
            })

        # Close any remaining position at end
        if position:
            final_price = df_with_indicators.iloc[-1]['close']
            final_time = df_with_indicators.index[-1]
            pnl = self._close_position(position, final_price, final_time, 'end_of_data')
            capital += pnl

        # Calculate performance metrics
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        metrics = PerformanceMetrics.calculate_metrics(
            equity_df,
            trades_df,
            self.initial_capital
        )

        self.logger.info(f"Backtest completed. Final equity: ${capital:.2f}")
        self.logger.info(f"Total return: {metrics['total_return']:.2%}")
        self.logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        self.logger.info(f"Win rate: {metrics['win_rate']:.2%}")

        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'strategy': strategy.name,
        }

    def _close_position(self, position: Dict, close_price: float,
                       close_time: datetime, reason: str) -> float:
        """
        Close a position and calculate PnL.

        Args:
            position: Position dict
            close_price: Closing price
            close_time: Closing time
            reason: Reason for closing

        Returns:
            PnL including commission
        """
        units = position['size']
        proceeds = units * close_price
        commission_cost = proceeds * self.commission

        if position['type'] == 'long':
            pnl = proceeds - position['cost'] - commission_cost
        else:  # short
            pnl = position['cost'] - proceeds - commission_cost

        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': close_time,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': close_price,
            'size': units,
            'pnl': pnl,
            'return': pnl / position['cost'],
            'reason': reason,
        }

        self.trades.append(trade)
        self.risk_manager.close_position(position, close_price)

        self.logger.debug(
            f"{close_time}: CLOSE {position['type'].upper()} {units:.6f} BTC @ "
            f"${close_price:.2f}, PnL: ${pnl:.2f} ({reason})"
        )

        return proceeds - commission_cost

    def run_multi_strategy(self, df: pd.DataFrame,
                          strategies: List[BaseStrategy],
                          weights: List[float] = None) -> Dict[str, Any]:
        """
        Run backtest with multiple strategies (ensemble).

        Args:
            df: DataFrame with OHLCV data
            strategies: List of strategies
            weights: Strategy weights (must sum to 1.0)

        Returns:
            Backtest results
        """
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)

        self.logger.info(f"Running multi-strategy backtest with {len(strategies)} strategies")

        # Calculate indicators for all strategies
        df_with_indicators = df.copy()
        for strategy in strategies:
            df_with_indicators = strategy.calculate_indicators(df_with_indicators)

        # Similar to single strategy but combine signals
        capital = self.initial_capital
        position = None
        equity = capital

        self.equity_curve = []
        self.trades = []

        for i in range(len(df_with_indicators)):
            current_data = df_with_indicators.iloc[:i+1]

            if len(current_data) < 50:
                continue

            current_price = current_data.iloc[-1]['close']
            current_time = current_data.index[-1]

            # Check position exits
            if position:
                stop_positions = self.risk_manager.check_stop_loss(current_price)
                tp_positions = self.risk_manager.check_take_profit(current_price)

                if stop_positions or tp_positions:
                    reason = 'stop_loss' if stop_positions else 'take_profit'
                    pnl = self._close_position(position, current_price, current_time, reason)
                    capital += pnl
                    equity = capital
                    position = None
                    continue

            # Ensemble signal
            signals = []
            strengths = []

            for strategy in strategies:
                try:
                    sig = strategy.generate_signal(current_data)
                    strength = strategy.get_signal_strength(current_data)
                    signals.append(sig.value)
                    strengths.append(strength)
                except:
                    signals.append(0)
                    strengths.append(0)

            # Weighted average signal
            weighted_signal = sum(s * w * st for s, w, st in zip(signals, weights, strengths))
            avg_strength = sum(st * w for st, w in zip(strengths, weights))

            # Execute based on ensemble signal
            if weighted_signal > 0.3 and position is None:
                if self.risk_manager.check_daily_loss_limit(capital) and \
                   self.risk_manager.check_position_limit():

                    volatility = current_data['volatility'].iloc[-1] if 'volatility' in current_data else 0.02
                    position_size = self.position_sizer.calculate_optimal_size({
                        'signal_strength': avg_strength,
                        'volatility': volatility,
                        'price': current_price,
                        'stop_loss_pct': self.risk_manager.config['stop_loss_pct']
                    })

                    position_size = min(position_size, capital * 0.95)
                    units = position_size / current_price
                    cost = units * current_price
                    commission_cost = cost * self.commission

                    if cost + commission_cost <= capital:
                        position = {
                            'type': 'long',
                            'entry_price': current_price,
                            'entry_time': current_time,
                            'size': units,
                            'cost': cost + commission_cost,
                            'stop_loss': self.risk_manager.calculate_stop_loss(current_price, 'long'),
                            'take_profit': self.risk_manager.calculate_take_profit(current_price, 'long'),
                        }
                        capital -= (cost + commission_cost)
                        self.risk_manager.add_position(position)

            elif weighted_signal < -0.3 and position is not None:
                pnl = self._close_position(position, current_price, current_time, 'signal')
                capital += pnl
                position = None

            # Update equity
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['size']
                equity = capital + position['cost'] + unrealized_pnl
            else:
                equity = capital

            self.equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'capital': capital,
            })

        # Close remaining position
        if position:
            final_price = df_with_indicators.iloc[-1]['close']
            final_time = df_with_indicators.index[-1]
            pnl = self._close_position(position, final_price, final_time, 'end_of_data')
            capital += pnl

        # Calculate metrics
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        metrics = PerformanceMetrics.calculate_metrics(equity_df, trades_df, self.initial_capital)

        self.logger.info(f"Multi-strategy backtest completed. Final equity: ${capital:.2f}")

        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'strategies': [s.name for s in strategies],
            'weights': weights,
        }
