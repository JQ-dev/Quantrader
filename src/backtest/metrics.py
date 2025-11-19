"""Performance metrics calculation"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Calculate trading performance metrics"""

    @staticmethod
    def calculate_metrics(equity_curve: pd.DataFrame,
                         trades: pd.DataFrame,
                         initial_capital: float) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity_curve: DataFrame with timestamp and equity columns
            trades: DataFrame with trade information
            initial_capital: Starting capital

        Returns:
            Dictionary with performance metrics
        """
        if equity_curve.empty:
            return PerformanceMetrics._empty_metrics()

        final_equity = equity_curve.iloc[-1]['equity']
        total_return = (final_equity - initial_capital) / initial_capital

        # Returns
        equity_curve = equity_curve.set_index('timestamp') if 'timestamp' in equity_curve.columns else equity_curve
        equity_curve['returns'] = equity_curve['equity'].pct_change()

        # Sharpe Ratio (annualized, assuming daily data)
        mean_return = equity_curve['returns'].mean()
        std_return = equity_curve['returns'].std()
        sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return != 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = equity_curve['returns'][equity_curve['returns'] < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std != 0 else 0

        # Maximum Drawdown
        cumulative = (1 + equity_curve['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        if not trades.empty:
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]

            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())
                           if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0)

            max_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
            max_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0

            # Calculate average trade duration
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                trades['duration'] = pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])
                avg_duration = trades['duration'].mean()
            else:
                avg_duration = pd.Timedelta(0)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_win = 0
            max_loss = 0
            avg_duration = pd.Timedelta(0)

        # Calmar Ratio (return / max drawdown)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        return {
            # Returns
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(equity_curve)) if len(equity_curve) > 0 else 0,
            'final_equity': final_equity,

            # Risk-adjusted metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,

            # Drawdown
            'max_drawdown': max_drawdown,
            'avg_drawdown': drawdown.mean(),

            # Trade statistics
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if not trades.empty else 0,
            'losing_trades': len(losing_trades) if not trades.empty else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_trade_duration': str(avg_duration),

            # Volatility
            'volatility': std_return,
            'annualized_volatility': std_return * np.sqrt(252),
        }

    @staticmethod
    def _empty_metrics() -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'final_equity': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'avg_drawdown': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_win': 0,
            'max_loss': 0,
            'avg_trade_duration': '0',
            'volatility': 0,
            'annualized_volatility': 0,
        }

    @staticmethod
    def print_metrics(metrics: Dict):
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)

        print(f"\nReturns:")
        print(f"  Total Return: {metrics['total_return']:>15.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:>10.2%}")
        print(f"  Final Equity: ${metrics['final_equity']:>14,.2f}")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:>16.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:>15.2f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:>16.2f}")

        print(f"\nDrawdown:")
        print(f"  Max Drawdown: {metrics['max_drawdown']:>16.2%}")
        print(f"  Avg Drawdown: {metrics['avg_drawdown']:>16.2%}")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {metrics['total_trades']:>16}")
        print(f"  Winning Trades: {metrics['winning_trades']:>14}")
        print(f"  Losing Trades: {metrics['losing_trades']:>15}")
        print(f"  Win Rate: {metrics['win_rate']:>20.2%}")
        print(f"  Avg Win: ${metrics['avg_win']:>19,.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:>18,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:>15.2f}")
        print(f"  Max Win: ${metrics['max_win']:>19,.2f}")
        print(f"  Max Loss: ${metrics['max_loss']:>18,.2f}")

        print(f"\nVolatility:")
        print(f"  Daily Volatility: {metrics['volatility']:>12.2%}")
        print(f"  Annualized Volatility: {metrics['annualized_volatility']:>7.2%}")

        print("="*50 + "\n")
