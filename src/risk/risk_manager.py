"""Risk management system"""

from typing import Dict, List
from datetime import datetime, timedelta
from ..utils.logger import setup_logger


class RiskManager:
    """Manage trading risk and enforce risk limits"""

    def __init__(self, config: Dict):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration with:
                - max_position_size: Max position as fraction of capital
                - stop_loss_pct: Stop loss percentage
                - take_profit_pct: Take profit percentage
                - max_daily_loss: Max daily loss as fraction
                - max_open_positions: Maximum concurrent positions
        """
        self.logger = setup_logger(__name__)
        self.config = config

        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().date()
        self.open_positions = []
        self.trade_history = []

        self.logger.info(f"Risk manager initialized with config: {config}")

    def check_position_limit(self) -> bool:
        """
        Check if we can open a new position.

        Returns:
            True if allowed, False otherwise
        """
        max_positions = self.config.get('max_open_positions', 3)

        if len(self.open_positions) >= max_positions:
            self.logger.warning(f"Maximum positions reached ({max_positions})")
            return False

        return True

    def check_daily_loss_limit(self, capital: float) -> bool:
        """
        Check if daily loss limit has been exceeded.

        Args:
            capital: Current capital

        Returns:
            True if trading allowed, False if limit exceeded
        """
        # Reset daily PnL if new day
        current_date = datetime.now().date()
        if current_date > self.daily_reset_time:
            self.logger.info(f"Resetting daily PnL. Previous: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_reset_time = current_date

        max_daily_loss = self.config.get('max_daily_loss', 0.05)
        max_loss_amount = capital * max_daily_loss

        if self.daily_pnl < -max_loss_amount:
            self.logger.error(
                f"Daily loss limit exceeded! PnL: ${self.daily_pnl:.2f}, "
                f"Limit: ${-max_loss_amount:.2f}"
            )
            return False

        return True

    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'

        Returns:
            Stop loss price
        """
        stop_loss_pct = self.config.get('stop_loss_pct', 0.02)

        if position_type == 'long':
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:  # short
            stop_loss = entry_price * (1 + stop_loss_pct)

        return stop_loss

    def calculate_take_profit(self, entry_price: float, position_type: str) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'

        Returns:
            Take profit price
        """
        take_profit_pct = self.config.get('take_profit_pct', 0.04)

        if position_type == 'long':
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # short
            take_profit = entry_price * (1 - take_profit_pct)

        return take_profit

    def check_stop_loss(self, current_price: float) -> List[Dict]:
        """
        Check if any positions hit stop loss.

        Args:
            current_price: Current market price

        Returns:
            List of positions to close
        """
        to_close = []

        for position in self.open_positions:
            if position['type'] == 'long' and current_price <= position['stop_loss']:
                self.logger.warning(
                    f"Stop loss hit for long position! "
                    f"Entry: ${position['entry_price']:.2f}, "
                    f"Current: ${current_price:.2f}"
                )
                to_close.append(position)

            elif position['type'] == 'short' and current_price >= position['stop_loss']:
                self.logger.warning(
                    f"Stop loss hit for short position! "
                    f"Entry: ${position['entry_price']:.2f}, "
                    f"Current: ${current_price:.2f}"
                )
                to_close.append(position)

        return to_close

    def check_take_profit(self, current_price: float) -> List[Dict]:
        """
        Check if any positions hit take profit.

        Args:
            current_price: Current market price

        Returns:
            List of positions to close
        """
        to_close = []

        for position in self.open_positions:
            if position['type'] == 'long' and current_price >= position['take_profit']:
                self.logger.info(
                    f"Take profit hit for long position! "
                    f"Entry: ${position['entry_price']:.2f}, "
                    f"Current: ${current_price:.2f}"
                )
                to_close.append(position)

            elif position['type'] == 'short' and current_price <= position['take_profit']:
                self.logger.info(
                    f"Take profit hit for short position! "
                    f"Entry: ${position['entry_price']:.2f}, "
                    f"Current: ${current_price:.2f}"
                )
                to_close.append(position)

        return to_close

    def add_position(self, position: Dict):
        """Add new position"""
        self.open_positions.append(position)
        self.logger.info(f"Added position: {position}")

    def close_position(self, position: Dict, close_price: float):
        """
        Close position and update PnL.

        Args:
            position: Position to close
            close_price: Closing price
        """
        if position in self.open_positions:
            # Calculate PnL
            if position['type'] == 'long':
                pnl = (close_price - position['entry_price']) * position['size']
            else:  # short
                pnl = (position['entry_price'] - close_price) * position['size']

            self.daily_pnl += pnl

            # Add to history
            trade = {
                **position,
                'close_price': close_price,
                'close_time': datetime.now(),
                'pnl': pnl
            }
            self.trade_history.append(trade)

            # Remove from open positions
            self.open_positions.remove(position)

            self.logger.info(
                f"Closed position. PnL: ${pnl:.2f}, "
                f"Daily PnL: ${self.daily_pnl:.2f}"
            )

    def get_statistics(self) -> Dict:
        """
        Get risk and trading statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
            }

        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]

        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trade_history) if self.trade_history else 0,
            'total_pnl': sum(t['pnl'] for t in self.trade_history),
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': abs(sum(losses) / len(losses)) if losses else 0,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
        }
