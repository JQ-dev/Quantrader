"""Position sizing calculations"""

import numpy as np
from typing import Dict
from ..utils.logger import setup_logger


class PositionSizer:
    """Calculate position sizes based on various methods"""

    def __init__(self, capital: float, max_position_size: float = 0.1):
        """
        Initialize position sizer.

        Args:
            capital: Total trading capital
            max_position_size: Maximum position size as fraction of capital
        """
        self.logger = setup_logger(__name__)
        self.capital = capital
        self.max_position_size = max_position_size

    def update_capital(self, new_capital: float):
        """Update trading capital"""
        self.capital = new_capital
        self.logger.info(f"Capital updated to ${new_capital:.2f}")

    def fixed_fractional(self, signal_strength: float = 1.0) -> float:
        """
        Fixed fractional position sizing.

        Args:
            signal_strength: Signal confidence (0-1)

        Returns:
            Position size in base currency
        """
        position_size = self.capital * self.max_position_size * signal_strength
        return position_size

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion position sizing.
        Kelly% = W - [(1-W) / R]
        Where W = win rate, R = avg_win/avg_loss

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive value)

        Returns:
            Position size in base currency
        """
        if avg_loss == 0:
            return self.fixed_fractional()

        R = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / R)

        # Use half Kelly for safety
        kelly_pct = max(0, min(kelly_pct * 0.5, self.max_position_size))

        position_size = self.capital * kelly_pct
        self.logger.debug(f"Kelly position size: ${position_size:.2f} ({kelly_pct:.2%})")

        return position_size

    def volatility_based(self, volatility: float, target_risk: float = 0.02) -> float:
        """
        Volatility-based position sizing.
        Position size inversely proportional to volatility.

        Args:
            volatility: Asset volatility (standard deviation of returns)
            target_risk: Target risk per trade as fraction of capital

        Returns:
            Position size in base currency
        """
        if volatility == 0:
            return self.fixed_fractional()

        # Calculate position size to achieve target risk
        position_size = (self.capital * target_risk) / volatility

        # Cap at max position size
        max_position = self.capital * self.max_position_size
        position_size = min(position_size, max_position)

        return position_size

    def risk_parity(self, price: float, stop_loss_distance: float,
                   risk_per_trade: float = 0.02) -> float:
        """
        Risk parity position sizing.
        Size position based on stop loss distance.

        Args:
            price: Current asset price
            stop_loss_distance: Distance to stop loss in price units
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Position size in number of units
        """
        if stop_loss_distance == 0:
            return 0

        # Amount willing to risk
        risk_amount = self.capital * risk_per_trade

        # Position size = risk amount / stop loss distance
        position_size = risk_amount / stop_loss_distance

        # Convert to base currency value
        position_value = position_size * price

        # Cap at max position size
        max_position = self.capital * self.max_position_size
        if position_value > max_position:
            position_size = max_position / price

        return position_size

    def calculate_optimal_size(self, params: Dict) -> float:
        """
        Calculate optimal position size using multiple methods.

        Args:
            params: Dictionary with:
                - signal_strength: Signal confidence
                - volatility: Asset volatility
                - price: Current price
                - stop_loss_pct: Stop loss percentage

        Returns:
            Optimal position size in base currency
        """
        sizes = []

        # Fixed fractional
        sizes.append(self.fixed_fractional(params.get('signal_strength', 1.0)))

        # Volatility-based
        if 'volatility' in params:
            sizes.append(self.volatility_based(params['volatility']))

        # Risk parity
        if 'price' in params and 'stop_loss_pct' in params:
            stop_loss_distance = params['price'] * params['stop_loss_pct']
            size_units = self.risk_parity(params['price'], stop_loss_distance)
            sizes.append(size_units * params['price'])

        # Take minimum for conservative approach
        optimal_size = min(sizes) if sizes else self.fixed_fractional()

        self.logger.info(f"Optimal position size: ${optimal_size:.2f}")
        return optimal_size
