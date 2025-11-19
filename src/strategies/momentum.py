"""Momentum strategy using RSI and MACD"""

import pandas as pd
from .base_strategy import BaseStrategy, Signal
from ..utils.indicators import TechnicalIndicators


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy based on:
    - RSI (Relative Strength Index) for overbought/oversold conditions
    - MACD for momentum direction
    - Price momentum
    """

    def __init__(self, params: dict = None):
        """
        Initialize momentum strategy.

        Args:
            params: Strategy parameters
                - rsi_oversold: RSI oversold threshold (default: 30)
                - rsi_overbought: RSI overbought threshold (default: 70)
                - rsi_period: RSI period (default: 14)
        """
        default_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_period': 14,
        }
        if params:
            default_params.update(params)

        super().__init__('Momentum', default_params)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        df = TechnicalIndicators.add_all_indicators(df)
        return df

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate signal based on momentum analysis.

        Signal logic:
        - BUY: RSI oversold + MACD bullish crossover
        - SELL: RSI overbought + MACD bearish crossover
        - HOLD: Otherwise
        """
        if not self.validate_data(df):
            return Signal.HOLD

        if len(df) < 50:
            self.logger.warning("Insufficient data for momentum strategy")
            return Signal.HOLD

        df = self.calculate_indicators(df)

        # Get latest values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        rsi = current['rsi']
        macd = current['macd']
        macd_signal = current['macd_signal']
        prev_macd = previous['macd']
        prev_macd_signal = previous['macd_signal']

        # MACD crossovers
        macd_bullish_cross = (prev_macd <= prev_macd_signal) and (macd > macd_signal)
        macd_bearish_cross = (prev_macd >= prev_macd_signal) and (macd < macd_signal)

        # RSI conditions
        rsi_oversold = rsi < self.params['rsi_oversold']
        rsi_overbought = rsi > self.params['rsi_overbought']

        # MACD momentum
        macd_positive = macd > 0
        macd_negative = macd < 0

        # Combined signals
        if (rsi_oversold or macd_bullish_cross) and macd_positive:
            self.logger.info(f"BUY signal: RSI={rsi:.2f}, MACD bullish")
            return Signal.BUY
        elif (rsi_overbought or macd_bearish_cross) and macd_negative:
            self.logger.info(f"SELL signal: RSI={rsi:.2f}, MACD bearish")
            return Signal.SELL
        elif rsi < 40 and macd > macd_signal:
            # Additional buy opportunity
            return Signal.BUY
        elif rsi > 60 and macd < macd_signal:
            # Additional sell opportunity
            return Signal.SELL
        else:
            return Signal.HOLD

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on RSI extremity and MACD divergence"""
        if len(df) < 50:
            return 0.0

        df = self.calculate_indicators(df)
        current = df.iloc[-1]

        rsi = current['rsi']
        macd_diff = abs(current['macd_diff'])

        # RSI extremity (distance from neutral 50)
        rsi_strength = abs(rsi - 50) / 50.0

        # MACD divergence strength (normalized)
        macd_strength = min(macd_diff / 100.0, 1.0)

        # Combined strength
        strength = (rsi_strength + macd_strength) / 2.0

        return min(strength, 1.0)
