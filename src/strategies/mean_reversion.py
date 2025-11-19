"""Mean reversion strategy using Bollinger Bands"""

import pandas as pd
from .base_strategy import BaseStrategy, Signal
from ..utils.indicators import TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on:
    - Bollinger Bands for price extremes
    - Price distance from moving average
    - Reversal confirmation
    """

    def __init__(self, params: dict = None):
        """
        Initialize mean reversion strategy.

        Args:
            params: Strategy parameters
                - bb_period: Bollinger Bands period (default: 20)
                - bb_std: Bollinger Bands standard deviation (default: 2)
                - reversal_threshold: Price distance threshold (default: 0.02)
        """
        default_params = {
            'bb_period': 20,
            'bb_std': 2,
            'reversal_threshold': 0.02,  # 2%
        }
        if params:
            default_params.update(params)

        super().__init__('MeanReversion', default_params)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        df = TechnicalIndicators.add_all_indicators(df)
        return df

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate signal based on mean reversion analysis.

        Signal logic:
        - BUY: Price touches or breaks below lower BB + reversal signal
        - SELL: Price touches or breaks above upper BB + reversal signal
        - HOLD: Price within BB or no reversal confirmation
        """
        if not self.validate_data(df):
            return Signal.HOLD

        if len(df) < self.params['bb_period'] + 10:
            self.logger.warning("Insufficient data for mean reversion")
            return Signal.HOLD

        df = self.calculate_indicators(df)

        # Get latest values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        price = current['close']
        prev_price = previous['close']

        bb_upper = current['bb_upper']
        bb_lower = current['bb_lower']
        bb_middle = current['bb_middle']

        rsi = current['rsi']

        # Price position relative to bands
        price_at_lower = price <= bb_lower
        price_at_upper = price >= bb_upper

        # Reversal signals
        bullish_reversal = prev_price < price  # Price starting to go up
        bearish_reversal = prev_price > price  # Price starting to go down

        # Distance from mean
        distance_from_mean = abs(price - bb_middle) / bb_middle

        # RSI confirmation
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70

        # Buy at lower band with reversal
        if price_at_lower and (bullish_reversal or rsi_oversold):
            self.logger.info(f"BUY signal: Price at lower BB ({price:.2f} <= {bb_lower:.2f})")
            return Signal.BUY

        # Sell at upper band with reversal
        elif price_at_upper and (bearish_reversal or rsi_overbought):
            self.logger.info(f"SELL signal: Price at upper BB ({price:.2f} >= {bb_upper:.2f})")
            return Signal.SELL

        # Mean reversion to middle
        elif price < bb_middle * 0.98 and bullish_reversal:
            # Price significantly below mean and reversing
            return Signal.BUY

        elif price > bb_middle * 1.02 and bearish_reversal:
            # Price significantly above mean and reversing
            return Signal.SELL

        else:
            return Signal.HOLD

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on distance from bands"""
        if len(df) < self.params['bb_period'] + 10:
            return 0.0

        df = self.calculate_indicators(df)
        current = df.iloc[-1]

        price = current['close']
        bb_upper = current['bb_upper']
        bb_lower = current['bb_lower']
        bb_middle = current['bb_middle']

        # Calculate how far price is from middle relative to band width
        band_width = bb_upper - bb_lower
        if band_width == 0:
            return 0.0

        distance_from_middle = abs(price - bb_middle)
        strength = min(distance_from_middle / (band_width / 2), 1.0)

        return strength
