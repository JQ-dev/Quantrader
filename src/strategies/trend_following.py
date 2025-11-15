"""Trend-following strategy using moving averages and ADX"""

import pandas as pd
from .base_strategy import BaseStrategy, Signal
from ..utils.indicators import TechnicalIndicators


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy based on:
    - Moving average crossovers (Golden Cross / Death Cross)
    - ADX for trend strength
    - Price position relative to moving averages
    """

    def __init__(self, params: dict = None):
        """
        Initialize trend-following strategy.

        Args:
            params: Strategy parameters
                - fast_ma: Fast MA period (default: 50)
                - slow_ma: Slow MA period (default: 200)
                - adx_threshold: ADX threshold for strong trend (default: 25)
        """
        default_params = {
            'fast_ma': 50,
            'slow_ma': 200,
            'adx_threshold': 25,
        }
        if params:
            default_params.update(params)

        super().__init__('TrendFollowing', default_params)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        df = TechnicalIndicators.add_all_indicators(df)
        return df

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate signal based on trend analysis.

        Signal logic:
        - BUY: Fast MA crosses above slow MA + ADX shows strong trend
        - SELL: Fast MA crosses below slow MA + ADX shows strong trend
        - HOLD: Otherwise
        """
        if not self.validate_data(df):
            return Signal.HOLD

        # Need at least slow_ma + 1 candles
        if len(df) < self.params['slow_ma'] + 5:
            self.logger.warning("Insufficient data for trend following")
            return Signal.HOLD

        df = self.calculate_indicators(df)

        # Get latest values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        fast_ma = current['sma_50']
        slow_ma = current['sma_200']
        prev_fast_ma = previous['sma_50']
        prev_slow_ma = previous['sma_200']

        adx = current['adx']
        current_price = current['close']

        # Golden Cross: Fast MA crosses above slow MA
        golden_cross = (prev_fast_ma <= prev_slow_ma) and (fast_ma > slow_ma)

        # Death Cross: Fast MA crosses below slow MA
        death_cross = (prev_fast_ma >= prev_slow_ma) and (fast_ma < slow_ma)

        # Strong trend confirmation
        strong_trend = adx > self.params['adx_threshold']

        # Additional confirmation: price above both MAs for buy
        price_above_mas = current_price > fast_ma and current_price > slow_ma
        price_below_mas = current_price < fast_ma and current_price < slow_ma

        if golden_cross and strong_trend:
            self.logger.info(f"BUY signal: Golden Cross detected (ADX: {adx:.2f})")
            return Signal.BUY
        elif death_cross and strong_trend:
            self.logger.info(f"SELL signal: Death Cross detected (ADX: {adx:.2f})")
            return Signal.SELL
        elif price_above_mas and fast_ma > slow_ma and strong_trend:
            # Strong uptrend continuation
            return Signal.BUY
        elif price_below_mas and fast_ma < slow_ma and strong_trend:
            # Strong downtrend continuation
            return Signal.SELL
        else:
            return Signal.HOLD

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on ADX and trend alignment"""
        if len(df) < self.params['slow_ma'] + 5:
            return 0.0

        df = self.calculate_indicators(df)
        current = df.iloc[-1]

        adx = current['adx']
        # Normalize ADX to 0-1 range (ADX typically ranges 0-100)
        strength = min(adx / 100.0, 1.0)

        return strength
