"""Basic tests for trading strategies"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('..')

from src.strategies import TrendFollowingStrategy, MomentumStrategy, MeanReversionStrategy
from src.strategies.base_strategy import Signal


class TestStrategies(unittest.TestCase):
    """Test trading strategies"""

    def setUp(self):
        """Create sample data for testing"""
        # Generate sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=300, freq='1H')

        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(300) * 100)

        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(300) * 50,
            'high': close_prices + abs(np.random.randn(300) * 100),
            'low': close_prices - abs(np.random.randn(300) * 100),
            'close': close_prices,
            'volume': np.random.randint(100, 1000, 300)
        })

        self.sample_data.set_index('timestamp', inplace=True)

    def test_trend_following_initialization(self):
        """Test TrendFollowingStrategy initialization"""
        strategy = TrendFollowingStrategy()
        self.assertEqual(strategy.name, 'TrendFollowing')
        self.assertIsNotNone(strategy.params)

    def test_momentum_initialization(self):
        """Test MomentumStrategy initialization"""
        strategy = MomentumStrategy()
        self.assertEqual(strategy.name, 'Momentum')
        self.assertIsNotNone(strategy.params)

    def test_mean_reversion_initialization(self):
        """Test MeanReversionStrategy initialization"""
        strategy = MeanReversionStrategy()
        self.assertEqual(strategy.name, 'MeanReversion')
        self.assertIsNotNone(strategy.params)

    def test_calculate_indicators(self):
        """Test indicator calculation"""
        strategy = TrendFollowingStrategy()
        df_with_indicators = strategy.calculate_indicators(self.sample_data)

        # Check that indicators were added
        self.assertIn('sma_50', df_with_indicators.columns)
        self.assertIn('sma_200', df_with_indicators.columns)
        self.assertIn('rsi', df_with_indicators.columns)
        self.assertIn('macd', df_with_indicators.columns)

    def test_signal_generation(self):
        """Test signal generation"""
        strategy = TrendFollowingStrategy()
        signal = strategy.generate_signal(self.sample_data)

        # Signal should be one of BUY, SELL, HOLD
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])

    def test_signal_strength(self):
        """Test signal strength calculation"""
        strategy = MomentumStrategy()
        strength = strategy.get_signal_strength(self.sample_data)

        # Strength should be between 0 and 1
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

    def test_data_validation(self):
        """Test data validation"""
        strategy = TrendFollowingStrategy()

        # Valid data
        self.assertTrue(strategy.validate_data(self.sample_data))

        # Invalid data (empty)
        empty_df = pd.DataFrame()
        self.assertFalse(strategy.validate_data(empty_df))

        # Invalid data (missing columns)
        invalid_df = pd.DataFrame({'close': [1, 2, 3]})
        self.assertFalse(strategy.validate_data(invalid_df))

    def test_multiple_strategies(self):
        """Test running multiple strategies"""
        strategies = [
            TrendFollowingStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy()
        ]

        signals = []
        for strategy in strategies:
            signal = strategy.generate_signal(self.sample_data)
            signals.append(signal)

        # All strategies should return valid signals
        self.assertEqual(len(signals), 3)
        for signal in signals:
            self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])


if __name__ == '__main__':
    unittest.main()
