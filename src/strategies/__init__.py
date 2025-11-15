"""Trading strategy modules"""

from .base_strategy import BaseStrategy, Signal
from .trend_following import TrendFollowingStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'TrendFollowingStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
]
