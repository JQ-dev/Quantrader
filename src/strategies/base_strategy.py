"""Base strategy class for all trading strategies"""

from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
from typing import Dict, Any
from ..utils.logger import setup_logger


class Signal(Enum):
    """Trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.logger = setup_logger(f"Strategy.{name}")
        self.logger.info(f"Initialized {name} strategy with params: {params}")

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate trading signal based on market data.

        Args:
            df: DataFrame with OHLCV and indicator data

        Returns:
            Signal (BUY, SELL, or HOLD)
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicators
        """
        pass

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate signal strength/confidence (0-1).

        Args:
            df: DataFrame with market data

        Returns:
            Signal strength (0.0 to 1.0)
        """
        return 0.5  # Default neutral strength

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that dataframe has required data.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False

        if len(df) < 1:
            self.logger.error("DataFrame is empty")
            return False

        return True

    def __str__(self):
        return f"{self.name}(params={self.params})"
