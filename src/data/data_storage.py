"""Data storage and caching for market data"""

import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from typing import Optional
from ..utils.logger import setup_logger


class DataStorage:
    """Handle data caching and storage"""

    def __init__(self, cache_dir: str = "./data"):
        """
        Initialize data storage.

        Args:
            cache_dir: Directory for cached data
        """
        self.logger = setup_logger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Data storage initialized at {self.cache_dir}")

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for a symbol and timeframe"""
        filename = f"{symbol.replace('/', '_')}_{timeframe}.pkl"
        return self.cache_dir / filename

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save dataframe to cache.

        Args:
            df: DataFrame to save
            symbol: Trading pair
            timeframe: Candle timeframe
        """
        try:
            cache_path = self._get_cache_path(symbol, timeframe)
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            self.logger.info(f"Saved {len(df)} candles to cache: {cache_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def load_data(self, symbol: str, timeframe: str,
                  max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Load dataframe from cache.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            max_age_hours: Maximum cache age in hours

        Returns:
            Cached DataFrame or None if not available/expired
        """
        try:
            cache_path = self._get_cache_path(symbol, timeframe)

            if not cache_path.exists():
                self.logger.debug(f"No cache found for {symbol} {timeframe}")
                return None

            # Check cache age
            file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_age > timedelta(hours=max_age_hours):
                self.logger.info(f"Cache expired for {symbol} {timeframe} (age: {file_age})")
                return None

            with open(cache_path, 'rb') as f:
                df = pickle.load(f)

            self.logger.info(f"Loaded {len(df)} candles from cache: {cache_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None

    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Export dataframe to CSV.

        Args:
            df: DataFrame to export
            filename: Output filename
        """
        try:
            output_path = self.cache_dir / filename
            df.to_csv(output_path)
            self.logger.info(f"Exported data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")

    def load_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load dataframe from CSV.

        Args:
            filename: CSV filename

        Returns:
            DataFrame or None if not found
        """
        try:
            file_path = self.cache_dir / filename
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return None

            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return None
