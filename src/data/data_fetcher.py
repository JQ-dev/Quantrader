"""Market data fetching using CCXT"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time
from ..utils.logger import setup_logger


class DataFetcher:
    """Fetch cryptocurrency market data from exchanges"""

    def __init__(self, exchange_name: str = 'binance', api_key: str = None,
                 api_secret: str = None, sandbox: bool = True):
        """
        Initialize data fetcher.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
            api_key: API key for authenticated requests
            api_secret: API secret for authenticated requests
            sandbox: Use sandbox/testnet mode
        """
        self.logger = setup_logger(__name__)
        self.exchange_name = exchange_name

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        config = {
            'enableRateLimit': True,
        }

        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret

        if sandbox:
            config['options'] = {'defaultType': 'future'}

        self.exchange = exchange_class(config)

        if sandbox and hasattr(self.exchange, 'set_sandbox_mode'):
            self.exchange.set_sandbox_mode(True)

        self.logger.info(f"Initialized {exchange_name} exchange (sandbox: {sandbox})")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h',
                    limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            since: Timestamp in milliseconds to fetch from

        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching {symbol} {timeframe} data (limit: {limit})")

            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            self.logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")
            raise

    def fetch_historical_data(self, symbol: str, timeframe: str,
                             start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data for a date range.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with historical OHLCV data
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []
        current = start

        # Timeframe to milliseconds
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000

        while current < end:
            since = int(current.timestamp() * 1000)

            try:
                df = self.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)

                if df.empty:
                    break

                all_data.append(df)

                # Move to next batch
                current = df.index[-1].to_pydatetime() + timedelta(milliseconds=timeframe_ms)

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                self.logger.error(f"Error fetching batch starting {current}: {e}")
                break

        if all_data:
            result = pd.concat(all_data)
            result = result[~result.index.duplicated(keep='first')]
            result = result.sort_index()

            # Filter to exact date range
            result = result[(result.index >= start) & (result.index <= end)]

            self.logger.info(f"Fetched {len(result)} total candles")
            return result
        else:
            return pd.DataFrame()

    def get_ticker(self, symbol: str) -> dict:
        """
        Get current ticker information.

        Args:
            symbol: Trading pair

        Returns:
            Ticker dictionary
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.logger.debug(f"Ticker for {symbol}: {ticker['last']}")
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            raise

    def get_account_balance(self) -> dict:
        """
        Get account balance.

        Returns:
            Balance dictionary
        """
        try:
            balance = self.exchange.fetch_balance()
            self.logger.info(f"Account balance fetched")
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
