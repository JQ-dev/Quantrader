"""Exchange interface for order execution"""

import ccxt
from typing import Dict, Optional
from ..utils.logger import setup_logger


class ExchangeInterface:
    """Interface for cryptocurrency exchange operations"""

    def __init__(self, exchange_name: str = 'binance',
                 api_key: str = None, api_secret: str = None,
                 sandbox: bool = True):
        """
        Initialize exchange interface.

        Args:
            exchange_name: Exchange name (binance, coinbase, etc.)
            api_key: API key
            api_secret: API secret
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

        self.logger.info(f"Exchange interface initialized: {exchange_name} (sandbox: {sandbox})")

    def get_balance(self, currency: str = None) -> Dict:
        """
        Get account balance.

        Args:
            currency: Specific currency (optional)

        Returns:
            Balance dictionary
        """
        try:
            balance = self.exchange.fetch_balance()

            if currency:
                return {
                    'free': balance['free'].get(currency, 0),
                    'used': balance['used'].get(currency, 0),
                    'total': balance['total'].get(currency, 0),
                }
            return balance

        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def create_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """
        Create a market order.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Amount in base currency

        Returns:
            Order info or None if failed
        """
        try:
            self.logger.info(f"Creating market {side} order: {amount} {symbol}")

            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount
            )

            self.logger.info(f"Order created: {order['id']}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating market order: {e}")
            return None

    def create_limit_order(self, symbol: str, side: str,
                          amount: float, price: float) -> Optional[Dict]:
        """
        Create a limit order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Amount in base currency
            price: Limit price

        Returns:
            Order info or None if failed
        """
        try:
            self.logger.info(
                f"Creating limit {side} order: {amount} {symbol} @ ${price}"
            )

            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price
            )

            self.logger.info(f"Order created: {order['id']}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating limit order: {e}")
            return None

    def create_stop_loss_order(self, symbol: str, side: str,
                              amount: float, stop_price: float) -> Optional[Dict]:
        """
        Create a stop loss order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Amount in base currency
            stop_price: Stop loss trigger price

        Returns:
            Order info or None if failed
        """
        try:
            self.logger.info(
                f"Creating stop loss {side} order: {amount} {symbol} @ ${stop_price}"
            )

            # Different exchanges have different stop loss implementations
            if self.exchange_name == 'binance':
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_LOSS_LIMIT',
                    side=side,
                    amount=amount,
                    price=stop_price,
                    params={'stopPrice': stop_price}
                )
            else:
                # Generic implementation
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=side,
                    amount=amount,
                    params={'stopPrice': stop_price}
                )

            self.logger.info(f"Stop loss order created: {order['id']}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating stop loss order: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID
            symbol: Trading pair

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Canceling order {order_id} for {symbol}")
            self.exchange.cancel_order(order_id, symbol)
            return True

        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """
        Get order status.

        Args:
            order_id: Order ID
            symbol: Trading pair

        Returns:
            Order info or None if failed
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order

        except Exception as e:
            self.logger.error(f"Error fetching order status: {e}")
            return None

    def get_open_orders(self, symbol: str = None) -> list:
        """
        Get open orders.

        Args:
            symbol: Trading pair (optional, None for all)

        Returns:
            List of open orders
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders

        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []
