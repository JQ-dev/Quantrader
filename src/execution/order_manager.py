"""Order management and execution"""

from typing import Dict, Optional
from enum import Enum
from .exchange import ExchangeInterface
from ..utils.logger import setup_logger


class OrderType(Enum):
    """Order types"""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop_loss'


class OrderManager:
    """Manage order execution and tracking"""

    def __init__(self, exchange: ExchangeInterface, dry_run: bool = True):
        """
        Initialize order manager.

        Args:
            exchange: Exchange interface
            dry_run: If True, simulate orders without executing
        """
        self.logger = setup_logger(__name__)
        self.exchange = exchange
        self.dry_run = dry_run
        self.pending_orders = []
        self.executed_orders = []

        mode = "DRY RUN" if dry_run else "LIVE"
        self.logger.info(f"Order manager initialized in {mode} mode")

    def execute_buy(self, symbol: str, amount: float,
                   order_type: OrderType = OrderType.MARKET,
                   limit_price: float = None) -> Optional[Dict]:
        """
        Execute a buy order.

        Args:
            symbol: Trading pair
            amount: Amount to buy (in base currency)
            order_type: Type of order
            limit_price: Price for limit orders

        Returns:
            Order info or None if failed
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] BUY {amount} {symbol}")
            order = self._create_mock_order(symbol, 'buy', amount, order_type, limit_price)
            self.executed_orders.append(order)
            return order

        try:
            if order_type == OrderType.MARKET:
                order = self.exchange.create_market_order(symbol, 'buy', amount)
            elif order_type == OrderType.LIMIT:
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                order = self.exchange.create_limit_order(symbol, 'buy', amount, limit_price)
            else:
                raise ValueError(f"Unsupported order type for buy: {order_type}")

            if order:
                self.executed_orders.append(order)
                self.logger.info(f"Buy order executed: {order['id']}")
                return order

        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return None

    def execute_sell(self, symbol: str, amount: float,
                    order_type: OrderType = OrderType.MARKET,
                    limit_price: float = None) -> Optional[Dict]:
        """
        Execute a sell order.

        Args:
            symbol: Trading pair
            amount: Amount to sell (in base currency)
            order_type: Type of order
            limit_price: Price for limit orders

        Returns:
            Order info or None if failed
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] SELL {amount} {symbol}")
            order = self._create_mock_order(symbol, 'sell', amount, order_type, limit_price)
            self.executed_orders.append(order)
            return order

        try:
            if order_type == OrderType.MARKET:
                order = self.exchange.create_market_order(symbol, 'sell', amount)
            elif order_type == OrderType.LIMIT:
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                order = self.exchange.create_limit_order(symbol, 'sell', amount, limit_price)
            else:
                raise ValueError(f"Unsupported order type for sell: {order_type}")

            if order:
                self.executed_orders.append(order)
                self.logger.info(f"Sell order executed: {order['id']}")
                return order

        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return None

    def set_stop_loss(self, symbol: str, amount: float, stop_price: float) -> Optional[Dict]:
        """
        Set a stop loss order.

        Args:
            symbol: Trading pair
            amount: Amount
            stop_price: Stop loss trigger price

        Returns:
            Order info or None if failed
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] STOP LOSS {amount} {symbol} @ ${stop_price}")
            order = self._create_mock_order(symbol, 'sell', amount, OrderType.STOP_LOSS, stop_price)
            self.pending_orders.append(order)
            return order

        try:
            order = self.exchange.create_stop_loss_order(symbol, 'sell', amount, stop_price)

            if order:
                self.pending_orders.append(order)
                self.logger.info(f"Stop loss order set: {order['id']}")
                return order

        except Exception as e:
            self.logger.error(f"Error setting stop loss: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Cancel order {order_id}")
            # Remove from pending orders
            self.pending_orders = [o for o in self.pending_orders if o['id'] != order_id]
            return True

        success = self.exchange.cancel_order(order_id, symbol)

        if success:
            self.pending_orders = [o for o in self.pending_orders if o['id'] != order_id]

        return success

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Get order status"""
        if self.dry_run:
            # Find in executed or pending orders
            for order in self.executed_orders + self.pending_orders:
                if order['id'] == order_id:
                    return order
            return None

        return self.exchange.get_order_status(order_id, symbol)

    def _create_mock_order(self, symbol: str, side: str, amount: float,
                          order_type: OrderType, price: float = None) -> Dict:
        """Create a mock order for dry run mode"""
        import uuid
        from datetime import datetime

        # Get current price for mock execution
        if not price and order_type == OrderType.MARKET:
            try:
                ticker = self.exchange.get_ticker(symbol)
                price = ticker.get('last', 0)
            except:
                price = 0

        return {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'type': order_type.value,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'closed' if order_type == OrderType.MARKET else 'open',
            'timestamp': datetime.now().isoformat(),
            'dry_run': True,
        }

    def get_execution_summary(self) -> Dict:
        """Get summary of executed orders"""
        total_buys = sum(1 for o in self.executed_orders if o.get('side') == 'buy')
        total_sells = sum(1 for o in self.executed_orders if o.get('side') == 'sell')

        return {
            'total_orders': len(self.executed_orders),
            'buy_orders': total_buys,
            'sell_orders': total_sells,
            'pending_orders': len(self.pending_orders),
            'dry_run': self.dry_run,
        }
