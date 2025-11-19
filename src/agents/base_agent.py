"""Base agent class for agent-based modeling"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class AgentAction(Enum):
    """Agent actions in the market"""
    BUY = 1
    SELL = -1
    HOLD = 0


class AgentType(Enum):
    """Types of market agents"""
    RETAIL = "retail"
    INSTITUTIONAL = "institutional"
    WHALE = "whale"
    ALGORITHMIC = "algorithmic"
    MOMENTUM_TRADER = "momentum_trader"
    CONTRARIAN = "contrarian"


class BaseAgent(ABC):
    """Base class for all market agents in the ABM"""

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        initial_capital: float,
        risk_tolerance: float = 0.5,
        other_investments: Dict[str, float] = None,
    ):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (retail, institutional, etc.)
            initial_capital: Initial capital allocated to Bitcoin trading
            risk_tolerance: Risk tolerance level (0.0 to 1.0)
            other_investments: Dictionary of other investments {asset: amount}
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_tolerance = np.clip(risk_tolerance, 0.0, 1.0)

        # Portfolio tracking
        self.btc_holdings = 0.0
        self.other_investments = other_investments or {}
        self.total_portfolio_value = initial_capital + sum(self.other_investments.values())

        # Trading history
        self.trade_history: List[Dict[str, Any]] = []
        self.last_action = AgentAction.HOLD
        self.consecutive_actions = 0

        # Agent memory and learning
        self.memory_window = 20  # Remember last N periods
        self.price_memory: List[float] = []
        self.action_memory: List[AgentAction] = []

        # Sentiment and social influence
        self.sentiment = 0.0  # -1 (bearish) to 1 (bullish)
        self.confidence = 0.5  # Confidence in own decisions (0 to 1)
        self.social_influence = 0.5  # How much influenced by other agents (0 to 1)

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    @abstractmethod
    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """
        Decide what action to take based on market conditions and other agents.

        Args:
            market_data: Historical market data (OHLCV)
            current_price: Current Bitcoin price
            market_sentiment: Overall market sentiment (-1 to 1)
            other_agents_actions: Distribution of other agents' actions

        Returns:
            Action to take (BUY, SELL, or HOLD)
        """
        pass

    @abstractmethod
    def calculate_position_size(self, current_price: float) -> float:
        """
        Calculate how much to buy/sell based on agent characteristics.

        Args:
            current_price: Current Bitcoin price

        Returns:
            Amount of BTC to trade
        """
        pass

    def execute_action(
        self,
        action: AgentAction,
        price: float,
        timestamp: datetime,
        amount: float = None,
    ) -> Dict[str, Any]:
        """
        Execute a trading action and update agent state.

        Args:
            action: Action to execute
            price: Current price
            timestamp: Timestamp of action
            amount: Amount to trade (if None, calculated automatically)

        Returns:
            Trade details
        """
        if amount is None:
            amount = self.calculate_position_size(price)

        trade = {
            "timestamp": timestamp,
            "action": action,
            "price": price,
            "amount": amount,
            "capital_before": self.current_capital,
            "holdings_before": self.btc_holdings,
        }

        if action == AgentAction.BUY and amount > 0:
            cost = amount * price
            if cost <= self.current_capital:
                self.current_capital -= cost
                self.btc_holdings += amount
                self.total_trades += 1

        elif action == AgentAction.SELL and amount > 0:
            if amount <= self.btc_holdings:
                revenue = amount * price
                self.current_capital += revenue
                self.btc_holdings -= amount
                self.total_trades += 1

                # Track P&L
                if len(self.trade_history) > 0:
                    avg_buy_price = self._calculate_average_buy_price()
                    pnl = (price - avg_buy_price) * amount
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.winning_trades += 1

        trade["capital_after"] = self.current_capital
        trade["holdings_after"] = self.btc_holdings

        self.trade_history.append(trade)
        self._update_action_memory(action)

        return trade

    def update_sentiment(
        self,
        market_data: pd.DataFrame,
        market_sentiment: float,
        other_agents_sentiment: float,
    ):
        """
        Update agent's sentiment based on market and social factors.

        Args:
            market_data: Recent market data
            market_sentiment: Overall market sentiment
            other_agents_sentiment: Average sentiment of other agents
        """
        # Calculate technical sentiment
        if len(market_data) > 1:
            price_change = (
                market_data['close'].iloc[-1] - market_data['close'].iloc[-2]
            ) / market_data['close'].iloc[-2]
            technical_sentiment = np.tanh(price_change * 100)
        else:
            technical_sentiment = 0.0

        # Blend personal analysis with social influence
        personal_weight = 1.0 - self.social_influence
        social_weight = self.social_influence

        self.sentiment = (
            personal_weight * technical_sentiment +
            social_weight * other_agents_sentiment * 0.7 +
            social_weight * market_sentiment * 0.3
        )

        self.sentiment = np.clip(self.sentiment, -1.0, 1.0)

    def update_price_memory(self, price: float):
        """Add price to agent's memory."""
        self.price_memory.append(price)
        if len(self.price_memory) > self.memory_window:
            self.price_memory.pop(0)

    def get_portfolio_value(self, current_btc_price: float) -> float:
        """Calculate total portfolio value."""
        btc_value = self.btc_holdings * current_btc_price
        return self.current_capital + btc_value + sum(self.other_investments.values())

    def get_btc_allocation_pct(self, current_btc_price: float) -> float:
        """Get percentage of portfolio allocated to Bitcoin."""
        total_value = self.get_portfolio_value(current_btc_price)
        if total_value == 0:
            return 0.0
        btc_value = self.btc_holdings * current_btc_price
        return btc_value / total_value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent's performance metrics."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "current_capital": self.current_capital,
            "btc_holdings": self.btc_holdings,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
        }

    def _update_action_memory(self, action: AgentAction):
        """Update memory of recent actions."""
        if action == self.last_action:
            self.consecutive_actions += 1
        else:
            self.consecutive_actions = 1
            self.last_action = action

        self.action_memory.append(action)
        if len(self.action_memory) > self.memory_window:
            self.action_memory.pop(0)

    def _calculate_average_buy_price(self) -> float:
        """Calculate average buy price from trade history."""
        buy_trades = [
            t for t in self.trade_history
            if t["action"] == AgentAction.BUY
        ]

        if not buy_trades:
            return 0.0

        total_cost = sum(t["price"] * t["amount"] for t in buy_trades)
        total_amount = sum(t["amount"] for t in buy_trades)

        return total_cost / total_amount if total_amount > 0 else 0.0

    def __repr__(self):
        return (
            f"{self.agent_type.value}Agent(id={self.agent_id}, "
            f"capital={self.current_capital:.2f}, "
            f"btc={self.btc_holdings:.6f}, "
            f"sentiment={self.sentiment:.2f})"
        )
