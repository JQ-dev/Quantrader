"""Agent-based market simulator for Bitcoin price prediction"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

from .base_agent import BaseAgent, AgentAction, AgentType
from .agent_types import (
    RetailAgent,
    InstitutionalAgent,
    WhaleAgent,
    AlgorithmicAgent,
    MomentumTrader,
    ContrarianAgent,
)


class AgentBasedMarketSimulator:
    """
    Simulates a market with multiple heterogeneous agents.

    The simulator:
    1. Creates a population of different agent types
    2. Allows agents to observe market data and other agents' actions
    3. Agents make decisions based on their strategies
    4. Aggregates agent actions to predict market movement
    """

    def __init__(
        self,
        num_retail: int = 50,
        num_institutional: int = 10,
        num_whales: int = 3,
        num_algorithmic: int = 15,
        num_momentum: int = 10,
        num_contrarian: int = 5,
        capital_distribution: str = "realistic",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the agent-based market simulator.

        Args:
            num_retail: Number of retail investors
            num_institutional: Number of institutional investors
            num_whales: Number of whale investors
            num_algorithmic: Number of algorithmic traders
            num_momentum: Number of momentum traders
            num_contrarian: Number of contrarian traders
            capital_distribution: How to distribute capital ('equal' or 'realistic')
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.agents: List[BaseAgent] = []
        self.agent_id_counter = 0

        # Initialize logger
        self.logger = logging.getLogger("ABMSimulator")

        # Create agent population
        self._create_agent_population(
            num_retail,
            num_institutional,
            num_whales,
            num_algorithmic,
            num_momentum,
            num_contrarian,
            capital_distribution,
        )

        # Simulation state
        self.market_sentiment = 0.0
        self.agent_actions_history: List[Dict[AgentAction, int]] = []
        self.price_predictions: List[float] = []

        self.logger.info(
            f"Initialized ABM with {len(self.agents)} agents: "
            f"{num_retail} retail, {num_institutional} institutional, "
            f"{num_whales} whales, {num_algorithmic} algorithmic, "
            f"{num_momentum} momentum, {num_contrarian} contrarian"
        )

    def _create_agent_population(
        self,
        num_retail: int,
        num_institutional: int,
        num_whales: int,
        num_algorithmic: int,
        num_momentum: int,
        num_contrarian: int,
        capital_distribution: str,
    ):
        """Create the initial population of agents with diverse characteristics."""

        # Define capital ranges for each agent type
        if capital_distribution == "realistic":
            capital_ranges = {
                "retail": (1000, 50000),
                "institutional": (1000000, 10000000),
                "whale": (50000000, 500000000),
                "algorithmic": (100000, 5000000),
                "momentum": (50000, 500000),
                "contrarian": (50000, 500000),
            }
        else:  # equal
            base_capital = 100000
            capital_ranges = {
                "retail": (base_capital, base_capital),
                "institutional": (base_capital, base_capital),
                "whale": (base_capital, base_capital),
                "algorithmic": (base_capital, base_capital),
                "momentum": (base_capital, base_capital),
                "contrarian": (base_capital, base_capital),
            }

        # Create retail agents
        for _ in range(num_retail):
            capital = np.random.uniform(*capital_ranges["retail"])
            risk_tolerance = np.random.beta(2, 5)  # Skewed toward lower risk
            other_investments = self._generate_other_investments(capital, "retail")
            agent = RetailAgent(self.agent_id_counter, capital, risk_tolerance, other_investments)
            self.agents.append(agent)
            self.agent_id_counter += 1

        # Create institutional agents
        for _ in range(num_institutional):
            capital = np.random.uniform(*capital_ranges["institutional"])
            risk_tolerance = np.random.beta(3, 3)  # More balanced
            other_investments = self._generate_other_investments(capital, "institutional")
            agent = InstitutionalAgent(self.agent_id_counter, capital, risk_tolerance, other_investments)
            self.agents.append(agent)
            self.agent_id_counter += 1

        # Create whale agents
        for _ in range(num_whales):
            capital = np.random.uniform(*capital_ranges["whale"])
            risk_tolerance = np.random.beta(4, 2)  # Skewed toward higher risk
            other_investments = self._generate_other_investments(capital, "whale")
            agent = WhaleAgent(self.agent_id_counter, capital, risk_tolerance, other_investments)
            self.agents.append(agent)
            self.agent_id_counter += 1

        # Create algorithmic agents
        for _ in range(num_algorithmic):
            capital = np.random.uniform(*capital_ranges["algorithmic"])
            risk_tolerance = np.random.uniform(0.3, 0.8)
            other_investments = self._generate_other_investments(capital, "algorithmic")
            agent = AlgorithmicAgent(self.agent_id_counter, capital, risk_tolerance, other_investments)
            self.agents.append(agent)
            self.agent_id_counter += 1

        # Create momentum traders
        for _ in range(num_momentum):
            capital = np.random.uniform(*capital_ranges["momentum"])
            risk_tolerance = np.random.uniform(0.5, 0.9)  # Higher risk
            other_investments = self._generate_other_investments(capital, "momentum")
            agent = MomentumTrader(self.agent_id_counter, capital, risk_tolerance, other_investments)
            self.agents.append(agent)
            self.agent_id_counter += 1

        # Create contrarian traders
        for _ in range(num_contrarian):
            capital = np.random.uniform(*capital_ranges["contrarian"])
            risk_tolerance = np.random.uniform(0.4, 0.7)
            other_investments = self._generate_other_investments(capital, "contrarian")
            agent = ContrarianAgent(self.agent_id_counter, capital, risk_tolerance, other_investments)
            self.agents.append(agent)
            self.agent_id_counter += 1

    def _generate_other_investments(self, btc_capital: float, agent_type: str) -> Dict[str, float]:
        """Generate other investment holdings for diversification."""

        if agent_type == "retail":
            # Retail: Small diversification
            other_total = btc_capital * np.random.uniform(0.5, 2.0)
            return {
                "stocks": other_total * 0.6,
                "savings": other_total * 0.3,
                "other_crypto": other_total * 0.1,
            }

        elif agent_type == "institutional":
            # Institutional: Large diversified portfolio
            other_total = btc_capital * np.random.uniform(10, 50)
            return {
                "stocks": other_total * 0.5,
                "bonds": other_total * 0.3,
                "real_estate": other_total * 0.15,
                "other_crypto": other_total * 0.05,
            }

        elif agent_type == "whale":
            # Whales: Massive diversification
            other_total = btc_capital * np.random.uniform(5, 20)
            return {
                "stocks": other_total * 0.4,
                "real_estate": other_total * 0.3,
                "bonds": other_total * 0.2,
                "other_crypto": other_total * 0.1,
            }

        else:
            # Others: Moderate diversification
            other_total = btc_capital * np.random.uniform(1, 5)
            return {
                "stocks": other_total * 0.6,
                "other_crypto": other_total * 0.4,
            }

    def simulate_step(
        self, market_data: pd.DataFrame, current_price: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Simulate one time step of the market.

        Args:
            market_data: Historical market data (OHLCV)
            current_price: Current Bitcoin price
            timestamp: Current timestamp

        Returns:
            Dictionary with simulation results
        """

        # Update agent memories
        for agent in self.agents:
            agent.update_price_memory(current_price)

        # Calculate market sentiment from recent price action
        self.market_sentiment = self._calculate_market_sentiment(market_data)

        # First pass: agents decide actions
        agent_actions = defaultdict(int)
        agent_decisions = []

        # Collect all agent decisions
        for agent in self.agents:
            # Get other agents' recent actions (excluding self)
            other_actions = self._get_other_agents_actions(agent.agent_id)

            # Agent decides action
            action = agent.decide_action(
                market_data, current_price, self.market_sentiment, other_actions
            )

            agent_actions[action] += 1
            agent_decisions.append({"agent": agent, "action": action})

        # Store actions history
        self.agent_actions_history.append(dict(agent_actions))

        # Second pass: agents update sentiment and execute actions
        avg_sentiment = self._calculate_average_agent_sentiment()

        executed_trades = []
        for decision in agent_decisions:
            agent = decision["agent"]
            action = decision["action"]

            # Update agent sentiment
            agent.update_sentiment(market_data, self.market_sentiment, avg_sentiment)

            # Execute action
            if action != AgentAction.HOLD:
                trade = agent.execute_action(action, current_price, timestamp)
                executed_trades.append(trade)

        # Predict price movement based on aggregated agent actions
        price_prediction = self._predict_price_movement(
            agent_actions, current_price, market_data
        )

        self.price_predictions.append(price_prediction)

        # Calculate aggregate metrics
        results = {
            "timestamp": timestamp,
            "current_price": current_price,
            "predicted_price": price_prediction,
            "market_sentiment": self.market_sentiment,
            "agent_actions": dict(agent_actions),
            "num_trades": len(executed_trades),
            "total_buy_volume": sum(
                t["amount"] for t in executed_trades if t["action"] == AgentAction.BUY
            ),
            "total_sell_volume": sum(
                t["amount"] for t in executed_trades if t["action"] == AgentAction.SELL
            ),
            "buy_pressure": agent_actions[AgentAction.BUY],
            "sell_pressure": agent_actions[AgentAction.SELL],
            "hold_count": agent_actions[AgentAction.HOLD],
        }

        return results

    def _calculate_market_sentiment(self, market_data: pd.DataFrame) -> float:
        """Calculate overall market sentiment from price action."""

        if len(market_data) < 10:
            return 0.0

        # Price momentum
        price_change_5 = (
            (market_data['close'].iloc[-1] - market_data['close'].iloc[-5]) /
            market_data['close'].iloc[-5]
        )
        price_change_10 = (
            (market_data['close'].iloc[-1] - market_data['close'].iloc[-10]) /
            market_data['close'].iloc[-10]
        )

        # Volume trend
        avg_volume = market_data['volume'].tail(10).mean()
        current_volume = market_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Combine signals
        momentum_sentiment = np.tanh((price_change_5 * 0.6 + price_change_10 * 0.4) * 10)
        volume_sentiment = np.tanh((volume_ratio - 1) * 2)

        sentiment = 0.7 * momentum_sentiment + 0.3 * volume_sentiment

        return np.clip(sentiment, -1.0, 1.0)

    def _get_other_agents_actions(self, exclude_agent_id: int) -> Dict[AgentAction, int]:
        """Get recent actions of other agents (excluding specified agent)."""

        if not self.agent_actions_history:
            return {AgentAction.BUY: 0, AgentAction.SELL: 0, AgentAction.HOLD: 0}

        # Use most recent actions
        return self.agent_actions_history[-1]

    def _calculate_average_agent_sentiment(self) -> float:
        """Calculate average sentiment across all agents."""

        if not self.agents:
            return 0.0

        total_sentiment = sum(agent.sentiment for agent in self.agents)
        return total_sentiment / len(self.agents)

    def _predict_price_movement(
        self, agent_actions: Dict[AgentAction, int], current_price: float, market_data: pd.DataFrame
    ) -> float:
        """
        Predict price movement based on agent actions.

        Uses a weighted voting system where different agent types have different influence.
        """

        # Calculate buy/sell pressure weighted by agent capital and type
        buy_pressure = 0.0
        sell_pressure = 0.0

        for agent in self.agents:
            if not agent.action_memory:
                continue

            last_action = agent.action_memory[-1]
            agent_weight = self._get_agent_weight(agent, current_price)

            if last_action == AgentAction.BUY:
                buy_pressure += agent_weight
            elif last_action == AgentAction.SELL:
                sell_pressure += agent_weight

        # Calculate net pressure
        total_pressure = buy_pressure + sell_pressure
        if total_pressure == 0:
            return current_price

        net_pressure = (buy_pressure - sell_pressure) / total_pressure

        # Calculate volatility from recent data
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().tail(20)
            volatility = returns.std()
        else:
            volatility = 0.02  # Default 2%

        # Predict price change based on net pressure and volatility
        # Strong pressure can move price by up to 2 * volatility
        predicted_change_pct = net_pressure * volatility * 2

        predicted_price = current_price * (1 + predicted_change_pct)

        return predicted_price

    def _get_agent_weight(self, agent: BaseAgent, current_price: float) -> float:
        """
        Calculate the market influence weight of an agent.

        Weight is based on:
        - Agent type (whales have more influence)
        - Portfolio size
        - Agent confidence
        """

        # Base weights by type
        type_weights = {
            AgentType.RETAIL: 1.0,
            AgentType.INSTITUTIONAL: 10.0,
            AgentType.WHALE: 50.0,
            AgentType.ALGORITHMIC: 5.0,
            AgentType.MOMENTUM_TRADER: 3.0,
            AgentType.CONTRARIAN: 3.0,
        }

        base_weight = type_weights.get(agent.agent_type, 1.0)

        # Adjust by portfolio size
        portfolio_value = agent.get_portfolio_value(current_price)
        size_multiplier = np.log10(max(portfolio_value, 1000)) / np.log10(100000)

        # Adjust by confidence
        confidence_multiplier = 0.5 + agent.confidence

        total_weight = base_weight * size_multiplier * confidence_multiplier

        return total_weight

    def get_prediction_signal(
        self, market_data: pd.DataFrame, current_price: float, timestamp: datetime
    ) -> Tuple[int, float]:
        """
        Get trading signal and confidence from agent-based simulation.

        Args:
            market_data: Historical market data
            current_price: Current price
            timestamp: Current timestamp

        Returns:
            Tuple of (signal, confidence) where:
                signal: 1 (BUY), -1 (SELL), or 0 (HOLD)
                confidence: 0.0 to 1.0
        """

        # Run simulation step
        results = self.simulate_step(market_data, current_price, timestamp)

        predicted_price = results["predicted_price"]
        price_change_pct = (predicted_price - current_price) / current_price

        # Determine signal based on predicted change
        if price_change_pct > 0.005:  # > 0.5% predicted increase
            signal = 1
        elif price_change_pct < -0.005:  # > 0.5% predicted decrease
            signal = -1
        else:
            signal = 0

        # Calculate confidence based on agent agreement
        total_actions = sum(results["agent_actions"].values())
        if total_actions > 0:
            buy_ratio = results["agent_actions"].get(AgentAction.BUY, 0) / total_actions
            sell_ratio = results["agent_actions"].get(AgentAction.SELL, 0) / total_actions

            # Higher confidence when agents agree
            if signal == 1:
                confidence = buy_ratio
            elif signal == -1:
                confidence = sell_ratio
            else:
                confidence = 0.5
        else:
            confidence = 0.5

        return signal, confidence

    def get_agent_statistics(self) -> pd.DataFrame:
        """Get statistics about all agents."""

        stats = []
        for agent in self.agents:
            metrics = agent.get_performance_metrics()
            stats.append(metrics)

        return pd.DataFrame(stats)

    def reset(self):
        """Reset the simulator state."""

        # Reset all agents
        for agent in self.agents:
            agent.current_capital = agent.initial_capital
            agent.btc_holdings = 0.0
            agent.trade_history = []
            agent.total_trades = 0
            agent.winning_trades = 0
            agent.total_pnl = 0.0
            agent.sentiment = 0.0
            agent.price_memory = []
            agent.action_memory = []

        # Reset simulator state
        self.market_sentiment = 0.0
        self.agent_actions_history = []
        self.price_predictions = []

        self.logger.info("Simulator reset")

    def __repr__(self):
        return f"ABMSimulator(agents={len(self.agents)}, sentiment={self.market_sentiment:.2f})"
