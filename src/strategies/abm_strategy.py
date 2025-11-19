"""Agent-based modeling strategy for Bitcoin trading"""

import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal
from ..agents import AgentBasedMarketSimulator
from datetime import datetime


class ABMStrategy(BaseStrategy):
    """
    Trading strategy based on agent-based market simulation.

    This strategy simulates a market with heterogeneous agents (retail, institutional,
    whales, algorithmic traders, etc.) and predicts price movements based on their
    collective behavior and interactions.
    """

    def __init__(self, name: str = "ABM", params: Dict[str, Any] = None):
        """
        Initialize ABM strategy.

        Args:
            name: Strategy name
            params: Strategy parameters including:
                - num_retail: Number of retail investors (default: 50)
                - num_institutional: Number of institutional investors (default: 10)
                - num_whales: Number of whale investors (default: 3)
                - num_algorithmic: Number of algorithmic traders (default: 15)
                - num_momentum: Number of momentum traders (default: 10)
                - num_contrarian: Number of contrarian traders (default: 5)
                - capital_distribution: 'realistic' or 'equal' (default: 'realistic')
                - signal_threshold: Confidence threshold for signals (default: 0.6)
                - random_seed: Random seed for reproducibility (default: None)
        """
        default_params = {
            "num_retail": 50,
            "num_institutional": 10,
            "num_whales": 3,
            "num_algorithmic": 15,
            "num_momentum": 10,
            "num_contrarian": 5,
            "capital_distribution": "realistic",
            "signal_threshold": 0.6,
            "random_seed": None,
        }

        # Merge default params with user params
        if params:
            default_params.update(params)

        super().__init__(name, default_params)

        # Initialize the agent-based market simulator
        self.simulator = AgentBasedMarketSimulator(
            num_retail=self.params["num_retail"],
            num_institutional=self.params["num_institutional"],
            num_whales=self.params["num_whales"],
            num_algorithmic=self.params["num_algorithmic"],
            num_momentum=self.params["num_momentum"],
            num_contrarian=self.params["num_contrarian"],
            capital_distribution=self.params["capital_distribution"],
            random_seed=self.params["random_seed"],
        )

        self.signal_threshold = self.params["signal_threshold"]
        self.last_signal_confidence = 0.5

        self.logger.info(
            f"Initialized ABM strategy with {len(self.simulator.agents)} agents"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ABM strategy doesn't need traditional indicators.
        Agents use their own internal indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Same DataFrame (no modifications needed)
        """
        # No additional indicators needed - agents calculate their own
        return df

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate trading signal based on agent-based simulation.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Signal (BUY, SELL, or HOLD)
        """

        if not self.validate_data(df):
            return Signal.HOLD

        if len(df) < 30:
            self.logger.warning("Insufficient data for ABM simulation")
            return Signal.HOLD

        # Get current price
        current_price = df['close'].iloc[-1]

        # Get timestamp (use index if datetime, otherwise create one)
        if isinstance(df.index, pd.DatetimeIndex):
            timestamp = df.index[-1]
        else:
            timestamp = datetime.now()

        # Run agent-based simulation
        try:
            signal_value, confidence = self.simulator.get_prediction_signal(
                df, current_price, timestamp
            )

            self.last_signal_confidence = confidence

            # Only act on high-confidence signals
            if confidence < self.signal_threshold:
                self.logger.debug(
                    f"Low confidence signal ({confidence:.2f}), returning HOLD"
                )
                return Signal.HOLD

            # Convert to Signal enum
            if signal_value == 1:
                self.logger.info(
                    f"ABM predicts BUY (confidence: {confidence:.2f})"
                )
                return Signal.BUY
            elif signal_value == -1:
                self.logger.info(
                    f"ABM predicts SELL (confidence: {confidence:.2f})"
                )
                return Signal.SELL
            else:
                return Signal.HOLD

        except Exception as e:
            self.logger.error(f"Error in ABM simulation: {e}")
            return Signal.HOLD

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Get confidence level of the current signal.

        Args:
            df: DataFrame with market data

        Returns:
            Signal strength (0.0 to 1.0)
        """
        return self.last_signal_confidence

    def get_agent_statistics(self) -> pd.DataFrame:
        """
        Get performance statistics of all agents in the simulation.

        Returns:
            DataFrame with agent statistics
        """
        return self.simulator.get_agent_statistics()

    def get_market_sentiment(self) -> float:
        """
        Get current market sentiment from the simulation.

        Returns:
            Market sentiment (-1.0 to 1.0)
        """
        return self.simulator.market_sentiment

    def get_agent_actions_distribution(self) -> Dict[str, int]:
        """
        Get distribution of current agent actions.

        Returns:
            Dictionary with counts of BUY, SELL, HOLD actions
        """
        if not self.simulator.agent_actions_history:
            return {"BUY": 0, "SELL": 0, "HOLD": 0}

        latest_actions = self.simulator.agent_actions_history[-1]

        return {
            "BUY": latest_actions.get("BUY", 0),
            "SELL": latest_actions.get("SELL", 0),
            "HOLD": latest_actions.get("HOLD", 0),
        }

    def reset(self):
        """Reset the ABM simulator state."""
        self.simulator.reset()
        self.last_signal_confidence = 0.5
        self.logger.info("ABM strategy reset")

    def __repr__(self):
        return (
            f"ABMStrategy(agents={len(self.simulator.agents)}, "
            f"sentiment={self.simulator.market_sentiment:.2f})"
        )
