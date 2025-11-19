"""Implementation of different agent types for agent-based modeling"""

import numpy as np
import pandas as pd
from typing import Dict
from .base_agent import BaseAgent, AgentAction, AgentType


class RetailAgent(BaseAgent):
    """
    Retail investor - individual investor with smaller capital.

    Characteristics:
    - Smaller position sizes
    - More emotional/sentiment-driven decisions
    - Higher social influence (follows the crowd)
    - Less sophisticated analysis
    """

    def __init__(self, agent_id: int, initial_capital: float, risk_tolerance: float = 0.5,
                 other_investments: Dict[str, float] = None):
        super().__init__(agent_id, AgentType.RETAIL, initial_capital, risk_tolerance, other_investments)
        self.social_influence = 0.7  # Highly influenced by others
        self.confidence = 0.4  # Lower confidence
        self.memory_window = 10  # Shorter memory
        self.fomo_threshold = 0.03  # 3% price increase triggers FOMO

    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """Retail investors are sentiment-driven and follow trends."""

        if len(market_data) < 5:
            return AgentAction.HOLD

        # Calculate recent price momentum
        recent_prices = market_data['close'].tail(5)
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        # FOMO buying on strong upward momentum
        if price_change > self.fomo_threshold and market_sentiment > 0.3:
            return AgentAction.BUY

        # Panic selling on downward momentum
        if price_change < -self.fomo_threshold and market_sentiment < -0.3:
            if self.btc_holdings > 0:
                return AgentAction.SELL

        # Follow the crowd
        total_actions = sum(other_agents_actions.values())
        if total_actions > 0:
            buy_ratio = other_agents_actions.get(AgentAction.BUY, 0) / total_actions
            sell_ratio = other_agents_actions.get(AgentAction.SELL, 0) / total_actions

            # Herding behavior
            if buy_ratio > 0.6:
                return AgentAction.BUY
            elif sell_ratio > 0.6 and self.btc_holdings > 0:
                return AgentAction.SELL

        # Simple moving average crossover
        if len(market_data) >= 20:
            sma_short = market_data['close'].tail(5).mean()
            sma_long = market_data['close'].tail(20).mean()

            if sma_short > sma_long and self.sentiment > 0:
                return AgentAction.BUY
            elif sma_short < sma_long and self.btc_holdings > 0:
                return AgentAction.SELL

        return AgentAction.HOLD

    def calculate_position_size(self, current_price: float) -> float:
        """Retail investors trade smaller amounts, often round numbers."""

        # Use 10-30% of available capital based on risk tolerance
        trade_pct = 0.1 + (self.risk_tolerance * 0.2)
        max_trade_value = self.current_capital * trade_pct

        # Calculate BTC amount
        btc_amount = max_trade_value / current_price

        # Retail tends to buy in "round" amounts
        btc_amount = round(btc_amount, 4)

        return max(btc_amount, 0.001)  # Minimum 0.001 BTC


class InstitutionalAgent(BaseAgent):
    """
    Institutional investor - large fund or institution.

    Characteristics:
    - Larger capital
    - Data-driven decisions
    - Lower social influence
    - Longer time horizons
    - Risk management focused
    """

    def __init__(self, agent_id: int, initial_capital: float, risk_tolerance: float = 0.5,
                 other_investments: Dict[str, float] = None):
        super().__init__(agent_id, AgentType.INSTITUTIONAL, initial_capital, risk_tolerance, other_investments)
        self.social_influence = 0.2  # Less influenced by crowd
        self.confidence = 0.8  # High confidence
        self.memory_window = 50  # Long memory
        self.rebalance_threshold = 0.1  # Rebalance when allocation drifts 10%

    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """Institutions use sophisticated analysis and portfolio rebalancing."""

        if len(market_data) < 30:
            return AgentAction.HOLD

        # Calculate multiple technical indicators
        close_prices = market_data['close']

        # Moving averages
        sma_20 = close_prices.tail(20).mean()
        sma_50 = close_prices.tail(50).mean() if len(close_prices) >= 50 else sma_20

        # Volatility
        returns = close_prices.pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized

        # RSI-like momentum
        price_changes = close_prices.diff()
        gains = price_changes.where(price_changes > 0, 0).tail(14).mean()
        losses = -price_changes.where(price_changes < 0, 0).tail(14).mean()
        rs = gains / losses if losses != 0 else 1
        rsi = 100 - (100 / (1 + rs))

        # Portfolio rebalancing logic
        target_btc_allocation = 0.05 + (self.risk_tolerance * 0.15)  # 5-20% target
        current_allocation = self.get_btc_allocation_pct(current_price)

        allocation_diff = current_allocation - target_btc_allocation

        # Rebalance if allocation drifts significantly
        if abs(allocation_diff) > self.rebalance_threshold:
            if allocation_diff > 0:
                return AgentAction.SELL  # Over-allocated, sell
            else:
                return AgentAction.BUY  # Under-allocated, buy

        # Trend following with risk management
        trend_bullish = sma_20 > sma_50
        momentum_positive = rsi > 50
        low_volatility = volatility < 1.0

        if trend_bullish and momentum_positive and current_allocation < target_btc_allocation:
            return AgentAction.BUY
        elif not trend_bullish and current_allocation > 0:
            # Defensive selling in downtrends
            if rsi < 40 or volatility > 1.5:
                return AgentAction.SELL

        return AgentAction.HOLD

    def calculate_position_size(self, current_price: float) -> float:
        """Institutions trade larger, calculated amounts based on portfolio theory."""

        # Calculate target allocation
        target_btc_allocation = 0.05 + (self.risk_tolerance * 0.15)
        current_allocation = self.get_btc_allocation_pct(current_price)

        total_portfolio = self.get_portfolio_value(current_price)
        target_btc_value = total_portfolio * target_btc_allocation
        current_btc_value = self.btc_holdings * current_price

        # Calculate difference
        value_diff = target_btc_value - current_btc_value
        btc_amount = abs(value_diff) / current_price

        # Institutions trade in larger chunks but with limits
        max_single_trade = self.current_capital * 0.3
        btc_amount = min(btc_amount, max_single_trade / current_price)

        return round(btc_amount, 6)


class WhaleAgent(BaseAgent):
    """
    Whale - extremely high net worth individual or entity.

    Characteristics:
    - Very large capital
    - Can influence market
    - Strategic, patient
    - Low frequency trading
    """

    def __init__(self, agent_id: int, initial_capital: float, risk_tolerance: float = 0.5,
                 other_investments: Dict[str, float] = None):
        super().__init__(agent_id, AgentType.WHALE, initial_capital, risk_tolerance, other_investments)
        self.social_influence = 0.1  # Minimal social influence
        self.confidence = 0.9  # Very high confidence
        self.memory_window = 100  # Very long memory
        self.accumulation_mode = True  # Long-term accumulation strategy
        self.distribution_threshold = 0.5  # Profit threshold to start selling

    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """Whales accumulate during fear, distribute during greed."""

        if len(market_data) < 50:
            return AgentAction.HOLD

        close_prices = market_data['close']

        # Calculate long-term metrics
        sma_50 = close_prices.tail(50).mean()
        sma_100 = close_prices.tail(100).mean() if len(close_prices) >= 100 else sma_50

        # Identify accumulation zones (market fear)
        current_allocation = self.get_btc_allocation_pct(current_price)

        # Calculate unrealized P&L
        if self.btc_holdings > 0:
            avg_buy_price = self._calculate_average_buy_price()
            unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price if avg_buy_price > 0 else 0
        else:
            unrealized_pnl_pct = 0

        # Accumulation strategy: buy when others are fearful
        if market_sentiment < -0.4 and current_price < sma_50:
            # Counter-trend accumulation
            if current_allocation < 0.3:  # Whales can hold large positions
                return AgentAction.BUY

        # Distribution strategy: sell when others are greedy
        if market_sentiment > 0.6 and unrealized_pnl_pct > self.distribution_threshold:
            # Take profits in euphoric markets
            if self.btc_holdings > 0:
                return AgentAction.SELL

        # Strategic rebalancing based on macro trends
        long_term_bullish = sma_50 > sma_100

        if long_term_bullish and current_allocation < 0.2:
            return AgentAction.BUY
        elif not long_term_bullish and unrealized_pnl_pct > 0.2:
            return AgentAction.SELL

        return AgentAction.HOLD

    def calculate_position_size(self, current_price: float) -> float:
        """Whales trade very large amounts but gradually to avoid market impact."""

        # Whales trade large amounts but split into smaller chunks
        target_trade_value = self.current_capital * 0.2 * (0.5 + self.risk_tolerance * 0.5)
        btc_amount = target_trade_value / current_price

        return round(btc_amount, 6)


class AlgorithmicAgent(BaseAgent):
    """
    Algorithmic trader - automated trading bot.

    Characteristics:
    - Quantitative, data-driven
    - No emotional bias
    - High frequency potential
    - Multiple strategy signals
    """

    def __init__(self, agent_id: int, initial_capital: float, risk_tolerance: float = 0.5,
                 other_investments: Dict[str, float] = None):
        super().__init__(agent_id, AgentType.ALGORITHMIC, initial_capital, risk_tolerance, other_investments)
        self.social_influence = 0.0  # No social influence
        self.confidence = 0.7
        self.memory_window = 30
        self.signal_threshold = 0.6  # Require strong signal to act

    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """Algorithm combines multiple quantitative signals."""

        if len(market_data) < 30:
            return AgentAction.HOLD

        # Calculate multiple technical signals
        signals = []

        # 1. Moving Average Signal
        sma_10 = market_data['close'].tail(10).mean()
        sma_30 = market_data['close'].tail(30).mean()
        if sma_10 > sma_30:
            signals.append(1)
        elif sma_10 < sma_30:
            signals.append(-1)
        else:
            signals.append(0)

        # 2. RSI Signal
        price_changes = market_data['close'].diff()
        gains = price_changes.where(price_changes > 0, 0).tail(14).mean()
        losses = -price_changes.where(price_changes < 0, 0).tail(14).mean()
        rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 50

        if rsi < 30:
            signals.append(1)  # Oversold
        elif rsi > 70:
            signals.append(-1)  # Overbought
        else:
            signals.append(0)

        # 3. Volume Signal
        avg_volume = market_data['volume'].tail(20).mean()
        current_volume = market_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]

        if volume_ratio > 1.5 and price_change > 0:
            signals.append(1)  # Volume confirming uptrend
        elif volume_ratio > 1.5 and price_change < 0:
            signals.append(-1)  # Volume confirming downtrend
        else:
            signals.append(0)

        # 4. Bollinger Band Signal
        sma_20 = market_data['close'].tail(20).mean()
        std_20 = market_data['close'].tail(20).std()
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)

        if current_price < lower_band:
            signals.append(1)  # Below lower band
        elif current_price > upper_band:
            signals.append(-1)  # Above upper band
        else:
            signals.append(0)

        # Aggregate signals
        total_signal = sum(signals) / len(signals)

        # Decision based on signal strength
        if total_signal > self.signal_threshold:
            return AgentAction.BUY
        elif total_signal < -self.signal_threshold:
            if self.btc_holdings > 0:
                return AgentAction.SELL

        return AgentAction.HOLD

    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position using Kelly Criterion approximation."""

        # Simple Kelly Criterion: f = (bp - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio

        win_rate = self.winning_trades / self.total_trades if self.total_trades > 5 else 0.5
        loss_rate = 1 - win_rate
        win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk

        kelly_fraction = (win_loss_ratio * win_rate - loss_rate) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Apply risk tolerance
        position_fraction = kelly_fraction * self.risk_tolerance
        trade_value = self.current_capital * position_fraction

        return round(trade_value / current_price, 6)


class MomentumTrader(BaseAgent):
    """
    Momentum trader - follows strong trends.

    Characteristics:
    - Trend following
    - Quick to enter/exit
    - Momentum-focused
    """

    def __init__(self, agent_id: int, initial_capital: float, risk_tolerance: float = 0.5,
                 other_investments: Dict[str, float] = None):
        super().__init__(agent_id, AgentType.MOMENTUM_TRADER, initial_capital, risk_tolerance, other_investments)
        self.social_influence = 0.4
        self.confidence = 0.6
        self.momentum_threshold = 0.02  # 2% momentum threshold

    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """Trade based on price momentum."""

        if len(market_data) < 10:
            return AgentAction.HOLD

        # Calculate momentum
        returns_5d = (market_data['close'].iloc[-1] - market_data['close'].iloc[-5]) / market_data['close'].iloc[-5]
        returns_10d = (market_data['close'].iloc[-1] - market_data['close'].iloc[-10]) / market_data['close'].iloc[-10]

        momentum_score = (returns_5d * 0.6 + returns_10d * 0.4)

        # Strong upward momentum
        if momentum_score > self.momentum_threshold:
            return AgentAction.BUY
        # Strong downward momentum
        elif momentum_score < -self.momentum_threshold and self.btc_holdings > 0:
            return AgentAction.SELL

        return AgentAction.HOLD

    def calculate_position_size(self, current_price: float) -> float:
        """Momentum traders use moderate position sizes."""
        trade_value = self.current_capital * 0.2 * self.risk_tolerance
        return round(trade_value / current_price, 6)


class ContrarianAgent(BaseAgent):
    """
    Contrarian trader - goes against the crowd.

    Characteristics:
    - Counter-trend
    - Buys fear, sells greed
    - Patient
    """

    def __init__(self, agent_id: int, initial_capital: float, risk_tolerance: float = 0.5,
                 other_investments: Dict[str, float] = None):
        super().__init__(agent_id, AgentType.CONTRARIAN, initial_capital, risk_tolerance, other_investments)
        self.social_influence = -0.5  # Negative social influence (contrarian)
        self.confidence = 0.7

    def decide_action(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        market_sentiment: float,
        other_agents_actions: Dict[AgentAction, int],
    ) -> AgentAction:
        """Go against the crowd and extreme sentiment."""

        if len(market_data) < 20:
            return AgentAction.HOLD

        # Extreme fear = buy opportunity
        if market_sentiment < -0.5:
            total_actions = sum(other_agents_actions.values())
            if total_actions > 0:
                sell_ratio = other_agents_actions.get(AgentAction.SELL, 0) / total_actions
                if sell_ratio > 0.7:  # Everyone selling = buy
                    return AgentAction.BUY

        # Extreme greed = sell opportunity
        if market_sentiment > 0.5 and self.btc_holdings > 0:
            total_actions = sum(other_agents_actions.values())
            if total_actions > 0:
                buy_ratio = other_agents_actions.get(AgentAction.BUY, 0) / total_actions
                if buy_ratio > 0.7:  # Everyone buying = sell
                    return AgentAction.SELL

        return AgentAction.HOLD

    def calculate_position_size(self, current_price: float) -> float:
        """Contrarians use conservative position sizes."""
        trade_value = self.current_capital * 0.15 * self.risk_tolerance
        return round(trade_value / current_price, 6)
