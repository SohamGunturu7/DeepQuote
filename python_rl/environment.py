"""
DeepQuote Trading Environment for Reinforcement Learning
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum

@dataclass
class MarketState:
    """Represents the current state of the market"""
    symbol: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    volume: float
    timestamp: float
    
    # Order book depth (top 5 levels)
    bid_prices: List[float]
    bid_quantities: List[float]
    ask_prices: List[float]
    ask_quantities: List[float]
    
    # Technical indicators
    moving_average_20: float
    moving_average_50: float
    volatility: float
    rsi: float

@dataclass
class AgentState:
    """Represents the current state of an agent"""
    cash: float
    inventory: Dict[str, float]  # symbol -> quantity
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    position_value: float
    available_cash: float

class ActionType(Enum):
    """Types of actions an agent can take"""
    BUY_MARKET = 0
    SELL_MARKET = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    CANCEL_ALL = 4
    HOLD = 5

@dataclass
class TradingAction:
    """Represents a trading action"""
    action_type: ActionType
    symbol: str
    quantity: float
    price: Optional[float] = None  # For limit orders

class DeepQuoteEnv(gym.Env):
    """
    DeepQuote Trading Environment for Reinforcement Learning
    
    This environment wraps the C++ market simulator and provides
    a standard gym interface for RL agents.
    """
    
    def __init__(self, 
                 symbols: List[str] = ["AAPL", "GOOGL"],
                 initial_cash: float = 100000.0,
                 max_position_size: float = 1000.0,
                 transaction_cost: float = 0.001,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.render_mode = render_mode
        
        # Market state tracking
        self.current_step = 0
        self.max_steps = 1000
        
        # Define action space
        # [action_type, symbol_idx, quantity_normalized, price_normalized]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([len(ActionType) - 1, len(symbols) - 1, 1, 1]),
            dtype=np.float32
        )
        
        # Define observation space
        # Market data + Agent state + Technical indicators
        market_features = 4 + 10 + 4  # basic + order book + technical
        agent_features = 6  # cash, inventory, pnl, etc.
        total_features = market_features * len(symbols) + agent_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )
        
        # State tracking
        self.market_states: Dict[str, MarketState] = {}
        self.agent_state = AgentState(
            cash=initial_cash,
            inventory={symbol: 0.0 for symbol in symbols},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            position_value=0.0,
            available_cash=initial_cash
        )
        
        # Price history for technical indicators
        self.price_history: Dict[str, List[float]] = {
            symbol: [] for symbol in symbols
        }
        
        # C++ simulator communication (placeholder for now)
        self.simulator_process = None
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Reset agent state
        self.agent_state = AgentState(
            cash=self.initial_cash,
            inventory={symbol: 0.0 for symbol in self.symbols},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            position_value=0.0,
            available_cash=self.initial_cash
        )
        
        # Reset price history
        for symbol in self.symbols:
            self.price_history[symbol] = []
        
        # Initialize market states with dummy data
        for symbol in self.symbols:
            self.market_states[symbol] = MarketState(
                symbol=symbol,
                best_bid=100.0,
                best_ask=100.1,
                mid_price=100.05,
                spread=0.1,
                volume=0.0,
                timestamp=time.time(),
                bid_prices=[100.0, 99.9, 99.8, 99.7, 99.6],
                bid_quantities=[100.0, 100.0, 100.0, 100.0, 100.0],
                ask_prices=[100.1, 100.2, 100.3, 100.4, 100.5],
                ask_quantities=[100.0, 100.0, 100.0, 100.0, 100.0],
                moving_average_20=100.0,
                moving_average_50=100.0,
                volatility=0.01,
                rsi=50.0
            )
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        
        # Parse action
        trading_action = self._parse_action(action)
        
        # Execute trading action
        reward = self._execute_action(trading_action)
        
        # Update market state (simulate market movement)
        self._update_market_state()
        
        # Update agent state
        self._update_agent_state()
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self._is_done()
        
        # Get info
        info = self._get_info()
        
        self.current_step += 1
        
        return observation, reward, done, False, info
    
    def _parse_action(self, action: np.ndarray) -> TradingAction:
        """Parse raw action array into TradingAction"""
        action_type_idx = int(action[0])
        symbol_idx = int(action[1])
        quantity_norm = action[2]
        price_norm = action[3]
        
        action_type = ActionType(action_type_idx)
        symbol = self.symbols[symbol_idx]
        
        # Denormalize quantity (0-1 -> 0-max_position_size)
        quantity = quantity_norm * self.max_position_size
        
        # Denormalize price (0-1 -> current_price ± 10%)
        current_price = self.market_states[symbol].mid_price
        price_range = current_price * 0.1
        price = current_price - price_range + (price_norm * 2 * price_range)
        
        return TradingAction(
            action_type=action_type,
            symbol=symbol,
            quantity=quantity,
            price=price if action_type in [ActionType.BUY_LIMIT, ActionType.SELL_LIMIT] else None
        )
    
    def _execute_action(self, action: TradingAction) -> float:
        """Execute a trading action and return reward"""
        
        if action.action_type == ActionType.HOLD:
            return 0.0
        
        # Calculate reward based on action
        reward = 0.0
        
        if action.action_type == ActionType.BUY_MARKET:
            # Simulate market buy
            price = self.market_states[action.symbol].best_ask
            cost = action.quantity * price * (1 + self.transaction_cost)
            
            if cost <= self.agent_state.available_cash:
                self.agent_state.cash -= cost
                self.agent_state.inventory[action.symbol] += action.quantity
                self.agent_state.available_cash -= cost
                
                # Reward based on expected profit
                reward = self._calculate_buy_reward(action.symbol, price)
        
        elif action.action_type == ActionType.SELL_MARKET:
            # Simulate market sell
            price = self.market_states[action.symbol].best_bid
            proceeds = action.quantity * price * (1 - self.transaction_cost)
            
            if self.agent_state.inventory[action.symbol] >= action.quantity:
                self.agent_state.cash += proceeds
                self.agent_state.inventory[action.symbol] -= action.quantity
                self.agent_state.available_cash += proceeds
                
                # Reward based on realized profit
                reward = self._calculate_sell_reward(action.symbol, price)
        
        return reward
    
    def _calculate_buy_reward(self, symbol: str, price: float) -> float:
        """Calculate reward for buying"""
        # Simple reward: negative for buying high, positive for buying low
        fair_value = self.market_states[symbol].moving_average_20
        return (fair_value - price) / fair_value
    
    def _calculate_sell_reward(self, symbol: str, price: float) -> float:
        """Calculate reward for selling"""
        # Simple reward: positive for selling high, negative for selling low
        fair_value = self.market_states[symbol].moving_average_20
        return (price - fair_value) / fair_value
    
    def _update_market_state(self):
        """Update market state (simulate price movements)"""
        for symbol in self.symbols:
            state = self.market_states[symbol]
            
            # Simulate price movement
            price_change = np.random.normal(0, state.volatility)
            new_mid_price = state.mid_price * (1 + price_change)
            
            # Update state
            state.mid_price = new_mid_price
            state.best_bid = new_mid_price - state.spread / 2
            state.best_ask = new_mid_price + state.spread / 2
            
            # Update price history
            self.price_history[symbol].append(new_mid_price)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol].pop(0)
            
            # Update technical indicators
            self._update_technical_indicators(symbol)
    
    def _update_technical_indicators(self, symbol: str):
        """Update technical indicators for a symbol"""
        prices = self.price_history[symbol]
        if len(prices) < 20:
            return
        
        state = self.market_states[symbol]
        
        # Moving averages
        state.moving_average_20 = np.mean(prices[-20:])
        if len(prices) >= 50:
            state.moving_average_50 = np.mean(prices[-50:])
        
        # Volatility
        if len(prices) >= 20:
            prices_array = np.array(prices[-20:])
            returns = np.diff(prices_array) / prices_array[:-1]
            state.volatility = np.std(returns)
        
        # RSI
        if len(prices) >= 14:
            gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
            losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
            
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                state.rsi = 100 - (100 / (1 + rs))
            else:
                state.rsi = 100
    
    def _update_agent_state(self):
        """Update agent state (calculate P&L, etc.)"""
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        position_value = 0.0
        
        for symbol, quantity in self.agent_state.inventory.items():
            if quantity != 0:
                current_price = self.market_states[symbol].mid_price
                position_value += abs(quantity * current_price)
                # Simplified P&L calculation
                unrealized_pnl += quantity * current_price
        
        self.agent_state.unrealized_pnl = unrealized_pnl
        self.agent_state.position_value = position_value
        self.agent_state.total_pnl = self.agent_state.realized_pnl + unrealized_pnl
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as numpy array"""
        obs = []
        
        # Market data for each symbol
        for symbol in self.symbols:
            state = self.market_states[symbol]
            
            # Basic market data
            obs.extend([
                state.best_bid,
                state.best_ask,
                state.mid_price,
                state.spread
            ])
            
            # Order book data (flattened)
            obs.extend(state.bid_prices)
            obs.extend(state.bid_quantities)
            obs.extend(state.ask_prices)
            obs.extend(state.ask_quantities)
            
            # Technical indicators
            obs.extend([
                state.moving_average_20,
                state.moving_average_50,
                state.volatility,
                state.rsi
            ])
        
        # Agent state
        obs.extend([
            self.agent_state.cash,
            self.agent_state.available_cash,
            self.agent_state.total_pnl,
            self.agent_state.unrealized_pnl,
            self.agent_state.realized_pnl,
            self.agent_state.position_value
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        return (self.current_step >= self.max_steps or 
                self.agent_state.cash <= 0 or
                self.agent_state.total_pnl < -self.initial_cash * 0.5)  # 50% drawdown
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            'step': self.current_step,
            'cash': self.agent_state.cash,
            'total_pnl': self.agent_state.total_pnl,
            'inventory': self.agent_state.inventory.copy(),
            'market_states': {s: {
                'mid_price': self.market_states[s].mid_price,
                'spread': self.market_states[s].spread
            } for s in self.symbols}
        }
    
    def render(self):
        """Render the environment (placeholder)"""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Cash: ${self.agent_state.cash:.2f}")
            print(f"Total P&L: ${self.agent_state.total_pnl:.2f}")
            print(f"Inventory: {self.agent_state.inventory}")
    
    def close(self):
        """Clean up resources"""
        if self.simulator_process:
            self.simulator_process.terminate()
            self.simulator_process.wait() 