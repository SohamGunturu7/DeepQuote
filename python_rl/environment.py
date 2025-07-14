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

# Market state representation
@dataclass
class MarketState:
    symbol: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    volume: float
    timestamp: float
    
    bid_prices: List[float]
    bid_quantities: List[float]
    ask_prices: List[float]
    ask_quantities: List[float]
    
    moving_average_20: float
    moving_average_50: float
    volatility: float
    rsi: float

# Agent state representation
@dataclass
class AgentState:
    cash: float
    inventory: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    position_value: float
    available_cash: float

# Action types
class ActionType(Enum):
    BUY_MARKET = 0
    SELL_MARKET = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    CANCEL_ALL = 4
    HOLD = 5

# Trading action representation
@dataclass
class TradingAction:
    action_type: ActionType
    symbol: str
    quantity: float
    price: Optional[float] = None

# Main trading environment
class DeepQuoteEnv(gym.Env):
    
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
        
        self.current_step = 0
        self.max_steps = 1000
        
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([len(ActionType) - 1, len(symbols) - 1, 1, 1]),
            dtype=np.float32
        )
        
        market_features = 4 + 10 + 4
        agent_features = 6
        total_features = market_features * len(symbols) + agent_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )
        
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
        
        self.price_history: Dict[str, List[float]] = {
            symbol: [] for symbol in symbols
        }
        
        self.simulator_process = None
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        
        self.agent_state = AgentState(
            cash=self.initial_cash,
            inventory={symbol: 0.0 for symbol in self.symbols},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            position_value=0.0,
            available_cash=self.initial_cash
        )
        
        for symbol in self.symbols:
            self.price_history[symbol] = []
        
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
        trading_action = self._parse_action(action)
        
        reward = self._execute_action(trading_action)
        
        self._update_market_state()
        
        self._update_agent_state()
        
        observation = self._get_observation()
        
        done = self._is_done()
        
        self.current_step += 1
        
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def _parse_action(self, action: np.ndarray) -> TradingAction:
        action_type_idx = int(action[0])
        symbol_idx = int(action[1])
        quantity_normalized = action[2]
        price_normalized = action[3]
        
        action_type = ActionType(action_type_idx)
        symbol = self.symbols[symbol_idx]
        
        quantity = quantity_normalized * self.max_position_size
        
        price = None
        if action_type in [ActionType.BUY_LIMIT, ActionType.SELL_LIMIT]:
            market_state = self.market_states[symbol]
            price_range = market_state.mid_price * 0.1
            price = market_state.mid_price - price_range + 2 * price_range * price_normalized
        
        return TradingAction(
            action_type=action_type,
            symbol=symbol,
            quantity=quantity,
            price=price
        )
    
    def _execute_action(self, action: TradingAction) -> float:
        reward = 0.0
        
        if action.action_type == ActionType.BUY_MARKET:
            reward = self._calculate_buy_reward(action.symbol, self.market_states[action.symbol].best_ask)
            self.agent_state.inventory[action.symbol] += action.quantity
            self.agent_state.cash -= action.quantity * self.market_states[action.symbol].best_ask * (1 + self.transaction_cost)
            
        elif action.action_type == ActionType.SELL_MARKET:
            reward = self._calculate_sell_reward(action.symbol, self.market_states[action.symbol].best_bid)
            self.agent_state.inventory[action.symbol] -= action.quantity
            self.agent_state.cash += action.quantity * self.market_states[action.symbol].best_bid * (1 - self.transaction_cost)
            
        elif action.action_type == ActionType.BUY_LIMIT:
            if action.price < self.market_states[action.symbol].best_ask:
                reward = self._calculate_buy_reward(action.symbol, action.price)
                self.agent_state.inventory[action.symbol] += action.quantity
                self.agent_state.cash -= action.quantity * action.price * (1 + self.transaction_cost)
                
        elif action.action_type == ActionType.SELL_LIMIT:
            if action.price > self.market_states[action.symbol].best_bid:
                reward = self._calculate_sell_reward(action.symbol, action.price)
                self.agent_state.inventory[action.symbol] -= action.quantity
                self.agent_state.cash += action.quantity * action.price * (1 - self.transaction_cost)
        
        return reward
    
    def _calculate_buy_reward(self, symbol: str, price: float) -> float:
        return -price * 0.001
    
    def _calculate_sell_reward(self, symbol: str, price: float) -> float:
        return price * 0.001
    
    def _update_market_state(self):
        for symbol in self.symbols:
            market_state = self.market_states[symbol]
            
            price_change = np.random.normal(0, 0.01)
            market_state.mid_price *= (1 + price_change)
            market_state.best_bid = market_state.mid_price - market_state.spread / 2
            market_state.best_ask = market_state.mid_price + market_state.spread / 2
            
            self.price_history[symbol].append(market_state.mid_price)
            
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            self._update_technical_indicators(symbol)
    
    def _update_technical_indicators(self, symbol: str):
        prices = self.price_history[symbol]
        if len(prices) < 20:
            return
        
        market_state = self.market_states[symbol]
        
        if len(prices) >= 20:
            market_state.moving_average_20 = np.mean(prices[-20:])
        
        if len(prices) >= 50:
            market_state.moving_average_50 = np.mean(prices[-50:])
        
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-21:-1]
            market_state.volatility = np.std(returns)
        
        if len(prices) >= 14:
            gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
            losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
            
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                market_state.rsi = 100 - (100 / (1 + rs))
            else:
                market_state.rsi = 100
    
    def _update_agent_state(self):
        self.agent_state.position_value = 0.0
        
        for symbol, quantity in self.agent_state.inventory.items():
            if quantity != 0:
                market_state = self.market_states[symbol]
                self.agent_state.position_value += abs(quantity) * market_state.mid_price
        
        self.agent_state.unrealized_pnl = self.agent_state.position_value - self.agent_state.cash
        self.agent_state.total_pnl = self.agent_state.realized_pnl + self.agent_state.unrealized_pnl
        self.agent_state.available_cash = self.agent_state.cash
    
    def _get_observation(self) -> np.ndarray:
        observation = []
        
        for symbol in self.symbols:
            market_state = self.market_states[symbol]
            
            symbol_features = [
                market_state.best_bid,
                market_state.best_ask,
                market_state.mid_price,
                market_state.spread,
                *market_state.bid_prices,
                *market_state.bid_quantities,
                *market_state.ask_prices,
                *market_state.ask_quantities,
                market_state.moving_average_20,
                market_state.moving_average_50,
                market_state.volatility,
                market_state.rsi
            ]
            
            observation.extend(symbol_features)
        
        agent_features = [
            self.agent_state.cash,
            self.agent_state.position_value,
            self.agent_state.unrealized_pnl,
            self.agent_state.realized_pnl,
            self.agent_state.total_pnl,
            self.agent_state.available_cash
        ]
        
        observation.extend(agent_features)
        
        return np.array(observation, dtype=np.float32)
    
    def _is_done(self) -> bool:
        return self.current_step >= self.max_steps
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'step': self.current_step,
            'cash': self.agent_state.cash,
            'total_pnl': self.agent_state.total_pnl,
            'position_value': self.agent_state.position_value,
            'inventory': self.agent_state.inventory.copy()
        }
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Cash: ${self.agent_state.cash:.2f}")
            print(f"Total PnL: ${self.agent_state.total_pnl:.2f}")
            print(f"Position Value: ${self.agent_state.position_value:.2f}")
            print("Inventory:", self.agent_state.inventory)
            print("---")
    
    def close(self):
        if self.simulator_process:
            self.simulator_process.terminate()
            self.simulator_process.wait() 