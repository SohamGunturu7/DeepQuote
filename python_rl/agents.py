"""
DeepQuote RL Agents

This module contains various reinforcement learning agents that can learn
trading strategies in the DeepQuote environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import wandb
import os

class TradingNetwork(nn.Module):
    """
    Neural network for trading decisions
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        features = self.feature_net(x)
        action = self.actor(features)
        value = self.critic(features)
        return action, value

class CustomPPOAgent:
    """
    Custom PPO agent for trading
    """
    
    def __init__(self, 
                 env: gym.Env,
                 learning_rate: float = 3e-4,
                 hidden_dim: int = 256,
                 device: str = "cpu"):
        
        self.env = env
        self.device = device
        
        # Network
        input_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.network = TradingNetwork(input_dim, action_dim, hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Get action from observation"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.network(obs_tensor)
            action = action.squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().numpy()
        
        # Add exploration noise
        noise = np.random.normal(0, 0.1, action.shape)
        action = np.clip(action + noise, -1, 1)
        
        return action, value, 0.0  # No log prob for now
    
    def update(self, batch: Dict[str, torch.Tensor]):
        """Update the agent using PPO"""
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_values = batch['values'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        # Forward pass
        new_actions, new_values = self.network(obs)
        
        # Calculate policy loss
        action_diff = new_actions - actions
        ratio = torch.exp(-0.5 * action_diff.pow(2).sum(dim=-1))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        value_loss = nn.MSELoss()(new_values.squeeze(), returns)
        
        # Calculate entropy loss
        entropy_loss = -torch.mean(new_actions.pow(2))
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path: str):
        """Save the agent"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
    
    def load(self, path: str):
        """Load the agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']

class MarketMakingAgent:
    """
    Specialized agent for market making strategies
    """
    
    def __init__(self, 
                 env: gym.Env,
                 spread_target: float = 0.001,
                 order_size: float = 10.0,
                 max_orders_per_side: int = 3):
        
        self.env = env
        self.spread_target = spread_target
        self.order_size = order_size
        self.max_orders_per_side = max_orders_per_side
        
        # Market making state
        self.active_orders = {}
        self.inventory = {symbol: 0.0 for symbol in env.symbols}
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate market making action"""
        # Parse observation to get market data
        market_data = self._parse_observation(obs)
        
        # Calculate optimal spreads based on inventory
        actions = []
        
        for symbol in self.env.symbols:
            mid_price = market_data[symbol]['mid_price']
            current_inventory = self.inventory[symbol]
            
            # Adjust spread based on inventory
            inventory_skew = current_inventory / self.order_size
            spread_adjustment = inventory_skew * 0.0001  # 1bp per unit of inventory
            
            bid_spread = self.spread_target + spread_adjustment
            ask_spread = self.spread_target - spread_adjustment
            
            # Place orders
            bid_price = mid_price - bid_spread / 2
            ask_price = mid_price + ask_spread / 2
            
            # Convert to action format
            bid_action = self._price_to_action(bid_price, mid_price, 0)  # BUY_LIMIT
            ask_action = self._price_to_action(ask_price, mid_price, 1)  # SELL_LIMIT
            
            actions.extend([bid_action, ask_action])
        
        # Pad to match action space
        while len(actions) < 4:
            actions.append(0.0)
        
        return np.array(actions[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        # This is a simplified parser - in practice you'd want more robust parsing
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18  # 18 features per symbol
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3]
            }
        
        return market_data
    
    def _price_to_action(self, price: float, mid_price: float, action_type: int) -> float:
        """Convert price to normalized action"""
        # Normalize price relative to mid price
        price_range = mid_price * 0.1
        normalized_price = (price - mid_price + price_range) / (2 * price_range)
        normalized_price = np.clip(normalized_price, 0, 1)
        
        return normalized_price

class MeanReversionAgent:
    """
    Specialized agent for mean reversion strategies
    """
    
    def __init__(self, 
                 env: gym.Env,
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        
        self.env = env
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # Price history
        self.price_history = {symbol: [] for symbol in env.symbols}
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate mean reversion action"""
        # Parse observation
        market_data = self._parse_observation(obs)
        
        actions = []
        
        for symbol in self.env.symbols:
            current_price = market_data[symbol]['mid_price']
            self.price_history[symbol].append(current_price)
            
            # Keep only recent prices
            if len(self.price_history[symbol]) > self.lookback_period:
                self.price_history[symbol].pop(0)
            
            # Calculate z-score
            if len(self.price_history[symbol]) >= self.lookback_period:
                prices = np.array(self.price_history[symbol])
                mean_price = np.mean(prices)
                std_price = np.std(prices)
                
                if std_price > 0:
                    z_score = (current_price - mean_price) / std_price
                else:
                    z_score = 0
            else:
                z_score = 0
            
            # Generate action based on z-score
            if z_score > self.entry_threshold:
                # Price is high, sell
                action = self._create_sell_action(current_price, 0.5)
            elif z_score < -self.entry_threshold:
                # Price is low, buy
                action = self._create_buy_action(current_price, 0.5)
            elif abs(z_score) < self.exit_threshold:
                # Close positions
                action = self._create_hold_action()
            else:
                # Hold
                action = self._create_hold_action()
            
            actions.extend(action)
        
        # Pad to match action space
        while len(actions) < 4:
            actions.append(0.0)
        
        return np.array(actions[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3]
            }
        
        return market_data
    
    def _create_buy_action(self, price: float, quantity: float) -> List[float]:
        """Create a buy action"""
        return [0, 0, quantity, 0.5]  # BUY_MARKET, symbol_idx=0, quantity, price
    
    def _create_sell_action(self, price: float, quantity: float) -> List[float]:
        """Create a sell action"""
        return [1, 0, quantity, 0.5]  # SELL_MARKET, symbol_idx=0, quantity, price
    
    def _create_hold_action(self) -> List[float]:
        """Create a hold action"""
        return [5, 0, 0, 0.5]  # HOLD, symbol_idx=0, quantity=0, price

class MomentumAgent:
    """
    Momentum-based trading agent that follows trends
    """
    
    def __init__(self, 
                 env: gym.Env,
                 short_window: int = 10,
                 long_window: int = 30,
                 momentum_threshold: float = 0.001,
                 position_size: float = 0.3):
        
        self.env = env
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size
        
        # Price history
        self.price_history = {symbol: [] for symbol in env.symbols}
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate momentum-based action"""
        market_data = self._parse_observation(obs)
        
        actions = []
        
        for symbol in self.env.symbols:
            current_price = market_data[symbol]['mid_price']
            self.price_history[symbol].append(current_price)
            
            # Keep only recent prices
            max_window = max(self.short_window, self.long_window)
            if len(self.price_history[symbol]) > max_window:
                self.price_history[symbol] = self.price_history[symbol][-max_window:]
            
            # Calculate momentum indicators
            if len(self.price_history[symbol]) >= self.long_window:
                prices = np.array(self.price_history[symbol])
                
                # Short-term momentum
                short_ma = np.mean(prices[-self.short_window:])
                # Long-term momentum
                long_ma = np.mean(prices[-self.long_window:])
                
                # Momentum signal
                momentum = (short_ma - long_ma) / long_ma
                
                # Generate action based on momentum
                if momentum > self.momentum_threshold:
                    # Strong upward momentum, buy
                    action = self._create_buy_action(current_price, self.position_size)
                elif momentum < -self.momentum_threshold:
                    # Strong downward momentum, sell
                    action = self._create_sell_action(current_price, self.position_size)
                else:
                    # Weak momentum, hold
                    action = self._create_hold_action()
            else:
                action = self._create_hold_action()
            
            actions.extend(action)
        
        # Pad to match action space
        while len(actions) < 4:
            actions.append(0.0)
        
        return np.array(actions[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3]
            }
        
        return market_data
    
    def _create_buy_action(self, price: float, quantity: float) -> List[float]:
        """Create a buy action"""
        return [0, 0, quantity, 0.5]  # BUY_MARKET, symbol_idx=0, quantity, price
    
    def _create_sell_action(self, price: float, quantity: float) -> List[float]:
        """Create a sell action"""
        return [1, 0, quantity, 0.5]  # SELL_MARKET, symbol_idx=0, quantity, price
    
    def _create_hold_action(self) -> List[float]:
        """Create a hold action"""
        return [5, 0, 0, 0.5]  # HOLD, symbol_idx=0, quantity=0, price

class ArbitrageAgent:
    """
    Statistical arbitrage agent that looks for price discrepancies
    """
    
    def __init__(self, 
                 env: gym.Env,
                 correlation_threshold: float = 0.8,
                 z_score_threshold: float = 2.0,
                 position_size: float = 0.2):
        
        self.env = env
        self.correlation_threshold = correlation_threshold
        self.z_score_threshold = z_score_threshold
        self.position_size = position_size
        
        # Price history for all symbols
        self.price_history = {symbol: [] for symbol in env.symbols}
        self.spread_history = []
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate arbitrage action"""
        market_data = self._parse_observation(obs)
        
        # Update price history
        for symbol in self.env.symbols:
            current_price = market_data[symbol]['mid_price']
            self.price_history[symbol].append(current_price)
            
            # Keep only recent prices
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Calculate spread between symbols if we have multiple
        if len(self.env.symbols) >= 2:
            spread = self.price_history[self.env.symbols[0]][-1] - self.price_history[self.env.symbols[1]][-1]
            self.spread_history.append(spread)
            
            if len(self.spread_history) > 100:
                self.spread_history = self.spread_history[-100:]
            
            # Calculate z-score of spread
            if len(self.spread_history) >= 20:
                spread_array = np.array(self.spread_history)
                mean_spread = np.mean(spread_array)
                std_spread = np.std(spread_array)
                
                if std_spread > 0:
                    z_score = (spread - mean_spread) / std_spread
                    
                    # Generate arbitrage action
                    if z_score > self.z_score_threshold:
                        # Spread is wide, sell first symbol, buy second
                        action = [1, 0, self.position_size, 0.5]  # SELL first symbol
                        action.extend([0, 1, self.position_size, 0.5])  # BUY second symbol
                    elif z_score < -self.z_score_threshold:
                        # Spread is narrow, buy first symbol, sell second
                        action = [0, 0, self.position_size, 0.5]  # BUY first symbol
                        action.extend([1, 1, self.position_size, 0.5])  # SELL second symbol
                    else:
                        # No arbitrage opportunity
                        action = [5, 0, 0, 0.5]  # HOLD
                        action.extend([5, 1, 0, 0.5])  # HOLD
                else:
                    action = [5, 0, 0, 0.5, 5, 1, 0, 0.5]  # HOLD both
            else:
                action = [5, 0, 0, 0.5, 5, 1, 0, 0.5]  # HOLD both
        else:
            # Single symbol, no arbitrage possible
            action = [5, 0, 0, 0.5]
        
        # Pad to match action space
        while len(action) < 4:
            action.append(0.0)
        
        return np.array(action[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3]
            }
        
        return market_data

class GridTradingAgent:
    """
    Grid trading agent that places orders at regular price intervals
    """
    
    def __init__(self, 
                 env: gym.Env,
                 grid_levels: int = 5,
                 grid_spacing: float = 0.01,
                 base_position_size: float = 10.0):
        
        self.env = env
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing
        self.base_position_size = base_position_size
        
        # Grid state
        self.grid_orders = {symbol: [] for symbol in env.symbols}
        self.base_prices = {symbol: None for symbol in env.symbols}
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate grid trading action"""
        market_data = self._parse_observation(obs)
        
        actions = []
        
        for symbol in self.env.symbols:
            current_price = market_data[symbol]['mid_price']
            
            # Initialize base price if not set
            if self.base_prices[symbol] is None:
                self.base_prices[symbol] = current_price
            
            # Calculate grid levels
            grid_prices = []
            for i in range(-self.grid_levels, self.grid_levels + 1):
                grid_price = self.base_prices[symbol] * (1 + i * self.grid_spacing)
                grid_prices.append(grid_price)
            
            # Find closest grid levels to current price
            closest_buy = None
            closest_sell = None
            
            for price in grid_prices:
                if price < current_price:
                    if closest_buy is None or price > closest_buy:
                        closest_buy = price
                elif price > current_price:
                    if closest_sell is None or price < closest_sell:
                        closest_sell = price
            
            # Generate actions
            if closest_buy is not None:
                # Place buy order at grid level
                buy_action = self._create_limit_buy_action(closest_buy, self.base_position_size)
                actions.extend(buy_action)
            else:
                actions.extend([5, 0, 0, 0.5])  # HOLD
            
            if closest_sell is not None:
                # Place sell order at grid level
                sell_action = self._create_limit_sell_action(closest_sell, self.base_position_size)
                actions.extend(sell_action)
            else:
                actions.extend([5, 0, 0, 0.5])  # HOLD
        
        # Pad to match action space
        while len(actions) < 4:
            actions.append(0.0)
        
        return np.array(actions[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3]
            }
        
        return market_data
    
    def _create_limit_buy_action(self, price: float, quantity: float) -> List[float]:
        """Create a limit buy action"""
        return [2, 0, quantity, 0.5]  # BUY_LIMIT, symbol_idx=0, quantity, price
    
    def _create_limit_sell_action(self, price: float, quantity: float) -> List[float]:
        """Create a limit sell action"""
        return [3, 0, quantity, 0.5]  # SELL_LIMIT, symbol_idx=0, quantity, price

class VolatilityBreakoutAgent:
    """
    Volatility breakout agent that trades on volatility expansion
    """
    
    def __init__(self, 
                 env: gym.Env,
                 volatility_window: int = 20,
                 breakout_threshold: float = 2.0,
                 position_size: float = 0.3):
        
        self.env = env
        self.volatility_window = volatility_window
        self.breakout_threshold = breakout_threshold
        self.position_size = position_size
        
        # Price and volatility history
        self.price_history = {symbol: [] for symbol in env.symbols}
        self.volatility_history = {symbol: [] for symbol in env.symbols}
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate volatility breakout action"""
        market_data = self._parse_observation(obs)
        
        actions = []
        
        for symbol in self.env.symbols:
            current_price = market_data[symbol]['mid_price']
            current_volatility = market_data[symbol].get('volatility', 0.01)
            
            self.price_history[symbol].append(current_price)
            self.volatility_history[symbol].append(current_volatility)
            
            # Keep only recent data
            if len(self.price_history[symbol]) > self.volatility_window:
                self.price_history[symbol] = self.price_history[symbol][-self.volatility_window:]
                self.volatility_history[symbol] = self.volatility_history[symbol][-self.volatility_window:]
            
            # Calculate volatility breakout signal
            if len(self.volatility_history[symbol]) >= self.volatility_window:
                recent_vol = np.mean(self.volatility_history[symbol][-5:])  # Recent volatility
                historical_vol = np.mean(self.volatility_history[symbol][:-5])  # Historical volatility
                
                if historical_vol > 0:
                    vol_ratio = recent_vol / historical_vol
                    
                    # Check for breakout
                    if vol_ratio > self.breakout_threshold:
                        # High volatility breakout - momentum strategy
                        price_change = current_price - self.price_history[symbol][-2]
                        
                        if price_change > 0:
                            # Upward breakout
                            action = self._create_buy_action(current_price, self.position_size)
                        else:
                            # Downward breakout
                            action = self._create_sell_action(current_price, self.position_size)
                    else:
                        # Normal volatility, mean reversion
                        prices = np.array(self.price_history[symbol])
                        mean_price = np.mean(prices)
                        
                        if current_price > mean_price * 1.01:
                            action = self._create_sell_action(current_price, self.position_size * 0.5)
                        elif current_price < mean_price * 0.99:
                            action = self._create_buy_action(current_price, self.position_size * 0.5)
                        else:
                            action = self._create_hold_action()
                else:
                    action = self._create_hold_action()
            else:
                action = self._create_hold_action()
            
            actions.extend(action)
        
        # Pad to match action space
        while len(actions) < 4:
            actions.append(0.0)
        
        return np.array(actions[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3],
                'volatility': obs[offset + 16] if len(obs) > offset + 16 else 0.01  # Volatility from technical indicators
            }
        
        return market_data
    
    def _create_buy_action(self, price: float, quantity: float) -> List[float]:
        """Create a buy action"""
        return [0, 0, quantity, 0.5]  # BUY_MARKET, symbol_idx=0, quantity, price
    
    def _create_sell_action(self, price: float, quantity: float) -> List[float]:
        """Create a sell action"""
        return [1, 0, quantity, 0.5]  # SELL_MARKET, symbol_idx=0, quantity, price
    
    def _create_hold_action(self) -> List[float]:
        """Create a hold action"""
        return [5, 0, 0, 0.5]  # HOLD, symbol_idx=0, quantity=0, price

class PairsTradingAgent:
    """
    Pairs trading agent that trades correlated assets
    """
    
    def __init__(self, 
                 env: gym.Env,
                 correlation_window: int = 50,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 position_size: float = 0.2):
        
        self.env = env
        self.correlation_window = correlation_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        
        # Price history for pairs
        self.price_history = {symbol: [] for symbol in env.symbols}
        self.spread_history = []
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Generate pairs trading action"""
        market_data = self._parse_observation(obs)
        
        # Update price history
        for symbol in self.env.symbols:
            current_price = market_data[symbol]['mid_price']
            self.price_history[symbol].append(current_price)
            
            # Keep only recent prices
            if len(self.price_history[symbol]) > self.correlation_window:
                self.price_history[symbol] = self.price_history[symbol][-self.correlation_window:]
        
        # Calculate spread between pairs if we have multiple symbols
        if len(self.env.symbols) >= 2:
            spread = self.price_history[self.env.symbols[0]][-1] - self.price_history[self.env.symbols[1]][-1]
            self.spread_history.append(spread)
            
            if len(self.spread_history) > self.correlation_window:
                self.spread_history = self.spread_history[-self.correlation_window:]
            
            # Calculate z-score of spread
            if len(self.spread_history) >= 20:
                spread_array = np.array(self.spread_history)
                mean_spread = np.mean(spread_array)
                std_spread = np.std(spread_array)
                
                if std_spread > 0:
                    z_score = (spread - mean_spread) / std_spread
                    
                    # Generate pairs trading action
                    if z_score > self.entry_threshold:
                        # Spread is wide, short the spread (sell first, buy second)
                        action = [1, 0, self.position_size, 0.5]  # SELL first symbol
                        action.extend([0, 1, self.position_size, 0.5])  # BUY second symbol
                    elif z_score < -self.entry_threshold:
                        # Spread is narrow, long the spread (buy first, sell second)
                        action = [0, 0, self.position_size, 0.5]  # BUY first symbol
                        action.extend([1, 1, self.position_size, 0.5])  # SELL second symbol
                    elif abs(z_score) < self.exit_threshold:
                        # Close positions
                        action = [4, 0, 0, 0.5]  # CANCEL_ALL for first symbol
                        action.extend([4, 1, 0, 0.5])  # CANCEL_ALL for second symbol
                    else:
                        # Hold positions
                        action = [5, 0, 0, 0.5, 5, 1, 0, 0.5]  # HOLD both
                else:
                    action = [5, 0, 0, 0.5, 5, 1, 0, 0.5]  # HOLD both
            else:
                action = [5, 0, 0, 0.5, 5, 1, 0, 0.5]  # HOLD both
        else:
            # Single symbol, no pairs trading possible
            action = [5, 0, 0, 0.5]
        
        # Pad to match action space
        while len(action) < 4:
            action.append(0.0)
        
        return np.array(action[:4])
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Parse observation into market data"""
        market_data = {}
        
        for i, symbol in enumerate(self.env.symbols):
            offset = i * 18
            market_data[symbol] = {
                'best_bid': obs[offset],
                'best_ask': obs[offset + 1],
                'mid_price': obs[offset + 2],
                'spread': obs[offset + 3]
            }
        
        return market_data

class StableBaselinesAgent:
    """
    Wrapper for Stable Baselines3 agents
    """
    
    def __init__(self, 
                 env: gym.Env,
                 agent_type: str = "PPO",
                 learning_rate: float = 3e-4,
                 tensorboard_log: Optional[str] = None):
        
        self.env = env
        self.agent_type = agent_type
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        # Create agent
        if agent_type == "PPO":
            self.agent = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=tensorboard_log
            )
        elif agent_type == "SAC":
            self.agent = SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                verbose=1,
                tensorboard_log=tensorboard_log
            )
        elif agent_type == "TD3":
            self.agent = TD3(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=1,
                tensorboard_log=tensorboard_log
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.vec_env = vec_env
    
    def train(self, total_timesteps: int, callback: Optional[Any] = None):
        """Train the agent"""
        self.agent.learn(total_timesteps=total_timesteps, callback=callback)
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from observation"""
        action, _ = self.agent.predict(obs, deterministic=True)
        return action
    
    def save(self, path: str):
        """Save the agent"""
        self.agent.save(path)
        self.vec_env.save(f"{path}_vec_normalize.pkl")
    
    def load(self, path: str):
        """Load the agent"""
        self.agent = self.agent.load(path)
        self.vec_env = VecNormalize.load(f"{path}_vec_normalize.pkl", self.vec_env)

def create_agent(agent_type: str, env: gym.Env, **kwargs) -> Any:
    """Factory function to create agents"""
    
    if agent_type == "CustomPPO":
        return CustomPPOAgent(env, **kwargs)
    elif agent_type == "MarketMaking":
        return MarketMakingAgent(env, **kwargs)
    elif agent_type == "MeanReversion":
        return MeanReversionAgent(env, **kwargs)
    elif agent_type == "Momentum":
        return MomentumAgent(env, **kwargs)
    elif agent_type == "Arbitrage":
        return ArbitrageAgent(env, **kwargs)
    elif agent_type == "GridTrading":
        return GridTradingAgent(env, **kwargs)
    elif agent_type == "VolatilityBreakout":
        return VolatilityBreakoutAgent(env, **kwargs)
    elif agent_type == "PairsTrading":
        return PairsTradingAgent(env, **kwargs)
    elif agent_type in ["PPO", "SAC", "TD3", "A2C"]:
        return StableBaselinesAgent(env, agent_type, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 