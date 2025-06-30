import gymnasium as gym
import numpy as np
from gymnasium import spaces
import deepquote_simulator as dq

class DeepQuoteEnv(gym.Env):
    """
    Gym environment wrapper for the DeepQuote C++ market simulator and RLTrader.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, symbols=["AAPL"], initial_cash=100000.0, trader_id="agent_1", strategy_type="DQN"):
        super().__init__()
        self.symbols = symbols
        self.sim = dq.MarketSimulator(symbols)
        self.trader = dq.RLTrader(trader_id, strategy_type, initial_cash)
        self.sim.add_rl_trader(self.trader)
        self.current_step = 0
        self.max_steps = 1000
        self.next_order_id = 1  # Order ID generator
        
        # Action space: [side, price, quantity] for our environment
        self.action_space = spaces.Box(low=np.array([0, 0, 1]), high=np.array([1, 1000, 1000]), dtype=np.float32)
        
        # Observation space: 18 features per symbol (as expected by agents)
        # [best_bid, best_ask, mid_price, spread, volume, high, low, open, close, ...]
        obs_dim = len(symbols) * 18
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.symbol = symbols[0]

    def reset(self):
        self.sim.reset()
        self.trader.reset()
        self.current_step = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        # Convert agent action format to our environment format
        # Agent action: [action_type, symbol_idx, quantity, price] (4 values)
        # Our action: [side, price, quantity] (3 values)
        
        if len(action) >= 4:
            # Agent format: [action_type, symbol_idx, quantity, price]
            action_type = int(action[0])
            quantity = float(action[2])
            price = float(action[3])
            
            # Handle HOLD action (action_type=5) - skip order creation
            if action_type == 5:  # HOLD
                self.current_step += 1
                obs = self._get_obs()
                reward = self.trader.get_episode_reward()
                done = self.current_step >= self.max_steps
                info = {"pnl": self.trader.get_realized_pnl(), "inventory": self.trader.get_inventory()}
                return obs, reward, done, info
            
            # Convert action_type to side
            if action_type == 0:  # BUY
                side = dq.Side.BUY
            elif action_type == 1:  # SELL
                side = dq.Side.SELL
            else:  # Unknown action type, default to HOLD
                self.current_step += 1
                obs = self._get_obs()
                reward = self.trader.get_episode_reward()
                done = self.current_step >= self.max_steps
                info = {"pnl": self.trader.get_realized_pnl(), "inventory": self.trader.get_inventory()}
                return obs, reward, done, info
        else:
            # Our format: [side, price, quantity]
            side = dq.Side.BUY if action[0] < 0.5 else dq.Side.SELL
            price = float(action[1])
            quantity = int(action[2])
        
        # Only create order if quantity is valid
        if quantity > 0 and price > 0:
            # Create and process order
            order = dq.Order()
            order.id = self.next_order_id
            order.side = side
            order.type = dq.OrderType.LIMIT
            order.price = price
            order.quantity = quantity
            order.symbol = self.symbol
            order.trader_id = self.trader.get_id()
            order.strategy_id = self.trader.get_strategy_type()
            
            self.sim.process_order(order)
            self.next_order_id += 1
        
        self.current_step += 1
        obs = self._get_obs()
        reward = self.trader.get_episode_reward()
        done = self.current_step >= self.max_steps
        info = {"pnl": self.trader.get_realized_pnl(), "inventory": self.trader.get_inventory()}
        return obs, reward, done, info

    def _get_obs(self):
        """Get observation in the format expected by agents (18 features per symbol)"""
        obs = []
        
        for symbol in self.symbols:
            best_bid = self.sim.get_best_bid(symbol)
            best_ask = self.sim.get_best_ask(symbol)
            
            # Calculate mid price and spread
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 100.0
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.1
            
            # Get trader info
            inventory = self.trader.get_inventory()
            cash = self.trader.get_cash()
            
            # Create 18 features per symbol (as expected by agents)
            symbol_features = [
                best_bid,                    # 0: best_bid
                best_ask,                    # 1: best_ask  
                mid_price,                   # 2: mid_price
                spread,                      # 3: spread
                1000.0,                      # 4: volume (placeholder)
                mid_price + spread/2,        # 5: high (placeholder)
                mid_price - spread/2,        # 6: low (placeholder)
                mid_price,                   # 7: open (placeholder)
                mid_price,                   # 8: close (placeholder)
                inventory,                   # 9: inventory
                cash,                        # 10: cash
                self.trader.get_realized_pnl(), # 11: realized_pnl
                self.trader.get_unrealized_pnl(), # 12: unrealized_pnl
                0.0,                         # 13: position_value
                0.0,                         # 14: position_cost
                0.0,                         # 15: position_pnl
                0.0,                         # 16: position_return
                0.0                          # 17: market_return
            ]
            
            obs.extend(symbol_features)
        
        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Inventory: {self.trader.get_inventory()}, Cash: {self.trader.get_cash()}, PnL: {self.trader.get_realized_pnl()}")

    def close(self):
        pass 