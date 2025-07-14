import gymnasium as gym
import numpy as np
from gymnasium import spaces
import deepquote_simulator as dq

# Gym environment wrapper for the DeepQuote C++ market simulator and RLTrader
class DeepQuoteEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, symbols=["AAPL"], initial_cash=100000.0, trader_id="agent_1", strategy_type="DQN"):
        super().__init__()
        self.symbols = symbols
        self.sim = dq.MarketSimulator(symbols)
        self.trader = dq.RLTrader(trader_id, strategy_type, initial_cash)
        self.sim.add_rl_trader(self.trader)
        self.current_step = 0
        self.max_steps = 1000
        self.next_order_id = 1
        
        self.action_space = spaces.Box(low=np.array([0, 0, 1]), high=np.array([1, 1000, 1000]), dtype=np.float32)
        
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
        if len(action) >= 4:
            action_type = int(action[0])
            quantity = float(action[2])
            price = float(action[3])
            
            if action_type == 5:
                self.current_step += 1
                obs = self._get_obs()
                reward = self.trader.get_episode_reward()
                done = self.current_step >= self.max_steps
                info = {"pnl": self.trader.get_realized_pnl(), "inventory": self.trader.get_inventory()}
                return obs, reward, done, info
            
            if action_type == 0:
                side = dq.Side.BUY
            elif action_type == 1:
                side = dq.Side.SELL
            else:
                self.current_step += 1
                obs = self._get_obs()
                reward = self.trader.get_episode_reward()
                done = self.current_step >= self.max_steps
                info = {"pnl": self.trader.get_realized_pnl(), "inventory": self.trader.get_inventory()}
                return obs, reward, done, info
        else:
            side = dq.Side.BUY if action[0] < 0.5 else dq.Side.SELL
            price = float(action[1])
            quantity = int(action[2])
        
        if quantity > 0 and price > 0:
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
        obs = []
        
        for symbol in self.symbols:
            best_bid = self.sim.get_best_bid(symbol)
            best_ask = self.sim.get_best_ask(symbol)
            
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 100.0
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.1
            
            inventory = self.trader.get_inventory()
            cash = self.trader.get_cash()
            
            symbol_features = [
                best_bid,
                best_ask,
                mid_price,
                spread,
                1000.0,
                mid_price + spread/2,
                mid_price - spread/2,
                mid_price,
                mid_price,
                inventory,
                cash,
                self.trader.get_realized_pnl(),
                self.trader.get_unrealized_pnl(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]
            
            obs.extend(symbol_features)
        
        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Inventory: {self.trader.get_inventory()}, Cash: {self.trader.get_cash()}, PnL: {self.trader.get_realized_pnl()}")

    def close(self):
        pass 