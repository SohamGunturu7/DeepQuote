"""
Gymnasium environment backed by the DeepQuote C++ market simulator.

Each step:
  1. the agent's action is sent to the C++ matching engine as a real order
     (fills happen against the resting book, so the agent pays the spread),
  2. the market advances: the price model moves each symbol's fair price and
     a synchronous market maker re-quotes around it,
  3. the agent is marked to market and rewarded with its change in equity.

Action (Box, 4 values), shared with the rule-based agents in agents.py:
  [action_type, symbol_idx, quantity_frac, price_norm]
  action_type: 0 BUY_MARKET, 1 SELL_MARKET, 2 BUY_LIMIT, 3 SELL_LIMIT, 4 CANCEL_ALL, 5 HOLD
  quantity_frac in [0, 1] is scaled by max_order_size
  price_norm in [0, 1] maps to mid * (1 - price_band) .. mid * (1 + price_band) for limit orders

Observation: FEATURES_PER_SYMBOL values per symbol, followed by agent features
  per symbol: best_bid, best_ask, mid, spread, 3 bid prices, 3 bid sizes,
              3 ask prices, 3 ask sizes, volatility, 20-step moving average
  agent:      cash, position_value, unrealized_pnl, realized_pnl, total_pnl,
              then inventory for each symbol
"""

import os
import sys
from collections import deque
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def load_backend(backend: str):
    """Return the simulator module: "cpp" (the C++ extension) or "python" (pysim.py)."""
    if backend == "python":
        import pysim
        return pysim
    if backend != "cpp":
        raise ValueError(f"Unknown backend {backend!r}; expected 'cpp' or 'python'")
    try:
        import deepquote_simulator
    except ImportError:
        # Fall back to the CMake build directory (mkdir build && cd build && cmake .. && make)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
        try:
            import deepquote_simulator
        except ImportError as e:
            raise ImportError(
                "deepquote_simulator extension not found. Build it with `pip install .` "
                "from the repo root, or `mkdir build && cd build && cmake .. && make`."
            ) from e
    return deepquote_simulator


class ActionType(IntEnum):
    BUY_MARKET = 0
    SELL_MARKET = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    CANCEL_ALL = 4
    HOLD = 5


FEATURES_PER_SYMBOL = 18
BOOK_LEVELS = 3
AGENT_BASE_FEATURES = 5
# Simulated time per step, in years (one trading minute), for the GBM price model
ONE_MINUTE = 1.0 / (252 * 390)


class DeepQuoteEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 symbols: List[str] = ["AAPL"],
                 initial_cash: float = 100000.0,
                 initial_price: float = 100.0,
                 max_order_size: float = 100.0,
                 max_position: float = 1000.0,
                 max_steps: int = 1000,
                 dt: float = ONE_MINUTE,
                 price_band: float = 0.01,
                 order_ttl: int = 10,
                 mm_spread_pct: float = 0.001,
                 mm_order_size: float = 100.0,
                 mm_levels: int = 5,
                 event_probability: float = 0.002,
                 enable_events: bool = True,
                 verbose: bool = False,
                 trader_id: str = "agent_1",
                 strategy_type: str = "RL",
                 render_mode: Optional[str] = None,
                 backend: str = "cpp"):
        super().__init__()

        self.dq = dq = load_backend(backend)
        self.backend = backend

        self.symbols = list(symbols)
        self.initial_cash = initial_cash
        self.initial_price = initial_price
        self.max_order_size = max_order_size
        self.max_position = max_position
        self.max_steps = max_steps
        self.dt = dt
        self.price_band = price_band
        self.order_ttl = order_ttl
        self.render_mode = render_mode
        self.trader_id = trader_id

        self.sim = dq.MarketSimulator(self.symbols)
        self.sim.enable_market_events(enable_events)
        self.sim.set_event_probability(event_probability)
        self.sim.set_event_logging(verbose)

        mm_config = dq.MarketMakerConfig()
        mm_config.trader_id = "market_maker"
        mm_config.symbols = self.symbols
        mm_config.base_price = initial_price
        mm_config.spread_pct = mm_spread_pct
        mm_config.order_size = mm_order_size
        mm_config.max_orders_per_side = mm_levels
        # Driven synchronously via step(), never start()ed on a background thread
        self.market_maker = dq.MarketMaker(self.sim, mm_config)

        self.trader = dq.RLTrader(trader_id, strategy_type, initial_cash)
        self.sim.add_rl_trader(self.trader)

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([len(ActionType) - 1, len(self.symbols) - 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = FEATURES_PER_SYMBOL * len(self.symbols) + AGENT_BASE_FEATURES + len(self.symbols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.open_orders: List[Tuple[int, str, int]] = []  # (order_id, symbol, step placed)
        self.price_history: Dict[str, deque] = {s: deque(maxlen=50) for s in self.symbols}
        self.last_equity = initial_cash

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        # Seed the C++ random sources from gymnasium's RNG so reset(seed=...) is reproducible
        self.sim.seed(int(self.np_random.integers(0, 2**31 - 1)))
        self.sim.reset()
        self.market_maker.reset()
        self.trader.reset(self.initial_cash)
        self.trader.reset_episode()

        for symbol in self.symbols:
            self.sim.set_fair_price(symbol, self.initial_price)
            self.price_history[symbol].clear()
        self.market_maker.step()

        self.current_step = 0
        self.open_orders = []
        self._record_prices()
        self.last_equity = self._equity()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        self._execute_action(action)

        # Advance the market: move fair prices, then refresh liquidity around them
        self.sim.update_market_events(self.dt)
        self.market_maker.step()
        self._expire_orders()

        self.current_step += 1
        self._record_prices()

        equity = self._equity()
        self.trader.mark_to_market(self._mid_prices())
        reward = (equity - self.last_equity) / self.initial_cash * 100.0  # % return this step
        self.last_equity = equity
        self.trader.add_reward(reward)

        terminated = equity <= 0.0
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        mids = self._mid_prices()
        print(f"Step {self.current_step}: equity=${self._equity():,.2f} cash=${self.trader.get_cash():,.2f} "
              + " ".join(f"{s}: mid={mids[s]:.2f} inv={self.trader.get_inventory(s):.0f}" for s in self.symbols))

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Order handling
    # ------------------------------------------------------------------

    def _execute_action(self, action: np.ndarray):
        if action.size < 4:
            raise ValueError(f"Expected action of length 4, got {action.size}")

        action_type = ActionType(int(np.clip(np.rint(action[0]), 0, len(ActionType) - 1)))
        symbol = self.symbols[int(np.clip(np.rint(action[1]), 0, len(self.symbols) - 1))]
        quantity = float(np.clip(action[2], 0.0, 1.0)) * self.max_order_size
        price_norm = float(np.clip(action[3], 0.0, 1.0))

        if action_type == ActionType.HOLD:
            return
        if action_type == ActionType.CANCEL_ALL:
            self._cancel_all()
            return

        is_buy = action_type in (ActionType.BUY_MARKET, ActionType.BUY_LIMIT)
        quantity = self._limit_quantity(symbol, quantity, is_buy)
        if quantity < 1.0:
            return

        order = self.dq.Order()
        order.id = self.sim.next_order_id()
        order.side = self.dq.Side.BUY if is_buy else self.dq.Side.SELL
        order.quantity = float(np.floor(quantity))
        order.symbol = symbol
        order.trader_id = self.trader_id
        order.strategy_id = self.trader.get_strategy_type()

        if action_type in (ActionType.BUY_MARKET, ActionType.SELL_MARKET):
            order.type = self.dq.OrderType.MARKET
            order.price = 0.0
        else:
            mid = self._mid_price(symbol)
            order.type = self.dq.OrderType.LIMIT
            order.price = round(mid * (1.0 + self.price_band * (2.0 * price_norm - 1.0)), 2)
            if order.price <= 0:
                return

        self.sim.process_order(order)
        if order.type == self.dq.OrderType.LIMIT:
            self.open_orders.append((order.id, symbol, self.current_step))

    def _limit_quantity(self, symbol: str, quantity: float, is_buy: bool) -> float:
        # Keep inventory within +/- max_position and buys within available cash
        inventory = self.trader.get_inventory(symbol)
        if is_buy:
            quantity = min(quantity, self.max_position - inventory)
            ask = self.sim.get_best_ask(symbol)
            price = ask if ask > 0 else self._mid_price(symbol)
            quantity = min(quantity, max(self.trader.get_cash(), 0.0) / (price * (1 + self.price_band)))
        else:
            quantity = min(quantity, self.max_position + inventory)
        return max(quantity, 0.0)

    def _cancel_all(self):
        for order_id, symbol, _ in self.open_orders:
            self.sim.cancel_order(symbol, order_id)
        self.open_orders = []

    def _expire_orders(self):
        keep = []
        for order_id, symbol, placed in self.open_orders:
            if self.current_step - placed >= self.order_ttl:
                self.sim.cancel_order(symbol, order_id)
            else:
                keep.append((order_id, symbol, placed))
        self.open_orders = keep

    # ------------------------------------------------------------------
    # Market / account state
    # ------------------------------------------------------------------

    def _mid_price(self, symbol: str) -> float:
        bid, ask = self.sim.get_best_bid(symbol), self.sim.get_best_ask(symbol)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        fair = self.sim.get_fair_price(symbol)
        if fair > 0:
            return fair
        history = self.price_history[symbol]
        return history[-1] if history else self.initial_price

    def _mid_prices(self) -> Dict[str, float]:
        return {s: self._mid_price(s) for s in self.symbols}

    def _record_prices(self):
        for symbol in self.symbols:
            self.price_history[symbol].append(self._mid_price(symbol))

    def _position_value(self) -> float:
        return sum(self.trader.get_inventory(s) * self._mid_price(s) for s in self.symbols)

    def _equity(self) -> float:
        return self.trader.get_cash() + self._position_value()

    def _get_obs(self) -> np.ndarray:
        obs: List[float] = []
        for symbol in self.symbols:
            bid, ask = self.sim.get_best_bid(symbol), self.sim.get_best_ask(symbol)
            mid = self._mid_price(symbol)
            spread = ask - bid if bid > 0 and ask > 0 else 0.0

            snapshot = self.sim.get_snapshot(symbol)
            bid_levels = self._levels(snapshot.bids)
            ask_levels = self._levels(snapshot.asks)

            prices = np.array(self.price_history[symbol], dtype=np.float64)
            if len(prices) >= 2:
                volatility = float(np.std(np.diff(prices) / prices[:-1]))
            else:
                volatility = 0.0
            ma20 = float(np.mean(prices[-20:])) if len(prices) else mid

            obs.extend([bid, ask, mid, spread])
            obs.extend(p for p, _ in bid_levels)
            obs.extend(q for _, q in bid_levels)
            obs.extend(p for p, _ in ask_levels)
            obs.extend(q for _, q in ask_levels)
            obs.extend([volatility, ma20])

        realized = self.trader.get_realized_pnl()
        total_pnl = self._equity() - self.initial_cash
        obs.extend([
            self.trader.get_cash(),
            self._position_value(),
            total_pnl - realized,
            realized,
            total_pnl,
        ])
        obs.extend(self.trader.get_inventory(s) for s in self.symbols)
        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _levels(levels) -> List[Tuple[float, float]]:
        out = [(lvl.price, lvl.total_quantity) for lvl in levels[:BOOK_LEVELS]]
        out += [(0.0, 0.0)] * (BOOK_LEVELS - len(out))
        return out

    def _get_info(self) -> Dict[str, Any]:
        equity = self._equity()
        return {
            "step": self.current_step,
            "cash": self.trader.get_cash(),
            "equity": equity,
            "total_pnl": equity - self.initial_cash,
            "realized_pnl": self.trader.get_realized_pnl(),
            "position_value": self._position_value(),
            "inventory": {s: self.trader.get_inventory(s) for s in self.symbols},
            "mid_prices": self._mid_prices(),
            "open_orders": len(self.open_orders),
        }
