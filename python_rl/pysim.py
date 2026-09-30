"""
Pure-Python port of the DeepQuote C++ market simulator.

Mirrors the C++ classes (order book, matching engine, traders, market maker,
price models, market events) and exposes the same API as the
deepquote_simulator extension module, so DeepQuoteEnv can run on either
backend. It exists to benchmark the C++ core against an equivalent Python
implementation; see benchmark.py.

Matching and accounting follow the C++ logic line for line. Random numbers come
from Python's generator instead of std::mt19937, so seeded price paths differ
between backends even though their statistics are the same.
"""

import bisect
import math
import random
from enum import Enum
from typing import Dict, List, Optional

QUANTITY_EPSILON = 1e-9
PRICE_EPSILON = 1e-9
# dt is in years (the price models' unit); events are timed in trading seconds
TRADING_SECONDS_PER_YEAR = 252.0 * 6.5 * 3600.0


def _round2(value: float) -> float:
    # Matches C++ std::round (half away from zero), unlike Python's banker's rounding
    return math.copysign(math.floor(abs(value) * 100.0 + 0.5), value) / 100.0


def _valid_price(price: float) -> bool:
    return price > PRICE_EPSILON and math.isfinite(price)


def _valid_quantity(qty: float) -> bool:
    return qty > QUANTITY_EPSILON and math.isfinite(qty)


# ============================================================================
# Core types
# ============================================================================

class Side(Enum):
    BUY = 0
    SELL = 1


class OrderType(Enum):
    MARKET = 0
    LIMIT = 1
    CANCEL = 2


class OrderStatus(Enum):
    PENDING = 0
    PARTIAL = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4


class Order:
    __slots__ = ("id", "side", "type", "price", "quantity", "filled_quantity",
                 "symbol", "trader_id", "strategy_id", "status")

    def __init__(self):
        self.id = 0
        self.side = Side.BUY
        self.type = OrderType.LIMIT
        self.price = 0.0
        self.quantity = 0.0
        self.filled_quantity = 0.0
        self.symbol = ""
        self.trader_id = ""
        self.strategy_id = ""
        self.status = OrderStatus.PENDING

    def is_valid(self) -> bool:
        return (self.id > 0 and
                (self.type == OrderType.MARKET or _valid_price(self.price)) and
                _valid_quantity(self.quantity) and
                bool(self.symbol) and bool(self.trader_id) and
                0 <= self.filled_quantity <= self.quantity)

    def is_fully_filled(self) -> bool:
        return (self.status == OrderStatus.FILLED or
                self.filled_quantity >= self.quantity - QUANTITY_EPSILON)

    def is_partially_filled(self) -> bool:
        return (self.status == OrderStatus.PARTIAL or
                (self.filled_quantity > QUANTITY_EPSILON and not self.is_fully_filled()))

    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    def get_remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity


class Trade:
    __slots__ = ("buy_order_id", "sell_order_id", "price", "quantity", "symbol")

    def __init__(self, buy_order_id, sell_order_id, price, quantity, symbol):
        self.buy_order_id = buy_order_id
        self.sell_order_id = sell_order_id
        self.price = price
        self.quantity = quantity
        self.symbol = symbol


class OrderBookLevel:
    __slots__ = ("price", "total_quantity", "order_count")

    def __init__(self, price, total_quantity, order_count):
        self.price = price
        self.total_quantity = total_quantity
        self.order_count = order_count


class OrderBookSnapshot:
    __slots__ = ("symbol", "bids", "asks", "mid_price", "spread", "bid_depth", "ask_depth")


# ============================================================================
# Order book: price levels kept in sorted order (like the C++ std::map)
# ============================================================================

class _Levels:
    """Price -> list of orders, iterated best price first."""

    def __init__(self, descending: bool):
        self.descending = descending
        self.orders: Dict[float, List[Order]] = {}
        self.prices: List[float] = []  # ascending

    def best(self) -> float:
        if not self.prices:
            return 0.0
        return self.prices[-1] if self.descending else self.prices[0]

    def add(self, order: Order):
        level = self.orders.get(order.price)
        if level is None:
            self.orders[order.price] = [order]
            bisect.insort(self.prices, order.price)
        else:
            level.append(order)

    def remove(self, order_id: int, price: float):
        level = self.orders.get(price)
        if level is None:
            return
        level[:] = [o for o in level if o.id != order_id]
        if not level:
            del self.orders[price]
            del self.prices[bisect.bisect_left(self.prices, price)]

    def iter_best_first(self):
        prices = reversed(self.prices) if self.descending else self.prices
        for price in prices:
            yield price, self.orders[price]


class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = _Levels(descending=True)
        self.asks = _Levels(descending=False)
        self.orders_by_id: Dict[int, Order] = {}

    def _side(self, side: Side) -> _Levels:
        return self.bids if side == Side.BUY else self.asks

    def add_order(self, order: Order) -> bool:
        if not (order.is_valid() and order.symbol == self.symbol and order.is_active()):
            return False
        if order.id in self.orders_by_id:
            return False
        self._side(order.side).add(order)
        self.orders_by_id[order.id] = order
        return True

    def cancel_order(self, order_id: int) -> bool:
        order = self.orders_by_id.get(order_id)
        if order is None or not order.is_active():
            return False
        self._side(order.side).remove(order_id, order.price)
        del self.orders_by_id[order_id]
        order.status = OrderStatus.CANCELLED
        return True

    def remove_inactive_orders(self, price: float, side: Side):
        levels = self._side(side)
        for order in list(levels.orders.get(price, ())):
            if not order.is_active():
                levels.remove(order.id, price)
                self.orders_by_id.pop(order.id, None)

    def get_orders_at_price(self, price: float, side: Side) -> List[Order]:
        return list(self._side(side).orders.get(price, ()))

    def best_bid(self) -> float:
        return self.bids.best()

    def best_ask(self) -> float:
        return self.asks.best()

    def mid_price(self) -> float:
        bid, ask = self.best_bid(), self.best_ask()
        return (bid + ask) / 2.0 if bid > 0 and ask > 0 else math.nan

    def spread(self) -> float:
        bid, ask = self.best_bid(), self.best_ask()
        return ask - bid if bid > 0 and ask > 0 else 0.0

    @staticmethod
    def _depth(levels: _Levels) -> float:
        return sum(o.get_remaining_quantity() for lvl in levels.orders.values() for o in lvl)

    def bid_depth(self) -> float:
        return self._depth(self.bids)

    def ask_depth(self) -> float:
        return self._depth(self.asks)

    @staticmethod
    def _levels(levels: _Levels, max_levels: int) -> List[OrderBookLevel]:
        out = []
        for price, orders in levels.iter_best_first():
            if len(out) >= max_levels:
                break
            total = sum(o.get_remaining_quantity() for o in orders)
            out.append(OrderBookLevel(price, total, len(orders)))
        return out

    def snapshot(self) -> OrderBookSnapshot:
        snap = OrderBookSnapshot()
        snap.symbol = self.symbol
        snap.bids = self._levels(self.bids, 10)
        snap.asks = self._levels(self.asks, 10)
        snap.mid_price = self.mid_price()
        snap.spread = self.spread()
        snap.bid_depth = self.bid_depth()
        snap.ask_depth = self.ask_depth()
        return snap

    def order_count(self) -> int:
        return len(self.orders_by_id)


# ============================================================================
# Matching engine
# ============================================================================

class MatchingEngine:
    def __init__(self, symbol: str):
        self.book = OrderBook(symbol)
        self.trade_callback = None

    def process_order(self, order: Order) -> List[Trade]:
        if not order.is_valid():
            raise ValueError("Invalid order")
        if order.type == OrderType.MARKET:
            return self._process_market(order)
        if order.type == OrderType.LIMIT:
            return self._process_limit(order)
        self.book.cancel_order(order.id)
        return []

    def cancel_order(self, order_id: int) -> bool:
        return self.book.cancel_order(order_id)

    def _match_level(self, order: Order, best_price: float, trades: List[Trade]):
        opposite = Side.SELL if order.side == Side.BUY else Side.BUY
        for resting in self.book.get_orders_at_price(best_price, opposite):
            if order.get_remaining_quantity() <= QUANTITY_EPSILON:
                break
            if not resting.is_active():
                continue
            qty = min(order.get_remaining_quantity(), resting.get_remaining_quantity())
            if order.side == Side.BUY:
                trade = Trade(order.id, resting.id, best_price, qty, self.book.symbol)
            else:
                trade = Trade(resting.id, order.id, best_price, qty, self.book.symbol)
            trades.append(trade)
            order.filled_quantity += qty
            resting.filled_quantity += qty
            self._update_status(order)
            self._update_status(resting)
            if self.trade_callback:
                self.trade_callback(trade)
        # Remove filled/cancelled orders so the level empties and the loop advances
        self.book.remove_inactive_orders(best_price, opposite)

    def _process_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        while order.get_remaining_quantity() > QUANTITY_EPSILON:
            best = self.book.best_ask() if order.side == Side.BUY else self.book.best_bid()
            if best <= 0:
                break
            opposite = Side.SELL if order.side == Side.BUY else Side.BUY
            if not self.book.get_orders_at_price(best, opposite):
                break
            self._match_level(order, best, trades)
        if order.get_remaining_quantity() > QUANTITY_EPSILON:
            order.status = OrderStatus.REJECTED
        return trades

    def _process_limit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        while order.get_remaining_quantity() > QUANTITY_EPSILON:
            best = self.book.best_ask() if order.side == Side.BUY else self.book.best_bid()
            if best <= 0:
                break
            if order.side == Side.BUY and order.price < best:
                break
            if order.side == Side.SELL and order.price > best:
                break
            opposite = Side.SELL if order.side == Side.BUY else Side.BUY
            if not self.book.get_orders_at_price(best, opposite):
                break
            self._match_level(order, best, trades)
        if order.get_remaining_quantity() > QUANTITY_EPSILON:
            self.book.add_order(order)
        return trades

    @staticmethod
    def _update_status(order: Order):
        if order.is_fully_filled():
            order.status = OrderStatus.FILLED
        elif order.is_partially_filled():
            order.status = OrderStatus.PARTIAL


# ============================================================================
# Traders
# ============================================================================

class _Position:
    __slots__ = ("quantity", "average_cost", "total_cost")

    def __init__(self, quantity: float, price: float):
        self.quantity = quantity
        self.average_cost = price
        self.total_cost = quantity * price


class Trader:
    def __init__(self, trader_id: str, initial_cash: float = 100000.0):
        self.trader_id = trader_id
        self._cash = initial_cash
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self.positions: Dict[str, _Position] = {}
        self.trade_history: List[Trade] = []

    def get_id(self) -> str:
        return self.trader_id

    def get_cash(self) -> float:
        return _round2(self._cash)

    def get_inventory(self, symbol: Optional[str] = None) -> float:
        if symbol is None:
            return next(iter(self.positions.values())).quantity if self.positions else 0.0
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    def get_realized_pnl(self) -> float:
        return _round2(self._realized_pnl)

    def get_unrealized_pnl(self, mark_prices: Optional[Dict[str, float]] = None) -> float:
        if mark_prices is None:
            return 0.0
        total = 0.0
        for symbol, pos in self.positions.items():
            price = mark_prices.get(symbol)
            if price is not None and not math.isnan(price):
                total += (price - pos.average_cost) * pos.quantity
        return _round2(total)

    def on_trade(self, trade: Trade, is_buyer: bool, fee: float = 0.0):
        value = trade.quantity * trade.price
        self._realized_pnl += self._realized_change(trade.symbol, trade.quantity, trade.price, is_buyer)
        if is_buyer:
            self._cash -= value + fee
        else:
            self._cash += value - fee
        self._update_position(trade.symbol, trade.quantity, trade.price, is_buyer)
        self.trade_history.append(trade)

    def _update_position(self, symbol: str, quantity: float, price: float, is_buyer: bool):
        pos = self.positions.get(symbol)
        if pos is None:
            self.positions[symbol] = _Position(quantity if is_buyer else -quantity, price)
            return
        old_qty = pos.quantity
        new_qty = old_qty + quantity if is_buyer else old_qty - quantity
        if new_qty == 0.0:
            del self.positions[symbol]
        elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            if is_buyer:
                pos.total_cost += quantity * price
                pos.quantity = new_qty
                pos.average_cost = pos.total_cost / abs(new_qty)
            else:
                pos.quantity = new_qty
                pos.total_cost = pos.average_cost * abs(new_qty)
        else:
            pos.quantity = new_qty
            pos.average_cost = price
            pos.total_cost = new_qty * price

    def _realized_change(self, symbol: str, quantity: float, price: float, is_buyer: bool) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        if pos.quantity > 0 and not is_buyer:
            return min(abs(pos.quantity), quantity) * (price - pos.average_cost)
        if pos.quantity < 0 and is_buyer:
            return min(abs(pos.quantity), quantity) * (pos.average_cost - price)
        return 0.0

    def mark_to_market(self, mark_prices: Dict[str, float]):
        self._unrealized_pnl = self.get_unrealized_pnl(mark_prices)

    def reset(self, initial_cash: float = 100000.0):
        self._cash = initial_cash
        self.positions.clear()
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self.trade_history.clear()


class RLTraderStats:
    def __init__(self):
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.episode_reward = 0.0
        self.cumulative_reward = 0.0
        self.episode_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.reward_history: List[float] = []
        self.pnl_history: List[float] = []


class RLTrader(Trader):
    def __init__(self, trader_id: str, strategy_type: str, initial_cash: float = 100000.0):
        super().__init__(trader_id, initial_cash)
        self.strategy_type = strategy_type
        self.stats = RLTraderStats()
        self.stats.cash = initial_cash
        self._peak_pnl = 0.0

    def get_strategy_type(self) -> str:
        return self.strategy_type

    def get_agent_id(self) -> str:
        return self.trader_id

    def get_stats(self) -> RLTraderStats:
        return self.stats

    def get_episode_reward(self) -> float:
        return self.stats.episode_reward

    def add_reward(self, reward: float):
        self.stats.episode_reward += reward

    def reset_episode(self):
        s = self.stats
        s.cumulative_reward += s.episode_reward
        s.reward_history.append(s.episode_reward)
        s.episode_reward = 0.0
        s.episode_count += 1
        self._update_metrics()

    def on_trade(self, trade: Trade, is_buyer: bool, fee: float = 0.0):
        super().on_trade(trade, is_buyer, fee)
        s = self.stats
        s.total_trades += 1
        s.cash = self.get_cash()
        s.realized_pnl = self.get_realized_pnl()
        if (-1 if is_buyer else 1) * trade.quantity * trade.price > 0:
            s.winning_trades += 1
        self._update_metrics()

    def mark_to_market(self, mark_prices: Dict[str, float]):
        super().mark_to_market(mark_prices)
        s = self.stats
        s.unrealized_pnl = self.get_unrealized_pnl(mark_prices)
        s.total_pnl = s.realized_pnl + s.unrealized_pnl
        s.pnl_history.append(s.total_pnl)
        if len(s.pnl_history) > 1000:
            s.pnl_history.pop(0)
        self._update_drawdown()
        s.sharpe_ratio = self._sharpe()

    def reset(self, initial_cash: float = 100000.0):
        super().reset(initial_cash)
        self.stats = RLTraderStats()
        self.stats.cash = initial_cash
        self._peak_pnl = 0.0

    def _update_metrics(self):
        s = self.stats
        if s.total_trades > 0:
            s.win_rate = s.winning_trades / s.total_trades
        self._update_drawdown()
        s.sharpe_ratio = self._sharpe()

    def _update_drawdown(self):
        s = self.stats
        self._peak_pnl = max(self._peak_pnl, s.total_pnl)
        s.current_drawdown = (self._peak_pnl - s.total_pnl) / self._peak_pnl if self._peak_pnl > 0 else 0.0
        s.max_drawdown = max(s.max_drawdown, s.current_drawdown)

    def _sharpe(self) -> float:
        h = self.stats.pnl_history
        if len(h) < 2:
            return 0.0
        returns = [h[i] - h[i - 1] for i in range(1, len(h))]
        mean = sum(returns) / len(returns)
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
        return mean / std if std > 0 else 0.0


# ============================================================================
# Price models and market events
# ============================================================================

class EventType(Enum):
    PRICE_SHOCK = 0
    VOLATILITY_SPIKE = 1
    LIQUIDITY_CRISIS = 2
    NEWS_EVENT = 3
    MARKET_CRASH = 4
    FLASH_CRASH = 5
    PUMP_AND_DUMP = 6
    EARNINGS_ANNOUNCEMENT = 7
    FED_ANNOUNCEMENT = 8
    TECHNICAL_BREAKOUT = 9
    CORRELATION_BREAKDOWN = 10
    MICROSTRUCTURE_NOISE = 11


class MarketEvent:
    __slots__ = ("type", "symbol", "magnitude", "duration", "description", "is_active", "sim_end_seconds")

    def __init__(self, type_, symbol, magnitude, duration, description):
        self.type = type_
        self.symbol = symbol
        self.magnitude = magnitude
        self.duration = duration
        self.description = description
        self.is_active = False
        self.sim_end_seconds = 0.0


class JumpDiffusionModel:
    def __init__(self, mu=0.0, sigma=0.2, lam=0.1):
        self.mu, self.sigma, self.lam = mu, sigma, lam
        self.jump_mu, self.jump_sigma = 0.0, 0.1
        self.rng = random.Random()

    def seed(self, seed: int):
        self.rng.seed(seed)

    def generate_price_change(self, price: float, dt: float) -> float:
        drift = self.mu * dt
        diffusion = self.sigma * math.sqrt(dt) * self.rng.gauss(0.0, 1.0)
        jump = 0.0
        if self.rng.random() < self.lam * dt:
            jump = self.jump_mu + self.jump_sigma * self.rng.gauss(0.0, 1.0)
        # Apply the log-return multiplicatively so the price can never go negative
        return price * (math.exp(drift + diffusion + jump) - 1.0)

    def update_parameters(self, event: MarketEvent):
        if event.type == EventType.PRICE_SHOCK:
            self.lam *= 1.0 + event.magnitude * 3.0
            self.jump_sigma *= 1.0 + event.magnitude
        elif event.type == EventType.FLASH_CRASH:
            self.lam *= 10.0
            self.jump_mu = -0.1
            self.jump_sigma *= 2.0
        elif event.type == EventType.VOLATILITY_SPIKE:
            self.sigma *= 1.0 + event.magnitude * 2.0
            self.lam *= 1.0 + event.magnitude

    def reset(self):
        self.mu, self.sigma, self.lam = 0.0, 0.2, 0.1
        self.jump_mu, self.jump_sigma = 0.0, 0.1


class MicrostructureNoise:
    def __init__(self, amplitude=0.001, mean_reversion=0.1):
        self.amplitude = amplitude
        self.mean_reversion = mean_reversion
        self.current = 0.0
        self.rng = random.Random()

    def seed(self, seed: int):
        self.rng.seed(seed)
        self.current = 0.0

    def generate_noise_increment(self, dt: float) -> float:
        previous = self.current
        self.current -= self.mean_reversion * self.current * dt
        self.current += self.amplitude * math.sqrt(dt) * self.rng.gauss(0.0, 1.0)
        return self.current - previous


_NEWS_TYPES = ["positive earnings guidance", "negative earnings guidance", "merger announcement",
               "regulatory approval", "product launch", "management change",
               "analyst upgrade", "analyst downgrade"]

_RANDOM_EVENT_TYPES = [EventType.PRICE_SHOCK, EventType.VOLATILITY_SPIKE, EventType.NEWS_EVENT,
                       EventType.EARNINGS_ANNOUNCEMENT, EventType.TECHNICAL_BREAKOUT,
                       EventType.LIQUIDITY_CRISIS, EventType.MARKET_CRASH,
                       EventType.FLASH_CRASH, EventType.FED_ANNOUNCEMENT]


class MarketEventGenerator:
    def __init__(self, symbols: List[str]):
        self.symbols = list(symbols)
        self.probability = 0.01
        self.verbose = True
        self.sim_seconds = 0.0
        self.active_events: List[MarketEvent] = []
        self.event_history: List[MarketEvent] = []
        self.rng = random.Random()
        self.models = {s: JumpDiffusionModel(0.0, 0.2, 0.05) for s in self.symbols}

    def seed(self, seed: int):
        self.rng.seed(seed)
        for i, symbol in enumerate(self.symbols):
            self.models[symbol].seed(seed + i + 1)

    def reset(self):
        self.sim_seconds = 0.0
        self.active_events.clear()
        for model in self.models.values():
            model.reset()

    def update(self, dt: float):
        self.sim_seconds += dt * TRADING_SECONDS_PER_YEAR
        self._clear_expired()
        if self.rng.random() < self.probability:
            event = self._make_event(self.rng.choice(_RANDOM_EVENT_TYPES), self.rng.choice(self.symbols))
            event.is_active = True
            event.sim_end_seconds = self.sim_seconds + event.duration
            self.active_events.append(event)
            self.event_history.append(event)
            for model in self.models.values():
                model.update_parameters(event)
            if self.verbose:
                print(f"Market Event: {event.description} (Magnitude: {event.magnitude}, "
                      f"Duration: {event.duration}s)")

    def _clear_expired(self):
        before = len(self.active_events)
        self.active_events = [e for e in self.active_events if self.sim_seconds <= e.sim_end_seconds]
        # An event's effect ends with it: rebuild model parameters from the events still active
        if len(self.active_events) != before:
            for model in self.models.values():
                model.reset()
                for event in self.active_events:
                    model.update_parameters(event)

    def _magnitude(self) -> float:
        return self.rng.random() ** 2.0  # small events more common

    def _duration(self) -> float:
        return 30.0 + self.rng.random() * 270.0

    def _make_event(self, kind: EventType, symbol: str) -> MarketEvent:
        mag, dur = self._magnitude(), self._duration()
        if kind == EventType.PRICE_SHOCK:
            return MarketEvent(kind, symbol, mag, dur, f"Price shock for {symbol} - sudden large movement")
        if kind == EventType.VOLATILITY_SPIKE:
            return MarketEvent(kind, symbol, mag, dur * 2.0, f"Volatility spike for {symbol} - increased price swings")
        if kind == EventType.LIQUIDITY_CRISIS:
            return MarketEvent(kind, symbol, mag, dur * 3.0, f"Liquidity crisis for {symbol} - reduced market depth")
        if kind == EventType.MARKET_CRASH:
            return MarketEvent(kind, "", mag * 2.0, dur * 5.0, "Market crash - broad market decline affecting all symbols")
        if kind == EventType.FLASH_CRASH:
            return MarketEvent(kind, "", mag * 3.0, 30.0, "Flash crash - rapid market decline")
        if kind == EventType.EARNINGS_ANNOUNCEMENT:
            return MarketEvent(kind, symbol, mag * 1.5, dur * 2.0, f"Earnings announcement for {symbol} - quarterly results")
        if kind == EventType.FED_ANNOUNCEMENT:
            return MarketEvent(kind, "", mag * 1.5, dur * 3.0, "Federal Reserve announcement - monetary policy changes")
        if kind == EventType.TECHNICAL_BREAKOUT:
            return MarketEvent(kind, symbol, mag, dur * 1.5, f"Technical breakout for {symbol} - price breaks key levels")
        news = _NEWS_TYPES[int(self.rng.random() * len(_NEWS_TYPES))]
        return MarketEvent(EventType.NEWS_EVENT, symbol, mag, dur, f"News event for {symbol}: {news}")


# ============================================================================
# Market simulator
# ============================================================================

class MarketSimulator:
    def __init__(self, symbols: List[str]):
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols")
        self.engines: Dict[str, MatchingEngine] = {}
        for symbol in symbols:
            self._init_engine(symbol)
        self.traders: Dict[str, Trader] = {}
        self.rl_traders: Dict[str, RLTrader] = {}
        self.order_to_trader: Dict[int, str] = {}
        self.all_trades: List[Trade] = []
        self.fair_prices: Dict[str, float] = {}
        self._next_order_id = 1
        self.events_enabled = False
        self.event_generator = MarketEventGenerator(symbols)
        self.noise = {s: MicrostructureNoise(0.001, 0.1) for s in symbols}

    def _init_engine(self, symbol: str):
        engine = MatchingEngine(symbol)
        engine.trade_callback = self._on_trade
        self.engines[symbol] = engine

    # Orders -------------------------------------------------------------

    def next_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def process_order(self, order: Order) -> List[Trade]:
        if not order.is_valid() or order.symbol not in self.engines:
            raise ValueError("Invalid order for market simulator")
        if order.trader_id:
            self.order_to_trader[order.id] = order.trader_id
        return self.engines[order.symbol].process_order(order)

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        engine = self.engines.get(symbol)
        return engine.cancel_order(order_id) if engine else False

    def _on_trade(self, trade: Trade):
        self.all_trades.append(trade)
        buyer = self.order_to_trader.get(trade.buy_order_id)
        seller = self.order_to_trader.get(trade.sell_order_id)
        for trader_map in (self.traders, self.rl_traders):
            if buyer in trader_map:
                trader_map[buyer].on_trade(trade, True, 0.0)
            if seller in trader_map:
                trader_map[seller].on_trade(trade, False, 0.0)

    # Market data --------------------------------------------------------

    def get_best_bid(self, symbol: str) -> float:
        return self.engines[symbol].book.best_bid() if symbol in self.engines else 0.0

    def get_best_ask(self, symbol: str) -> float:
        return self.engines[symbol].book.best_ask() if symbol in self.engines else 0.0

    def get_mid_price(self, symbol: str) -> float:
        return self.engines[symbol].book.mid_price() if symbol in self.engines else 0.0

    def get_spread(self, symbol: str) -> float:
        return self.engines[symbol].book.spread() if symbol in self.engines else 0.0

    def get_snapshot(self, symbol: str) -> OrderBookSnapshot:
        return self.engines[symbol].book.snapshot()

    def get_bid_depth(self, symbol: str) -> float:
        return self.engines[symbol].book.bid_depth()

    def get_ask_depth(self, symbol: str) -> float:
        return self.engines[symbol].book.ask_depth()

    def get_symbols(self) -> List[str]:
        return list(self.engines)

    def get_total_order_count(self) -> int:
        return sum(e.book.order_count() for e in self.engines.values())

    def get_total_trade_count(self) -> int:
        return len(self.all_trades)

    # Traders ------------------------------------------------------------

    def register_trader(self, trader_id: str, initial_cash: float = 0.0):
        if not trader_id:
            raise ValueError("Trader ID cannot be empty")
        if trader_id in self.traders:
            raise ValueError(f"Trader already exists: {trader_id}")
        self.traders[trader_id] = Trader(trader_id, initial_cash)

    def has_trader(self, trader_id: str) -> bool:
        return trader_id in self.traders

    def get_trader(self, trader_id: str) -> Trader:
        return self.traders[trader_id]

    def add_rl_trader(self, rl_trader: RLTrader):
        agent_id = rl_trader.get_agent_id()
        if not agent_id:
            raise ValueError("RL trader agent ID cannot be empty")
        if agent_id in self.rl_traders:
            raise ValueError(f"RL trader already exists: {agent_id}")
        self.rl_traders[agent_id] = rl_trader

    # Prices and events --------------------------------------------------

    def enable_market_events(self, enable: bool = True):
        self.events_enabled = enable
        print("Market events enabled - realistic price movements and random events active"
              if enable else "Market events disabled")

    def set_event_probability(self, probability: float):
        self.event_generator.probability = probability

    def set_event_logging(self, enable: bool):
        self.event_generator.verbose = enable

    def seed(self, seed: int):
        self.event_generator.seed(seed)
        for offset, symbol in enumerate(sorted(self.engines), start=1000):
            self.noise[symbol].seed(seed + offset)

    def get_fair_price(self, symbol: str) -> float:
        return self.fair_prices.get(symbol, 0.0)

    def set_fair_price(self, symbol: str, price: float):
        self.fair_prices[symbol] = price

    def update_market_events(self, dt: float):
        if not self.events_enabled:
            return
        self.event_generator.update(dt)
        # Advance each symbol's fair price; market makers re-quote around it
        for symbol in self.engines:
            price = self.get_fair_price(symbol)
            if price <= 0.0:
                price = self.get_mid_price(symbol)
            if not price > 0.0:
                price = 100.0
            change = self.event_generator.models[symbol].generate_price_change(price, dt)
            change += self.noise[symbol].generate_noise_increment(dt) * price
            self.fair_prices[symbol] = max(0.01, price + change)

    def reset(self):
        self.all_trades.clear()
        for symbol in list(self.engines):
            self._init_engine(symbol)
        self.fair_prices.clear()
        self.event_generator.reset()
        self.order_to_trader.clear()


# ============================================================================
# Market maker (synchronous; the C++ background-thread mode isn't ported)
# ============================================================================

class MarketMakerConfig:
    def __init__(self):
        self.trader_id = "market_maker"
        self.symbols: List[str] = []
        self.base_price = 100.0
        self.spread_pct = 0.001
        self.order_size = 10.0
        self.max_orders_per_side = 3
        self.update_interval_ms = 100
        self.adaptive_spread = False
        self.min_spread_pct = 0.0005
        self.max_spread_pct = 0.005
        self.volatility_window = 20


class MarketMaker:
    def __init__(self, simulator: MarketSimulator, config: MarketMakerConfig):
        self.sim = simulator
        self.config = config
        self.sim.register_trader(config.trader_id, 1000000.0)
        self.active_orders: Dict[str, List[Order]] = {s: [] for s in config.symbols}
        self.price_history: Dict[str, List[float]] = {s: [] for s in config.symbols}
        print("Market Maker initialized for symbols: " + " ".join(config.symbols) + " ")

    def step(self):
        for symbol in self.config.symbols:
            self._update_symbol(symbol)

    def reset(self):
        for orders in self.active_orders.values():
            orders.clear()
        for history in self.price_history.values():
            history.clear()

    def _update_symbol(self, symbol: str):
        bid, ask = self.sim.get_best_bid(symbol), self.sim.get_best_ask(symbol)
        fair = self.sim.get_fair_price(symbol)
        if fair > 0:
            mid = fair
        elif bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        else:
            mid = self.config.base_price

        history = self.price_history[symbol]
        history.append(mid)
        if len(history) > self.config.volatility_window:
            history.pop(0)

        spread_pct = self._adaptive_spread(symbol) if self.config.adaptive_spread else self.config.spread_pct
        half = mid * spread_pct / 2.0

        for order in self.active_orders[symbol]:
            if order.is_active():
                self.sim.cancel_order(symbol, order.id)
        self.active_orders[symbol].clear()

        for i in range(self.config.max_orders_per_side):
            self._place(symbol, Side.BUY, mid - half - i * 0.01)
        for i in range(self.config.max_orders_per_side):
            self._place(symbol, Side.SELL, mid + half + i * 0.01)

    def _place(self, symbol: str, side: Side, price: float):
        if not (price > 0.0 and self.config.order_size > 0.0):
            return  # e.g. a deep bid level below zero on a very low-priced symbol
        order = Order()
        order.id = self.sim.next_order_id()
        order.side = side
        order.type = OrderType.LIMIT
        order.price = price
        order.quantity = self.config.order_size
        order.symbol = symbol
        order.trader_id = self.config.trader_id
        order.strategy_id = "market_maker"
        self.sim.process_order(order)
        self.active_orders[symbol].append(order)

    def _adaptive_spread(self, symbol: str) -> float:
        prices = self.price_history[symbol]
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] > 0]
        if not returns:
            return self.config.min_spread_pct
        mean = sum(returns) / len(returns)
        vol = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
        factor = min(vol * 100.0, 1.0)
        return self.config.min_spread_pct + (self.config.max_spread_pct - self.config.min_spread_pct) * factor
