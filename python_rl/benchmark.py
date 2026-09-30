#!/usr/bin/env python3
"""
Speed comparison: C++ simulator core vs the pure-Python port (pysim.py).

Three measurements, each run on both backends with identical inputs:
  1. matching:  orders/sec through the matching engine alone
  2. market:    one market tick (price move + market maker re-quote), at several book depths
  3. env step:  full DeepQuoteEnv.step() with random actions, as an RL training loop sees it

Usage: python benchmark.py [--quick]
"""

import argparse
import time

import numpy as np

from deepquote_env import DeepQuoteEnv, load_backend

BACKENDS = ["cpp", "python"]


def best_time(fn, repeats: int) -> float:
    """Fastest of several runs, which filters out noise from other processes."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


# 1. Matching engine alone ---------------------------------------------------

def make_order_stream(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    stream = []
    for _ in range(n):
        is_market = rng.random() < 0.2
        side = "BUY" if rng.random() < 0.5 else "SELL"
        # Limit prices straddle 100 so roughly half of them cross and trade
        price = round(100.0 + rng.normal(0, 0.5), 2)
        qty = float(rng.integers(1, 50))
        stream.append((is_market, side, price, qty))
    return stream


def bench_matching(backend: str, stream, repeats: int) -> float:
    dq = load_backend(backend)

    def run():
        sim = dq.MarketSimulator(["AAPL"])
        for is_market, side, price, qty in stream:
            order = dq.Order()
            order.id = sim.next_order_id()
            order.side = dq.Side.BUY if side == "BUY" else dq.Side.SELL
            order.type = dq.OrderType.MARKET if is_market else dq.OrderType.LIMIT
            order.price = 0.0 if is_market else price
            order.quantity = qty
            order.symbol = "AAPL"
            order.trader_id = "t1" if side == "BUY" else "t2"
            sim.process_order(order)

    return len(stream) / best_time(run, repeats)


# 2. One market tick -------------------------------------------------------

def bench_market_tick(backend: str, levels: int, n_steps: int, repeats: int) -> float:
    env = DeepQuoteEnv(symbols=["AAPL"], backend=backend, mm_levels=levels)
    env.reset(seed=0)
    sim, mm = env.sim, env.market_maker

    def run():
        for _ in range(n_steps):
            sim.update_market_events(env.dt)
            mm.step()

    return n_steps / best_time(run, repeats)


# 3. Full environment step ---------------------------------------------------

def bench_env_step(backend: str, symbols, levels: int, n_steps: int, repeats: int) -> float:
    env = DeepQuoteEnv(symbols=symbols, backend=backend, mm_levels=levels, max_steps=10**9)
    rng = np.random.default_rng(0)
    actions = [env.action_space.sample() for _ in range(n_steps)]
    env.action_space.seed(0)

    def run():
        env.reset(seed=0)
        for a in actions:
            env.step(a)

    return n_steps / best_time(run, repeats)


def report(title: str, unit: str, results):
    print(f"\n{title}")
    print(f"  {'case':<28}{'C++':>14}{'Python':>14}{'C++ speedup':>14}")
    for case, cpp, py in results:
        print(f"  {case:<28}{cpp:>11,.0f} {unit}{py:>11,.0f} {unit}{cpp / py:>13.1f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Fewer iterations")
    args = parser.parse_args()
    scale, repeats = (0.2, 2) if args.quick else (1.0, 3)

    print("DeepQuote speed comparison: C++ core vs pure Python (higher is better)")

    stream = make_order_stream(int(50_000 * scale))
    report("1. Matching engine only", "/s", [
        (f"{len(stream):,} random orders",
         bench_matching("cpp", stream, repeats), bench_matching("python", stream, repeats)),
    ])

    n = int(5_000 * scale)
    report("2. Market tick (price move + market maker re-quote, 1 symbol)", "/s", [
        (f"{levels} levels per side", *(bench_market_tick(b, levels, n, repeats) for b in BACKENDS))
        for levels in (5, 20, 50)
    ])

    n = int(3_000 * scale)
    report("3. Full env.step() with random actions", "/s", [
        (f"{len(syms)} symbol(s), {levels} levels",
         *(bench_env_step(b, syms, levels, n, repeats) for b in BACKENDS))
        for syms, levels in ((["AAPL"], 5), (["AAPL"], 50), (["AAPL", "GOOGL"], 5))
    ])


if __name__ == "__main__":
    main()
