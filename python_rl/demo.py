#!/usr/bin/env python3
"""
DeepQuote Demo

Runs each rule-based agent for one episode on the C++ market simulator and
plots the price path alongside each agent's equity curve.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents import MarketMakingAgent, MeanReversionAgent, MomentumAgent, VolatilityBreakoutAgent
from deepquote_env import DeepQuoteEnv

SYMBOL = "AAPL"
STEPS = 500
SEED = 7


# Run one agent for an episode and record prices, equity and trades
def run_agent(name, agent_cls, **agent_kwargs):
    env = DeepQuoteEnv(symbols=[SYMBOL], max_steps=STEPS)
    agent = agent_cls(env, **agent_kwargs)
    obs, info = env.reset(seed=SEED)

    prices, equity = [info["mid_prices"][SYMBOL]], [info["equity"]]
    total_reward = 0.0
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        prices.append(info["mid_prices"][SYMBOL])
        equity.append(info["equity"])
        done = terminated or truncated

    trades = env.trader.get_stats().total_trades
    print(f"{name:<18} trades={trades:<5} inventory={info['inventory'][SYMBOL]:>7.0f} "
          f"pnl=${info['total_pnl']:>10,.2f}  reward={total_reward:>8.3f}")
    return prices, equity


def main():
    print("DeepQuote Demo")
    print("=" * 50)
    print(f"{STEPS} steps of {SYMBOL}, same market seed for every agent\n")

    agents = [
        ("MarketMaking", MarketMakingAgent, {"spread_target": 0.002, "order_size": 20.0}),
        ("MeanReversion", MeanReversionAgent, {"entry_threshold": 1.5, "exit_threshold": 0.3}),
        ("Momentum", MomentumAgent, {"momentum_threshold": 0.0005, "position_size": 0.2}),
        ("VolatilityBreakout", VolatilityBreakoutAgent, {"breakout_threshold": 1.2, "position_size": 0.2}),
    ]

    results = {name: run_agent(name, cls, **kwargs) for name, cls, kwargs in agents}

    fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    prices = next(iter(results.values()))[0]
    ax_price.plot(prices, color="black")
    ax_price.set_title(f"{SYMBOL} mid price")
    ax_price.set_ylabel("Price ($)")
    for name, (_, equity) in results.items():
        ax_equity.plot(np.array(equity) - equity[0], label=name)
    ax_equity.axhline(0, color="gray", linewidth=0.8)
    ax_equity.set_title("Agent P&L")
    ax_equity.set_xlabel("Step")
    ax_equity.set_ylabel("P&L ($)")
    ax_equity.legend()
    plt.tight_layout()
    plt.savefig("price_movement_demo.png", dpi=150)
    print("\nSaved plot to price_movement_demo.png")


if __name__ == "__main__":
    main()
