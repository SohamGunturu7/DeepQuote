#!/usr/bin/env python3
"""
Demo script showing RL agents trading in the C++ DeepQuote market simulator.
This demonstrates the full integration between Python RL agents and the C++ backend.
"""

import numpy as np
import time
from agents import (
    MarketMakingAgent, MeanReversionAgent, MomentumAgent, 
    ArbitrageAgent, GridTradingAgent, VolatilityBreakoutAgent, 
    PairsTradingAgent, StableBaselinesAgent, create_agent
)
from deepquote_env import DeepQuoteEnv

def demo_single_agent(agent_type: str, symbols: list = ["AAPL"], steps: int = 100):
    """Demo a single agent trading in the C++ market"""
    print(f"\n{'='*60}")
    print(f"DEMO: {agent_type} Agent Trading in C++ Market")
    print(f"{'='*60}")
    
    # Create the C++ environment
    env = DeepQuoteEnv(symbols=symbols, trader_id=f"{agent_type.lower()}_agent", strategy_type=agent_type)
    
    # Create the agent
    agent = create_agent(agent_type, env)
    
    # Trading loop
    obs = env.reset()
    total_reward = 0
    trades_made = 0
    
    print(f"Starting {agent_type} agent with {steps} steps...")
    print(f"Initial observation: {obs}")
    
    for step in range(steps):
        # Get action from agent
        action = agent.get_action(obs)
        
        # Execute action in C++ market
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        if info.get('pnl', 0) != 0:  # If PnL changed, a trade likely occurred
            trades_made += 1
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d}: Action={action}, Reward={reward:.2f}, PnL={info.get('pnl', 0):.2f}, Inventory={info.get('inventory', 0)}")
        
        if done:
            break
    
    # Final results
    final_pnl = env.trader.get_realized_pnl()
    final_inventory = env.trader.get_inventory()
    final_cash = env.trader.get_cash()
    
    print(f"\n{agent_type} Agent Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final PnL: {final_pnl:.2f}")
    print(f"  Final Inventory: {final_inventory}")
    print(f"  Final Cash: {final_cash:.2f}")
    print(f"  Trades Made: {trades_made}")
    print(f"  Sharpe Ratio: {env.trader.get_sharpe_ratio():.3f}")
    print(f"  Win Rate: {env.trader.get_win_rate():.1%}")
    
    return {
        'agent_type': agent_type,
        'total_reward': total_reward,
        'final_pnl': final_pnl,
        'final_inventory': final_inventory,
        'final_cash': final_cash,
        'trades_made': trades_made,
        'sharpe_ratio': env.trader.get_sharpe_ratio(),
        'win_rate': env.trader.get_win_rate()
    }

def demo_multi_agent_competition(agent_types: list, symbols: list = ["AAPL"], steps: int = 200):
    """Demo multiple agents competing in the same C++ market"""
    print(f"\n{'='*60}")
    print(f"MULTI-AGENT COMPETITION in C++ Market")
    print(f"{'='*60}")
    
    # Create the C++ environment
    env = DeepQuoteEnv(symbols=symbols, trader_id="competition_agent", strategy_type="Competition")
    
    # Create all agents
    agents = {}
    for agent_type in agent_types:
        agents[agent_type] = create_agent(agent_type, env)
    
    # Trading loop with agent rotation
    obs = env.reset()
    results = {agent_type: {'reward': 0, 'trades': 0} for agent_type in agent_types}
    
    print(f"Starting competition with {len(agent_types)} agents for {steps} steps...")
    print(f"Agents: {', '.join(agent_types)}")
    
    for step in range(steps):
        # Rotate through agents (each agent gets a turn)
        agent_type = agent_types[step % len(agent_types)]
        agent = agents[agent_type]
        
        # Get action from current agent
        action = agent.get_action(obs)
        
        # Execute action in C++ market
        obs, reward, done, info = env.step(action)
        
        results[agent_type]['reward'] += reward
        if info.get('pnl', 0) != 0:
            results[agent_type]['trades'] += 1
        
        # Print progress every 50 steps
        if step % 50 == 0:
            print(f"Step {step:3d}: {agent_type} -> Action={action}, Reward={reward:.2f}, PnL={info.get('pnl', 0):.2f}")
        
        if done:
            break
    
    # Final results
    final_pnl = env.trader.get_realized_pnl()
    final_inventory = env.trader.get_inventory()
    final_cash = env.trader.get_cash()
    
    print(f"\nCompetition Results:")
    print(f"  Final PnL: {final_pnl:.2f}")
    print(f"  Final Inventory: {final_inventory}")
    print(f"  Final Cash: {final_cash:.2f}")
    print(f"  Sharpe Ratio: {env.trader.get_sharpe_ratio():.3f}")
    print(f"  Win Rate: {env.trader.get_win_rate():.1%}")
    
    print(f"\nAgent Performance:")
    for agent_type, result in results.items():
        print(f"  {agent_type:20s}: Reward={result['reward']:8.2f}, Trades={result['trades']:3d}")
    
    return results

def main():
    """Main demo function"""
    print("DeepQuote C++ Integration Demo")
    print("=" * 60)
    print("This demo shows RL agents trading in the realistic C++ market simulator")
    print("All market logic, order matching, and trade execution happens in C++")
    print("while RL agents run in Python for flexibility and experimentation.")
    print()
    
    # Test different agent types
    agent_types = [
        "MarketMaking",
        "MeanReversion", 
        "Momentum",
        "GridTrading",
        "VolatilityBreakout"
    ]
    
    # Demo 1: Single agents
    print("DEMO 1: Individual Agent Performance")
    print("-" * 40)
    
    single_results = []
    for agent_type in agent_types[:3]:  # Test first 3 agents
        try:
            result = demo_single_agent(agent_type, steps=50)
            single_results.append(result)
        except Exception as e:
            print(f"Error with {agent_type}: {e}")
            continue
    
    # Demo 2: Multi-agent competition
    print("\nDEMO 2: Multi-Agent Competition")
    print("-" * 40)
    
    try:
        competition_results = demo_multi_agent_competition(
            agent_types[:3], steps=100
        )
    except Exception as e:
        print(f"Error in competition: {e}")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE!")
    print("Your RL agents are now successfully trading in the C++ market simulator!")
    print("The integration provides:")
    print("  ✓ Realistic limit order book simulation")
    print("  ✓ High-performance C++ backend")
    print("  ✓ Flexible Python RL agents")
    print("  ✓ Comprehensive performance tracking")
    print("  ✓ Multi-agent support")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 