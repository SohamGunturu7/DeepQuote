"""
DeepQuote RL Demo

A simple demo showing how to use the reinforcement learning system.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import DeepQuoteEnv
from agents import create_agent

def demo_basic_environment():
    """Demo the basic environment"""
    print("=" * 50)
    print("DeepQuote RL Environment Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL", "GOOGL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    print(f"Environment created with symbols: {env.symbols}")
    print(f"Initial cash: ${env.initial_cash:,.2f}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few episodes with random actions
    print("\nRunning episodes with random actions...")
    
    episode_rewards = []
    episode_pnls = []
    
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Initial cash: ${info['cash']:,.2f}")
        
        while step < 100:  # Run for 100 steps
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1
            
            if step % 20 == 0:
                print(f"    Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_pnls.append(info['total_pnl'])
        
        print(f"  Final cash: ${info['cash']:,.2f}")
        print(f"  Final P&L: ${info['total_pnl']:,.2f}")
        print(f"  Episode reward: {episode_reward:.2f}")
    
    print(f"\nAverage episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Average P&L: ${np.mean(episode_pnls):,.2f}")

def demo_market_making_agent():
    """Demo the market making agent"""
    print("\n" + "=" * 50)
    print("Market Making Agent Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create market making agent
    agent = create_agent("MarketMaking", env, spread_target=0.001, order_size=10.0)
    
    print("Running market making agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_mean_reversion_agent():
    """Demo the mean reversion agent"""
    print("\n" + "=" * 50)
    print("Mean Reversion Agent Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create mean reversion agent
    agent = create_agent("MeanReversion", env, lookback_period=20, entry_threshold=2.0)
    
    print("Running mean reversion agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_price_movement():
    """Demo price movement and technical indicators"""
    print("\n" + "=" * 50)
    print("Price Movement and Technical Indicators Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    obs, info = env.reset()
    
    # Track price movements
    prices = []
    spreads = []
    steps = []
    
    for step in range(100):
        # Take no action, just observe
        action = np.array([5, 0, 0, 0.5])  # HOLD action
        obs, reward, done, truncated, info = env.step(action)
        
        # Extract price data from observation
        mid_price = obs[2]  # Mid price is at index 2
        spread = obs[3]     # Spread is at index 3
        
        prices.append(mid_price)
        spreads.append(spread)
        steps.append(step)
        
        if step % 20 == 0:
            print(f"Step {step}: Price=${mid_price:.2f}, Spread=${spread:.4f}")
    
    # Plot price movement
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(steps, prices, 'b-', linewidth=2)
    plt.title('AAPL Price Movement')
    plt.ylabel('Price ($)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(steps, spreads, 'r-', linewidth=2)
    plt.title('Bid-Ask Spread')
    plt.ylabel('Spread ($)')
    plt.xlabel('Step')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('price_movement_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Price movement plot saved as 'price_movement_demo.png'")

def demo_momentum_agent():
    """Demo the momentum trading agent"""
    print("\n" + "=" * 50)
    print("Momentum Trading Agent Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create momentum agent
    agent = create_agent("Momentum", env, short_window=10, long_window=30, momentum_threshold=0.001)
    
    print("Running momentum agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_arbitrage_agent():
    """Demo the arbitrage agent"""
    print("\n" + "=" * 50)
    print("Arbitrage Agent Demo")
    print("=" * 50)
    
    # Create environment with multiple symbols for arbitrage
    env = DeepQuoteEnv(
        symbols=["AAPL", "GOOGL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create arbitrage agent
    agent = create_agent("Arbitrage", env, z_score_threshold=2.0, position_size=0.2)
    
    print("Running arbitrage agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_grid_trading_agent():
    """Demo the grid trading agent"""
    print("\n" + "=" * 50)
    print("Grid Trading Agent Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create grid trading agent
    agent = create_agent("GridTrading", env, grid_levels=5, grid_spacing=0.01, base_position_size=10.0)
    
    print("Running grid trading agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_volatility_breakout_agent():
    """Demo the volatility breakout agent"""
    print("\n" + "=" * 50)
    print("Volatility Breakout Agent Demo")
    print("=" * 50)
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=["AAPL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create volatility breakout agent
    agent = create_agent("VolatilityBreakout", env, volatility_window=20, breakout_threshold=2.0)
    
    print("Running volatility breakout agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_pairs_trading_agent():
    """Demo the pairs trading agent"""
    print("\n" + "=" * 50)
    print("Pairs Trading Agent Demo")
    print("=" * 50)
    
    # Create environment with multiple symbols for pairs trading
    env = DeepQuoteEnv(
        symbols=["AAPL", "GOOGL"],
        initial_cash=100000.0,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Create pairs trading agent
    agent = create_agent("PairsTrading", env, entry_threshold=2.0, exit_threshold=0.5, position_size=0.2)
    
    print("Running pairs trading agent...")
    
    obs, info = env.reset()
    episode_reward = 0
    step = 0
    
    print(f"Initial cash: ${info['cash']:,.2f}")
    
    while step < 200:  # Run for 200 steps
        # Get action from agent
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Cash=${info['cash']:,.2f}, P&L=${info['total_pnl']:,.2f}")
            print(f"  Inventory: {info['inventory']}")
        
        if done or truncated:
            break
    
    print(f"Final cash: ${info['cash']:,.2f}")
    print(f"Final P&L: ${info['total_pnl']:,.2f}")
    print(f"Episode reward: {episode_reward:.2f}")

def demo_all_agents():
    """Demo all available agents"""
    print("\n" + "=" * 50)
    print("All Agents Comparison Demo")
    print("=" * 50)
    
    # List of all available agents
    agents = [
        ("MarketMaking", {"spread_target": 0.001, "order_size": 10.0}),
        ("MeanReversion", {"lookback_period": 20, "entry_threshold": 2.0}),
        ("Momentum", {"short_window": 10, "long_window": 30, "momentum_threshold": 0.001}),
        ("Arbitrage", {"z_score_threshold": 2.0, "position_size": 0.2}),
        ("GridTrading", {"grid_levels": 5, "grid_spacing": 0.01, "base_position_size": 10.0}),
        ("VolatilityBreakout", {"volatility_window": 20, "breakout_threshold": 2.0}),
        ("PairsTrading", {"entry_threshold": 2.0, "exit_threshold": 0.5, "position_size": 0.2})
    ]
    
    results = {}
    
    for agent_name, agent_params in agents:
        print(f"\nTesting {agent_name} agent...")
        
        # Create environment
        env = DeepQuoteEnv(
            symbols=["AAPL", "GOOGL"] if agent_name in ["Arbitrage", "PairsTrading"] else ["AAPL"],
            initial_cash=100000.0,
            max_position_size=1000.0,
            transaction_cost=0.001
        )
        
        # Create agent
        agent = create_agent(agent_name, env, **agent_params)
        
        # Run episode
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        while step < 100:  # Run for 100 steps
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1
            
            if done or truncated:
                break
        
        results[agent_name] = {
            'final_cash': info['cash'],
            'total_pnl': info['total_pnl'],
            'episode_reward': episode_reward
        }
        
        print(f"  Final cash: ${info['cash']:,.2f}")
        print(f"  Final P&L: ${info['total_pnl']:,.2f}")
        print(f"  Episode reward: {episode_reward:.2f}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary of All Agents")
    print("=" * 50)
    
    for agent_name, result in results.items():
        print(f"{agent_name:20} | Cash: ${result['final_cash']:>10,.2f} | P&L: ${result['total_pnl']:>10,.2f} | Reward: {result['episode_reward']:>8.2f}")

def main():
    """Run all demos"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demos
    demo_basic_environment()
    demo_market_making_agent()
    demo_mean_reversion_agent()
    demo_price_movement()
    demo_momentum_agent()
    demo_arbitrage_agent()
    demo_grid_trading_agent()
    demo_volatility_breakout_agent()
    demo_pairs_trading_agent()
    demo_all_agents()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python train.py")
    print("3. Experiment with different agents and parameters")

if __name__ == "__main__":
    main() 