"""
DeepQuote RL Training Script

This script demonstrates how to train reinforcement learning agents
on the DeepQuote trading environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json
import time
from datetime import datetime

from environment import DeepQuoteEnv
from agents import create_agent, StableBaselinesAgent
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import wandb

class TrainingCallback:
    """Custom callback for tracking training progress"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def on_step(self, locals: Dict, globals: Dict) -> bool:
        """Called after each step"""
        self.current_episode_reward += locals['rewards'][0]
        self.current_episode_length += 1
        
        if locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Log to wandb
            if len(self.episode_rewards) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                wandb.log({
                    'episode_reward': avg_reward,
                    'episode_length': avg_length,
                    'episode': len(self.episode_rewards)
                })
        
        return True

def train_agent(agent_type: str = "PPO",
                symbols: List[str] = ["AAPL", "GOOGL"],
                initial_cash: float = 100000.0,
                max_steps: int = 1000,
                total_timesteps: int = 100000,
                learning_rate: float = 3e-4,
                use_wandb: bool = True,
                save_path: str = "models") -> Dict[str, Any]:
    """
    Train a reinforcement learning agent on the DeepQuote environment
    
    Args:
        agent_type: Type of agent to train ("PPO", "SAC", "TD3", "MarketMaking", "MeanReversion")
        symbols: List of trading symbols
        initial_cash: Initial cash for the agent
        max_steps: Maximum steps per episode
        total_timesteps: Total timesteps for training
        learning_rate: Learning rate for the agent
        use_wandb: Whether to use Weights & Biases for logging
        save_path: Path to save trained models
    
    Returns:
        Dictionary containing training results
    """
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="deepquote-rl",
            name=f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "agent_type": agent_type,
                "symbols": symbols,
                "initial_cash": initial_cash,
                "max_steps": max_steps,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate
            }
        )
    
    # Create environment
    env = DeepQuoteEnv(
        symbols=symbols,
        initial_cash=initial_cash,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    print(f"Training {agent_type} agent on {symbols}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create agent
    if agent_type in ["PPO", "SAC", "TD3", "A2C"]:
        # Use Stable Baselines3
        agent = StableBaselinesAgent(
            env=env,
            agent_type=agent_type,
            learning_rate=learning_rate,
            tensorboard_log=f"{save_path}/tensorboard_logs"
        )
        
        # Create callbacks
        eval_env = DeepQuoteEnv(symbols=symbols, initial_cash=initial_cash)
        eval_env = Monitor(eval_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}/best_model",
            log_path=f"{save_path}/eval_logs",
            eval_freq=max(1000 // max_steps, 1),
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000 // max_steps, 1),
            save_path=f"{save_path}/checkpoints",
            name_prefix=f"{agent_type}_model"
        )
        
        # Train the agent
        start_time = time.time()
        agent.train(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = f"{save_path}/{agent_type}_final"
        agent.save(final_model_path)
        
    else:
        # Use custom agents (MarketMaking, MeanReversion)
        agent = create_agent(agent_type, env)
        
        # For custom agents, we'll just run some episodes to test
        print(f"Testing {agent_type} agent...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(10):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                action = agent.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode % 5 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        training_time = 0  # No training for rule-based agents
    
    # Evaluate the trained agent
    print("Evaluating agent...")
    eval_results = evaluate_agent(agent, env, n_episodes=10)
    
    # Save results
    results = {
        "agent_type": agent_type,
        "symbols": symbols,
        "initial_cash": initial_cash,
        "training_time": training_time,
        "total_timesteps": total_timesteps,
        "evaluation_results": eval_results,
        "model_path": f"{save_path}/{agent_type}_final" if agent_type in ["PPO", "SAC", "TD3", "A2C"] else None
    }
    
    with open(f"{save_path}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Log final results to wandb
    if use_wandb:
        wandb.log({
            "final_avg_reward": eval_results["avg_reward"],
            "final_avg_pnl": eval_results["avg_pnl"],
            "training_time": training_time
        })
        wandb.finish()
    
    print(f"Training completed!")
    print(f"Average reward: {eval_results['avg_reward']:.2f}")
    print(f"Average P&L: ${eval_results['avg_pnl']:.2f}")
    print(f"Training time: {training_time:.2f}s")
    
    return results

def evaluate_agent(agent, env: DeepQuoteEnv, n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate a trained agent"""
    
    episode_rewards = []
    episode_pnls = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(env.max_steps):
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_pnls.append(info['total_pnl'])
        episode_lengths.append(episode_length)
    
    return {
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_pnl": np.mean(episode_pnls),
        "std_pnl": np.std(episode_pnls),
        "avg_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_pnls": episode_pnls
    }

def compare_agents(agent_types: List[str] = ["PPO", "SAC", "MarketMaking", "MeanReversion"],
                  symbols: List[str] = ["AAPL", "GOOGL"],
                  initial_cash: float = 100000.0,
                  total_timesteps: int = 50000,
                  save_path: str = "comparison_results") -> Dict[str, Any]:
    """
    Compare multiple agents on the same environment
    
    Args:
        agent_types: List of agent types to compare
        symbols: List of trading symbols
        initial_cash: Initial cash for agents
        total_timesteps: Total timesteps for training (for RL agents)
        save_path: Path to save comparison results
    
    Returns:
        Dictionary containing comparison results
    """
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="deepquote-agent-comparison",
        name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "agent_types": agent_types,
            "symbols": symbols,
            "initial_cash": initial_cash,
            "total_timesteps": total_timesteps
        }
    )
    
    results = {}
    
    # Define agent configurations
    agent_configs = {
        "MarketMaking": {"spread_target": 0.001, "order_size": 10.0},
        "MeanReversion": {"lookback_period": 20, "entry_threshold": 2.0, "exit_threshold": 0.5},
        "Momentum": {"short_window": 10, "long_window": 30, "momentum_threshold": 0.001, "position_size": 0.3},
        "Arbitrage": {"z_score_threshold": 2.0, "position_size": 0.2},
        "GridTrading": {"grid_levels": 5, "grid_spacing": 0.01, "base_position_size": 10.0},
        "VolatilityBreakout": {"volatility_window": 20, "breakout_threshold": 2.0, "position_size": 0.3},
        "PairsTrading": {"entry_threshold": 2.0, "exit_threshold": 0.5, "position_size": 0.2}
    }
    
    for agent_type in agent_types:
        print(f"\nTraining/Testing {agent_type} agent...")
        
        # Determine symbols for this agent
        if agent_type in ["Arbitrage", "PairsTrading"]:
            agent_symbols = symbols if len(symbols) >= 2 else ["AAPL", "GOOGL"]
        else:
            agent_symbols = symbols[:1] if symbols else ["AAPL"]
        
        # Create environment
        env = DeepQuoteEnv(
            symbols=agent_symbols,
            initial_cash=initial_cash,
            max_position_size=1000.0,
            transaction_cost=0.001
        )
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        # Get agent configuration
        config = agent_configs.get(agent_type, {})
        
        # Train/test agent
        if agent_type in ["PPO", "SAC", "TD3", "A2C"]:
            # RL agents - train them
            result = train_agent(
                agent_type=agent_type,
                symbols=agent_symbols,
                initial_cash=initial_cash,
                total_timesteps=total_timesteps,
                use_wandb=False  # Already initialized
            )
        else:
            # Rule-based agents - test them
            agent = create_agent(agent_type, env, **config)
            
            # Run multiple episodes for evaluation
            episode_rewards = []
            episode_pnls = []
            episode_lengths = []
            
            for episode in range(10):
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(1000):  # Max 1000 steps per episode
                    action = agent.get_action(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if done or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_pnls.append(info['total_pnl'])
                episode_lengths.append(episode_length)
            
            # Calculate evaluation metrics
            eval_results = {
                'episode_reward_mean': np.mean(episode_rewards),
                'episode_reward_std': np.std(episode_rewards),
                'total_pnl_mean': np.mean(episode_pnls),
                'total_pnl_std': np.std(episode_pnls),
                'episode_length_mean': np.mean(episode_lengths),
                'win_rate': np.mean([1 if pnl > 0 else 0 for pnl in episode_pnls]),
                'max_drawdown': min(episode_pnls) if episode_pnls else 0,
                'sharpe_ratio': np.mean(episode_pnls) / np.std(episode_pnls) if np.std(episode_pnls) > 0 else 0
            }
            
            result = {
                "agent_type": agent_type,
                "symbols": agent_symbols,
                "initial_cash": initial_cash,
                "training_time": 0,  # No training for rule-based agents
                "total_timesteps": len(episode_rewards) * np.mean(episode_lengths),
                "evaluation_results": eval_results,
                "model_path": None
            }
        
        results[agent_type] = result
        
        # Log to wandb
        wandb.log({
            f"{agent_type}_episode_reward": result["evaluation_results"]["episode_reward_mean"],
            f"{agent_type}_total_pnl": result["evaluation_results"]["total_pnl_mean"],
            f"{agent_type}_win_rate": result["evaluation_results"]["win_rate"],
            f"{agent_type}_sharpe_ratio": result["evaluation_results"]["sharpe_ratio"]
        })
        
        print(f"  Episode reward: {result['evaluation_results']['episode_reward_mean']:.2f} ± {result['evaluation_results']['episode_reward_std']:.2f}")
        print(f"  Total P&L: ${result['evaluation_results']['total_pnl_mean']:,.2f} ± ${result['evaluation_results']['total_pnl_std']:,.2f}")
        print(f"  Win rate: {result['evaluation_results']['win_rate']:.2%}")
        print(f"  Sharpe ratio: {result['evaluation_results']['sharpe_ratio']:.3f}")
    
    # Save results
    with open(f"{save_path}/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create comparison plots
    create_comparison_plots(results, save_path)
    
    # Close wandb
    wandb.finish()
    
    return results

def create_comparison_plots(results: Dict[str, Any], save_path: str):
    """Create comparison plots for different agents"""
    
    # Extract data for plotting
    agent_names = []
    avg_rewards = []
    avg_pnls = []
    training_times = []
    
    for agent_type, result in results.items():
        if "error" not in result:
            agent_names.append(agent_type)
            avg_rewards.append(result["evaluation_results"]["episode_reward_mean"])
            avg_pnls.append(result["evaluation_results"]["total_pnl_mean"])
            training_times.append(result["training_time"])
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Average rewards
    axes[0].bar(agent_names, avg_rewards)
    axes[0].set_title("Average Episode Reward")
    axes[0].set_ylabel("Reward")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Average P&L
    axes[1].bar(agent_names, avg_pnls)
    axes[1].set_title("Average P&L")
    axes[1].set_ylabel("P&L ($)")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Training time
    axes[2].bar(agent_names, training_times)
    axes[2].set_title("Training Time")
    axes[2].set_ylabel("Time (s)")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/agent_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run training and comparison"""
    
    print("DeepQuote RL Training and Comparison")
    print("=" * 50)
    
    # Example 1: Train a single PPO agent
    print("\n1. Training a PPO agent...")
    try:
        results = train_agent(
            agent_type="PPO",
            symbols=["AAPL", "GOOGL"],
            initial_cash=100000.0,
            total_timesteps=10000,  # Reduced for demo
            learning_rate=3e-4,
            use_wandb=False  # Disable for demo
        )
        print(f"Training completed! Final P&L: ${results['evaluation_results']['total_pnl_mean']:,.2f}")
    except Exception as e:
        print(f"Training failed: {e}")
    
    # Example 2: Compare multiple agents
    print("\n2. Comparing multiple agents...")
    try:
        comparison_results = compare_agents(
            agent_types=[
                "MarketMaking", 
                "MeanReversion", 
                "Momentum", 
                "Arbitrage", 
                "GridTrading", 
                "VolatilityBreakout", 
                "PairsTrading"
            ],
            symbols=["AAPL", "GOOGL"],
            initial_cash=100000.0,
            total_timesteps=5000,  # Reduced for demo
            save_path="demo_comparison"
        )
        
        print("\nComparison Results Summary:")
        print("-" * 80)
        print(f"{'Agent':<20} {'Reward':<10} {'P&L':<15} {'Win Rate':<10} {'Sharpe':<8}")
        print("-" * 80)
        
        for agent_type, result in comparison_results.items():
            if "error" not in result:
                eval_results = result["evaluation_results"]
                print(f"{agent_type:<20} {eval_results['episode_reward_mean']:<10.2f} "
                      f"${eval_results['total_pnl_mean']:<14,.2f} "
                      f"{eval_results['win_rate']:<10.1%} "
                      f"{eval_results['sharpe_ratio']:<8.3f}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
    
    # Example 3: Train RL agents
    print("\n3. Training RL agents...")
    try:
        rl_results = compare_agents(
            agent_types=["PPO", "SAC"],
            symbols=["AAPL"],
            initial_cash=100000.0,
            total_timesteps=10000,  # Reduced for demo
            save_path="rl_comparison"
        )
        
        print("\nRL Agents Results Summary:")
        print("-" * 80)
        print(f"{'Agent':<20} {'Reward':<10} {'P&L':<15} {'Win Rate':<10} {'Sharpe':<8}")
        print("-" * 80)
        
        for agent_type, result in rl_results.items():
            if "error" not in result:
                eval_results = result["evaluation_results"]
                print(f"{agent_type:<20} {eval_results['episode_reward_mean']:<10.2f} "
                      f"${eval_results['total_pnl_mean']:<14,.2f} "
                      f"{eval_results['win_rate']:<10.1%} "
                      f"{eval_results['sharpe_ratio']:<8.3f}")
        
    except Exception as e:
        print(f"RL training failed: {e}")
    
    print("\n" + "=" * 50)
    print("Training and comparison completed!")
    print("Check the generated directories for detailed results and plots.")

if __name__ == "__main__":
    main() 