"""
DeepQuote RL Training Script
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

# Custom training callback
class TrainingCallback:
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def on_step(self, locals: Dict, globals: Dict) -> bool:
        self.current_episode_reward += locals['rewards'][0]
        self.current_episode_length += 1
        
        if locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if len(self.episode_rewards) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                wandb.log({
                    'episode_reward': avg_reward,
                    'episode_length': avg_length,
                    'episode': len(self.episode_rewards)
                })
        
        return True

# Main training function
def train_agent(agent_type: str = "PPO",
                symbols: List[str] = ["AAPL", "GOOGL"],
                initial_cash: float = 100000.0,
                max_steps: int = 1000,
                total_timesteps: int = 100000,
                learning_rate: float = 3e-4,
                use_wandb: bool = True,
                save_path: str = "models") -> Dict[str, Any]:
    
    os.makedirs(save_path, exist_ok=True)
    
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
    
    env = DeepQuoteEnv(
        symbols=symbols,
        initial_cash=initial_cash,
        max_position_size=1000.0,
        transaction_cost=0.001
    )
    
    env = Monitor(env)
    
    print(f"Training {agent_type} agent on {symbols}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    if agent_type in ["PPO", "SAC", "TD3", "A2C"]:
        agent = StableBaselinesAgent(
            env=env,
            agent_type=agent_type,
            learning_rate=learning_rate,
            tensorboard_log=f"{save_path}/tensorboard_logs"
        )
        
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
        
        start_time = time.time()
        agent.train(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        training_time = time.time() - start_time
        
        final_model_path = f"{save_path}/{agent_type}_final"
        agent.save(final_model_path)
        
    else:
        agent = create_agent(agent_type, env)
        
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
        
        training_time = 0
    
    print("Evaluating agent...")
    eval_results = evaluate_agent(agent, env, n_episodes=10)
    
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
    
    if use_wandb:
        wandb.finish()
    
    return results

# Agent evaluation function
def evaluate_agent(agent, env: DeepQuoteEnv, n_episodes: int = 10) -> Dict[str, float]:
    episode_rewards = []
    episode_lengths = []
    final_pnls = []
    max_drawdowns = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_pnls = []
        
        for step in range(1000):
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            episode_pnls.append(info['total_pnl'])
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_pnls.append(info['total_pnl'])
        
        if episode_pnls:
            peak = max(episode_pnls)
            final = episode_pnls[-1]
            drawdown = (peak - final) / peak if peak > 0 else 0
            max_drawdowns.append(drawdown)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_final_pnl": np.mean(final_pnls),
        "std_final_pnl": np.std(final_pnls),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "win_rate": np.mean([1 if pnl > 0 else 0 for pnl in final_pnls])
    }

# Agent comparison function
def compare_agents(agent_types: List[str] = ["PPO", "SAC", "MarketMaking", "MeanReversion"],
                  symbols: List[str] = ["AAPL", "GOOGL"],
                  initial_cash: float = 100000.0,
                  total_timesteps: int = 50000,
                  save_path: str = "comparison_results") -> Dict[str, Any]:
    
    os.makedirs(save_path, exist_ok=True)
    
    results = {}
    
    for agent_type in agent_types:
        print(f"\nTraining {agent_type} agent...")
        
        agent_save_path = f"{save_path}/{agent_type}"
        os.makedirs(agent_save_path, exist_ok=True)
        
        try:
            agent_results = train_agent(
                agent_type=agent_type,
                symbols=symbols,
                initial_cash=initial_cash,
                total_timesteps=total_timesteps,
                use_wandb=False,
                save_path=agent_save_path
            )
            
            results[agent_type] = agent_results
            
        except Exception as e:
            print(f"Error training {agent_type}: {e}")
            results[agent_type] = {"error": str(e)}
    
    comparison_summary = {
        "agent_types": agent_types,
        "symbols": symbols,
        "initial_cash": initial_cash,
        "total_timesteps": total_timesteps,
        "results": results
    }
    
    with open(f"{save_path}/comparison_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)
    
    create_comparison_plots(comparison_summary, save_path)
    
    return comparison_summary

# Plotting function
def create_comparison_plots(results: Dict[str, Any], save_path: str):
    agent_types = results["agent_types"]
    agent_results = results["results"]
    
    metrics = ["mean_reward", "mean_final_pnl", "win_rate", "mean_max_drawdown"]
    metric_names = ["Mean Reward", "Mean Final PnL", "Win Rate", "Mean Max Drawdown"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = []
        labels = []
        
        for agent_type in agent_types:
            if agent_type in agent_results and "evaluation_results" in agent_results[agent_type]:
                eval_results = agent_results[agent_type]["evaluation_results"]
                if metric in eval_results:
                    values.append(eval_results[metric])
                    labels.append(agent_type)
        
        if values:
            bars = axes[i].bar(labels, values)
            axes[i].set_title(metric_name)
            axes[i].set_ylabel(metric_name)
            
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/agent_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    training_times = []
    labels = []
    
    for agent_type in agent_types:
        if agent_type in agent_results and "training_time" in agent_results[agent_type]:
            training_times.append(agent_results[agent_type]["training_time"])
            labels.append(agent_type)
    
    if training_times:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, training_times)
        plt.title("Training Time Comparison")
        plt.ylabel("Training Time (seconds)")
        
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

# Main execution
def main():
    print("DeepQuote RL Training")
    print("=" * 50)
    
    symbols = ["AAPL", "GOOGL"]
    initial_cash = 100000.0
    total_timesteps = 50000
    
    agent_types = ["PPO", "SAC", "MarketMaking", "MeanReversion"]
    
    print(f"Training agents: {agent_types}")
    print(f"Symbols: {symbols}")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print(f"Total timesteps: {total_timesteps:,}")
    
    results = compare_agents(
        agent_types=agent_types,
        symbols=symbols,
        initial_cash=initial_cash,
        total_timesteps=total_timesteps,
        save_path="training_results"
    )
    
    print("\nTraining completed!")
    print("Results saved to training_results/")
    
    for agent_type, agent_results in results["results"].items():
        if "evaluation_results" in agent_results:
            eval_results = agent_results["evaluation_results"]
            print(f"\n{agent_type}:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
            print(f"  Mean Final PnL: ${eval_results['mean_final_pnl']:.2f}")
            print(f"  Win Rate: {eval_results['win_rate']:.2%}")
            print(f"  Training Time: {agent_results['training_time']:.1f}s")

if __name__ == "__main__":
    main() 