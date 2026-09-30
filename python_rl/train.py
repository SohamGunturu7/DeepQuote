"""
DeepQuote RL Training Script
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional
import os
import json
import time
from datetime import datetime

from deepquote_env import DeepQuoteEnv
from agents import create_agent, StableBaselinesAgent
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    import wandb
except ImportError:
    wandb = None

try:
    import tensorboard  # noqa: F401  (optional: only needed for SB3's tensorboard logging)
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

RL_AGENTS = ["PPO", "SAC", "TD3", "A2C"]

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
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        'episode_reward': avg_reward,
                        'episode_length': avg_length,
                        'episode': len(self.episode_rewards)
                    })
        
        return True

# Main training function
def train_agent(agent_type: str = "PPO",
                symbols: List[str] = ["AAPL"],
                initial_cash: float = 100000.0,
                max_steps: int = 1000,
                total_timesteps: int = 100000,
                learning_rate: float = 3e-4,
                use_wandb: bool = False,
                eval_episodes: int = 10,
                save_path: str = "models") -> Dict[str, Any]:
    
    os.makedirs(save_path, exist_ok=True)
    
    if use_wandb and wandb is None:
        print("wandb is not installed; continuing without it")
        use_wandb = False
    
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
    
    def make_env():
        return Monitor(DeepQuoteEnv(symbols=symbols, initial_cash=initial_cash, max_steps=max_steps))
    
    env = make_env()
    
    print(f"Training {agent_type} agent on {symbols}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    if agent_type in RL_AGENTS:
        agent = StableBaselinesAgent(
            env=env,
            agent_type=agent_type,
            learning_rate=learning_rate,
            tensorboard_log=f"{save_path}/tensorboard_logs" if HAS_TENSORBOARD else None
        )
        
        # Eval env must be normalized like the training env; EvalCallback syncs the statistics
        eval_env = VecNormalize(DummyVecEnv([make_env]), training=False, norm_reward=False)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}/best_model",
            log_path=f"{save_path}/eval_logs",
            eval_freq=max(total_timesteps // 10, max_steps),
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, max_steps),
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
        # Rule-based agents read env attributes (symbols, limits), so give them the raw env
        agent = create_agent(agent_type, env.unwrapped)
        
        print(f"Testing {agent_type} agent...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(eval_episodes):
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
    eval_results = evaluate_agent(agent, env, n_episodes=eval_episodes, max_steps=max_steps)
    
    results = {
        "agent_type": agent_type,
        "symbols": symbols,
        "initial_cash": initial_cash,
        "training_time": training_time,
        "total_timesteps": total_timesteps,
        "evaluation_results": eval_results,
        "model_path": f"{save_path}/{agent_type}_final" if agent_type in RL_AGENTS else None
    }
    
    with open(f"{save_path}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if use_wandb:
        wandb.finish()
    
    return results

# Agent evaluation function
def evaluate_agent(agent, env: DeepQuoteEnv, n_episodes: int = 10, max_steps: int = 1000) -> Dict[str, float]:
    episode_rewards = []
    episode_lengths = []
    final_pnls = []
    max_drawdowns = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_pnls = []
        
        for step in range(max_steps):
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
            # Largest peak-to-trough drop in P&L, as a fraction of initial cash
            equity = np.array(episode_pnls) + env.unwrapped.initial_cash
            peaks = np.maximum.accumulate(equity)
            max_drawdowns.append(float(np.max((peaks - equity) / peaks)))
    
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_final_pnl": float(np.mean(final_pnls)),
        "std_final_pnl": float(np.std(final_pnls)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)) if max_drawdowns else 0.0,
        "win_rate": float(np.mean([1 if pnl > 0 else 0 for pnl in final_pnls]))
    }

# Agent comparison function
def compare_agents(agent_types: List[str] = ["PPO", "SAC", "MarketMaking", "MeanReversion"],
                  symbols: List[str] = ["AAPL"],
                  initial_cash: float = 100000.0,
                  total_timesteps: int = 50000,
                  max_steps: int = 1000,
                  eval_episodes: int = 10,
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
                max_steps=max_steps,
                eval_episodes=eval_episodes,
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
    parser = argparse.ArgumentParser(description="Train and compare DeepQuote agents on the C++ market simulator")
    parser.add_argument("--agents", nargs="+", default=["PPO", "SAC", "MarketMaking", "MeanReversion"],
                        help="Agent types: PPO SAC TD3 A2C MarketMaking MeanReversion Momentum ...")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps per RL agent")
    parser.add_argument("--max-steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-path", default="training_results")
    parser.add_argument("--quick", action="store_true", help="Tiny run to check everything works end to end")
    args = parser.parse_args()
    
    if args.quick:
        args.timesteps, args.max_steps, args.eval_episodes = 2048, 200, 2
    
    print("DeepQuote RL Training")
    print("=" * 50)
    print(f"Training agents: {args.agents}")
    print(f"Symbols: {args.symbols}")
    print(f"Initial cash: ${args.initial_cash:,.2f}")
    print(f"Total timesteps: {args.timesteps:,}")
    
    results = compare_agents(
        agent_types=args.agents,
        symbols=args.symbols,
        initial_cash=args.initial_cash,
        total_timesteps=args.timesteps,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        save_path=args.save_path
    )
    
    print("\nTraining completed!")
    print(f"Results saved to {args.save_path}/")
    
    for agent_type, agent_results in results["results"].items():
        if "evaluation_results" in agent_results:
            eval_results = agent_results["evaluation_results"]
            print(f"\n{agent_type}:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
            print(f"  Mean Final PnL: ${eval_results['mean_final_pnl']:.2f}")
            print(f"  Win Rate: {eval_results['win_rate']:.2%}")
            print(f"  Training Time: {agent_results['training_time']:.1f}s")
        else:
            print(f"\n{agent_type}: FAILED - {agent_results.get('error')}")

if __name__ == "__main__":
    main()
