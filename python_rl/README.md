# DeepQuote Reinforcement Learning System

This directory contains the Python-based reinforcement learning system for DeepQuote, which provides AI agents that can learn trading strategies in the C++ market simulator.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python demo.py
```

This will show you:
- Basic environment functionality
- Market making agent behavior
- Mean reversion agent behavior
- Price movement visualization

### 3. Train an Agent

```bash
python train.py
```

This will train a PPO agent and compare it with other strategies.

## 🧠 Architecture

### Environment (`environment.py`)

The `DeepQuoteEnv` class provides a Gym-compatible interface to the C++ market simulator:

- **Observation Space**: Market data + agent state + technical indicators
- **Action Space**: [action_type, symbol_idx, quantity, price] normalized to [0,1]
- **Reward Function**: Based on trading performance and risk management

### Agents (`agents.py`)

Multiple types of RL agents:

1. **CustomPPOAgent**: Custom PPO implementation with PyTorch
2. **StableBaselinesAgent**: Wrapper for Stable Baselines3 (PPO, SAC, TD3, A2C)
3. **MarketMakingAgent**: Rule-based market making strategy
4. **MeanReversionAgent**: Rule-based mean reversion strategy

### Training (`train.py`)

Complete training pipeline with:
- Agent training and evaluation
- Performance comparison
- Model checkpointing
- Weights & Biases integration
- Visualization tools

## 🎯 Available Agents

### Learning-Based Agents

- **PPO** (Proximal Policy Optimization): On-policy, stable learning
- **SAC** (Soft Actor-Critic): Off-policy, sample efficient
- **TD3** (Twin Delayed DDPG): Off-policy, good for continuous actions
- **A2C** (Advantage Actor-Critic): On-policy, simpler than PPO

### Rule-Based Agents

- **MarketMaking**: Places bids and asks around mid-price
- **MeanReversion**: Buys low, sells high based on moving averages
- **Momentum**: Follows trends using short and long-term moving averages
- **Arbitrage**: Exploits price discrepancies between correlated assets
- **GridTrading**: Places orders at regular price intervals
- **VolatilityBreakout**: Trades on volatility expansion and contraction
- **PairsTrading**: Statistical arbitrage on correlated asset pairs

## 📊 State and Action Spaces

### Observation Space
For each symbol (e.g., AAPL, GOOGL):
- **Market Data**: Best bid/ask, mid price, spread
- **Order Book**: Top 5 levels on each side
- **Technical Indicators**: Moving averages, volatility, RSI
- **Agent State**: Cash, inventory, P&L, position value

### Action Space
- **Action Type**: BUY_MARKET, SELL_MARKET, BUY_LIMIT, SELL_LIMIT, CANCEL_ALL, HOLD
- **Symbol Index**: Which symbol to trade
- **Quantity**: Normalized position size (0-1)
- **Price**: Normalized price relative to current mid price

## 🎮 Usage Examples

### Basic Environment Usage

```python
from environment import DeepQuoteEnv
from agents import create_agent

# Create environment
env = DeepQuoteEnv(
    symbols=["AAPL", "GOOGL"],
    initial_cash=100000.0,
    max_position_size=1000.0,
    transaction_cost=0.001
)

# Create agent
agent = create_agent("PPO", env)

# Run episode
obs, info = env.reset()
for step in range(1000):
    action = agent.get_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

### Training a Custom Agent

```python
from train import train_agent

# Train PPO agent
results = train_agent(
    agent_type="PPO",
    symbols=["AAPL", "GOOGL"],
    initial_cash=100000.0,
    total_timesteps=100000,
    learning_rate=3e-4,
    use_wandb=True
)
```

### Comparing Multiple Agents

```python
from train import compare_agents

# Compare different strategies
results = compare_agents(
    agent_types=["PPO", "SAC", "MarketMaking", "MeanReversion", "Momentum", "Arbitrage", "GridTrading", "VolatilityBreakout", "PairsTrading"],
    symbols=["AAPL", "GOOGL"],
    total_timesteps=50000
)
```

### Using Specific Trading Strategies

```python
# Market Making Strategy
market_maker = create_agent("MarketMaking", env, spread_target=0.001, order_size=10.0)

# Mean Reversion Strategy
mean_reverter = create_agent("MeanReversion", env, lookback_period=20, entry_threshold=2.0)

# Momentum Strategy
momentum_trader = create_agent("Momentum", env, short_window=10, long_window=30, momentum_threshold=0.001)

# Arbitrage Strategy (requires multiple symbols)
arbitrage_trader = create_agent("Arbitrage", env, z_score_threshold=2.0, position_size=0.2)

# Grid Trading Strategy
grid_trader = create_agent("GridTrading", env, grid_levels=5, grid_spacing=0.01, base_position_size=10.0)

# Volatility Breakout Strategy
volatility_trader = create_agent("VolatilityBreakout", env, volatility_window=20, breakout_threshold=2.0)

# Pairs Trading Strategy (requires multiple symbols)
pairs_trader = create_agent("PairsTrading", env, entry_threshold=2.0, exit_threshold=0.5, position_size=0.2)
```

## 🔧 Configuration

### Environment Parameters

- `symbols`: List of trading symbols
- `initial_cash`: Starting capital
- `max_position_size`: Maximum position size
- `transaction_cost`: Trading fees as percentage
- `max_steps`: Maximum steps per episode

### Agent Parameters

#### PPO/SAC/TD3
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `gamma`: Discount factor for future rewards

#### Market Making
- `spread_target`: Target bid-ask spread
- `order_size`: Size of each order
- `inventory_scale`: How much to adjust spread based on inventory

#### Mean Reversion
- `lookback_period`: Period for moving average calculation
- `entry_threshold`: Z-score threshold for entry
- `exit_threshold`: Z-score threshold for exit

#### Momentum
- `short_window`: Short-term moving average window
- `long_window`: Long-term moving average window
- `momentum_threshold`: Threshold for momentum signal
- `position_size`: Size of positions to take

#### Arbitrage
- `correlation_threshold`: Minimum correlation between assets
- `z_score_threshold`: Z-score threshold for arbitrage opportunities
- `position_size`: Size of positions to take

#### Grid Trading
- `grid_levels`: Number of grid levels above and below current price
- `grid_spacing`: Spacing between grid levels (as percentage)
- `base_position_size`: Base size for grid orders

#### Volatility Breakout
- `volatility_window`: Window for volatility calculation
- `breakout_threshold`: Threshold for volatility breakout detection
- `position_size`: Size of positions to take

#### Pairs Trading
- `correlation_window`: Window for correlation calculation
- `entry_threshold`: Z-score threshold for entry
- `exit_threshold`: Z-score threshold for exit
- `position_size`: Size of positions to take

## 📈 Performance Metrics

The system tracks:
- **Episode Reward**: Total reward per episode
- **P&L**: Realized and unrealized profit/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss from peak
- **Win Rate**: Percentage of profitable trades

## 🔗 Integration with C++ Simulator

The Python RL system is designed to integrate with the C++ market simulator:

1. **Current**: Standalone Python environment for development/testing
2. **Future**: Direct communication with C++ simulator via:
   - Shared memory
   - Network sockets
   - File-based communication
   - Python bindings (pybind11)

## 🛠️ Development

### Adding New Agents

1. Create a new agent class inheriting from base strategy
2. Implement `get_action()` method
3. Add to `create_agent()` factory function
4. Test with demo script

### Customizing Rewards

Modify the reward calculation in `DeepQuoteEnv._execute_action()`:
- Risk-adjusted returns
- Transaction cost penalties
- Drawdown penalties
- Sharpe ratio optimization

### Adding Technical Indicators

Extend `DeepQuoteEnv._update_technical_indicators()`:
- MACD, Bollinger Bands
- Volume indicators
- Market microstructure features

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **Stable Baselines3**: RL algorithms
- **Gymnasium**: RL environment interface
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Weights & Biases**: Experiment tracking (optional)

## 🎯 Next Steps

1. **Connect to C++ Simulator**: Implement real-time communication
2. **Multi-Agent Training**: Train agents that compete against each other
3. **Advanced Strategies**: Implement pairs trading, statistical arbitrage
4. **Risk Management**: Add position limits, stop losses, portfolio constraints
5. **Backtesting**: Historical data replay and performance analysis
6. **Live Trading**: Real market integration (with proper risk controls)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is part of DeepQuote and follows the same license terms. 