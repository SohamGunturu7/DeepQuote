# DeepQuote

DeepQuote is a high-performance market simulation and reinforcement learning environment for developing, testing, and benchmarking trading strategies. It combines a C++ core for speed with Python bindings for ease of use and integration with RL agents.

## Features
- Fast C++ core for simulating realistic market microstructure
- Python bindings for easy integration with RL and data science workflows
- Modular design: core, market, strategies, and bindings
- Built-in support for market making, mean reversion, and pairs trading strategies
- Extensible for custom strategies and agents
- Includes RL environment for training and evaluating agents

## Directory Structure
```
DeepQuote/
├── CMakeLists.txt     # CMake build (C++ library + Python module)
├── setup.py           # pip build of the Python module
├── pyproject.toml
├── include/           # C++ headers (core, market, strategies)
├── src/               # C++ sources and pybind11 bindings
└── python_rl/         # Gymnasium environment, agents, training and demo
```

## Installation

### Prerequisites
- C++17 compiler
- Python 3.8+
- CMake 3.16+ (only for the CMake build)

### Python package (recommended)
Builds the C++ simulator and installs it as the `deepquote_simulator` module:

```bash
pip install .
pip install -r python_rl/requirements.txt
```

### CMake build
```bash
pip install pybind11
mkdir -p build && cd build
cmake ..
make
```
This puts `deepquote_simulator*.so` in `build/`. `python_rl/deepquote_env.py` finds it there
automatically if the module isn't installed.

## Usage

### Python RL Environment
```python
from deepquote_env import DeepQuoteEnv

env = DeepQuoteEnv(symbols=["AAPL", "GOOGL"])
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Run from `python_rl/`:
```bash
python demo.py            # rule-based agents on one simulated market, saves a plot
python train.py --quick   # end-to-end training check (PPO, SAC, rule-based baselines)
python train.py           # full comparison run
```

See `python_rl/README.md` for the observation/action spaces and agent options.

### C++ Core
```cpp
#include "market/market_simulator.h"
#include "market/market_maker.h"

deepquote::MarketSimulator sim({"AAPL"});
deepquote::MarketMakerConfig config;
config.symbols = {"AAPL"};
deepquote::MarketMaker mm(&sim, config);
mm.step();  // place quotes around the fair price
```

## Development

### C++
- Core logic: `src/core/`, `src/market/`, `src/strategies/`
- Headers: `include/`
- Python bindings: `src/bindings/`

### Python
- RL environment and agents: `python_rl/`
