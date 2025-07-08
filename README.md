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
├── CMakeLists.txt                # CMake build configuration
├── deepquote_simulator.cpython-310-darwin.so  # Pre-built Python extension
├── include/                      # C++ header files
├── src/                          # C++ source files
├── build/                        # Build artifacts
├── python_rl/                    # Python RL environment, agents, and demos
├── test/                         # C++ unit tests
├── setup.py                      # Python package setup
├── README.md                     # This file
```

## Installation

### Prerequisites
- C++17 compatible compiler
- Python 3.10+
- CMake (for building from source)

### Python Package (Recommended)
If the pre-built `deepquote_simulator.cpython-310-darwin.so` is compatible with your system, you can use the Python interface directly:

```bash
cd python_rl
pip install -r requirements.txt
```

### Build from Source
If you need to rebuild the C++ core and Python bindings:

```bash
mkdir -p build && cd build
cmake ..
make
cd ../python_rl
pip install -r requirements.txt
```

## Usage

### Python RL Environment
Example usage in Python:

```python
from deepquote_env import DeepQuoteEnv
env = DeepQuoteEnv()
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

See `python_rl/demo.py` for more examples.

### C++ Core
You can use the C++ classes directly for custom simulations. See the `test/` directory for usage examples.

## Development

### C++
- Core logic: `src/core/`, `src/market/`, `src/strategies/`
- Headers: `include/`
- Python bindings: `src/bindings/`

### Python
- RL environment and agents: `python_rl/`

## Testing

### Python
Run integration tests:
```bash
cd python_rl
python test_simple_integration.py
```

### C++
Build and run tests:
```bash
mkdir -p build && cd build
cmake ..
make
ctest
```