# Search & Rescue: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-search/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A multi-agent reinforcement learning environment and training framework for search-and-rescue operations. This project simulates a cooperative rescue mission where multiple rescuers (cooperative agents) must guide victims to designated safe zones while navigating around obstacles, using the PettingZoo MPE framework and TorchRL for training.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Training](#training)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Docker Support](#docker-support)
- [Development](#development)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Team Members](#team-members)
- [License](#license)

## Overview

This project implements a multi-agent reinforcement learning environment where:

- **Rescuers** (cooperative agents) must locate and guide victims to safe zones
- **Victims** are environmental entities that commit to rescuers when approached and follow them
- **Safe Zones** are located at the four corners of the map, each with a unique type (A, B, C, D)
- **Trees** act as obstacles that block vision and cause collisions
- **Type-based Clustering**: Each victim type must be taken to its matching safe zone type

The environment is built using [PettingZoo](https://pettingzoo.farama.org/) and integrates with [TorchRL](https://github.com/pytorch/rl) for training multi-agent policies using MAPPO (Multi-Agent Proximal Policy Optimization).

### Why MAPPO?

We chose MAPPO (Multi-Agent Proximal Policy Optimization) for this environment because:

1. **Decentralized execution with centralized training**: Agents learn with global information but act using only local observations - essential for real-world search-and-rescue scenarios.
2. **Credit assignment**: The centralized critic helps attribute rewards correctly among cooperating agents.
3. **Scalability**: MAPPO handles varying numbers of agents without major architectural changes.
4. **Stability**: PPO's clipping mechanism provides stable training even with the complex multi-agent dynamics.

### Why PettingZoo + TorchRL?

- **PettingZoo** provides a standardized API for multi-agent environments, making our environment interoperable with various RL frameworks.
- **TorchRL** offers efficient data collection, replay buffers, and loss modules optimized for PyTorch, enabling fast experimentation.

## Quick Start

### Training a Policy

Train a multi-agent policy with default configuration:

```bash
python dfhdfgjfdgdjkghkmain.py train.active=true
```

Train with custom parameters:

```bash
python main.py train.active=true train.total_timesteps=200000 env.victims=8 env.rescuers=4
```

### Evaluating a Trained Policy

Evaluate the latest trained model:

```bash
python main.py eval.active=true
```

Evaluate with rendering:

```bash
python main.py eval.active=true eval.render_mode=human eval.games=10
```

### Launching TensorBoard

View training metrics:

```bash
python main.py tensorboard.active=true
```

Then open http://localhost:6006 in your browser.

## Environment Details

### Observation Space

Each rescuer agent receives an observation vector containing:

1. **Self State** (4 values):

   - Velocity: `[vx, vy]`
   - Position: `[x, y]`
2. **Agent ID** (`num_rescuers` values):

   - One-hot encoding for symmetry breaking
3. **Safe Zones** (`num_safe_zones * 3` values):

   - Relative position: `[rel_x, rel_y]`
   - Type index: `[type]` (0-3)
4. **Trees** (`num_trees * 2` values):

   - Relative position: `[rel_x, rel_y]` (masked as `[0, 0]` if not visible)
5. **Victims** (`num_victims * 3` values):

   - Relative position: `[rel_x, rel_y]`
   - Type index: `[type]` (masked as `[0, 0, -1]` if not visible or saved)

**Total observation dimension**: `4 + num_rescuers + (num_safe_zones * 3) + (num_trees * 2) + (num_victims * 3)`

### Action Space

Continuous 2D acceleration vector: `[ax, ay]` in range `[-1, 1]`

Actions are applied as:

```python
velocity = velocity * 0.8 + action * 0.1
velocity = clip(velocity, max_speed=0.08)
position = position + velocity
```

### Reward Structure

| Event                    | Reward                  | Recipient                                  |
| ------------------------ | ----------------------- | ------------------------------------------ |
| Successful rescue        | +100.0                  | Assigned rescuer                           |
| Assistance during rescue | +10.0                   | Nearby rescuers (within `follow_radius`) |
| Escorting victim         | +1.0 / (distance + eps) | Assigned rescuer (per step)                |
| Distance shaping         | +0.1\* delta_distance   | All rescuers (getting closer)              |
| Tree collision           | -1.0                    | Colliding rescuer                          |
| Boundary violation       | -1.0                    | Violating rescuer                          |
| Agent collision          | -5.0                    | Both colliding rescuers                    |

### Termination Conditions

- **Success**: All victims are saved (termination)
- **Timeout**: Maximum steps (`max_cycles`) reached (truncation)

### Environment Parameters

| Parameter          | Default | Description                              |
| ------------------ | ------- | ---------------------------------------- |
| `num_rescuers`   | 2       | Number of rescuer agents                 |
| `num_victims`    | 2       | Number of victim entities                |
| `num_trees`      | 5       | Number of obstacle trees                 |
| `num_safe_zones` | 4       | Number of safe zones (always at corners) |
| `max_cycles`     | 200     | Maximum steps per episode                |
| `vision_radius`  | 0.5     | Maximum observation distance             |
| `rescue_radius`  | 0.15    | Distance threshold for rescuing          |
| `follow_radius`  | 0.2     | Distance threshold for victim commitment |
| `world_size`     | 2.0     | World bounds ([-1, 1] range)             |

### Environment Features

1. **Multi-Agent Coordination**

   - Multiple rescuers work together to rescue victims
   - Decentralized policy execution (each agent uses local observations)
   - Centralized value function for better credit assignment
2. **Victim Commitment System**

   - Victims commit to rescuers when approached (within `follow_radius`)
   - Hysteresis prevents rapid switching between rescuers
   - Victims follow committed rescuers until rescued or rescuer moves away
3. **Vision and Occlusion**

   - Limited vision radius for realistic observation constraints
   - Tree obstacles block line-of-sight
   - Masked observations for non-visible entities
4. **Physics Simulation**

   - Velocity-based movement with acceleration actions
   - Collision detection and response (walls, trees, agents)
   - Soft repulsion between agents to prevent clustering
5. **Rendering**

   - Real-time visualization using Pygame
   - Color-coded victims and safe zones by type
   - Vision radius visualization
   - Episode recording support

## Training

### Training Algorithm

The project uses **MAPPO (Multi-Agent Proximal Policy Optimization)**:

- **Policy (Actor)**: Decentralized, each agent uses local observations
- **Value (Critic)**: Centralized, uses all agent observations
- **Shared Parameters**: Homogeneous agents share network weights
- **GAE**: Generalized Advantage Estimation (γ=0.99, λ=0.95)
- **Clipping**: PPO clip epsilon = 0.2
- **Entropy Bonus**: Coefficient = 0.1 (encourages exploration)

### Training Process

1. **Data Collection**: `SyncDataCollector` collects `frames_per_batch` steps
2. **Advantage Computation**: GAE computes advantages from collected data
3. **PPO Updates**: Multiple epochs (`num_epochs`) of minibatch updates
4. **Logging**: Metrics logged to TensorBoard
5. **Checkpointing**: Model saved at end of training

### Training Configuration

Key training parameters (configurable in `conf/config.yaml`):

```yaml
train:
  active: false
  total_timesteps: 102400 # Total environment steps
  batch_size: 2048 # Minibatch size for PPO updates
  n_epochs: 20 # PPO update epochs per batch
  learning_rate: 0.001 # Adam optimizer learning rate
  seed: 0 # Random seed
  render_mode: null # Rendering during training (usually None)
```

### Training Examples

**Basic training:**

```bash
python main.py train.active=true
```

**Extended training with more steps:**

```bash
python main.py train.active=true train.total_timesteps=500000
```

**Training with custom environment:**

```bash
python main.py train.active=true env.victims=12 env.rescuers=6 env.trees=8
```

**Training without TensorBoard logging:**

```bash
python main.py train.active=true tensorboard.enabled=false
```

### Monitoring Training

Training metrics are logged to TensorBoard. To view:

```bash
python main.py tensorboard.active=true
```

Key metrics logged:

- `train/loss/objective`: PPO clipped objective loss
- `train/loss/critic`: Value function loss
- `train/loss/entropy`: Entropy bonus
- `train/episode_reward`: Mean episode reward
- `train/total_frames`: Cumulative frames processed

## Evaluation

### Evaluation Process

1. **Model Loading**: Loads checkpoint latest --input
2. **Environment Setup**: Creates environment with saved configuration
3. **Policy Execution**: Runs episodes with trained policy
4. **Metric Collection**: Tracks rewards, steps, rescues, etc.
5. **Logging**: Logs metrics to TensorBoard

### Evaluation Configuration

```yaml
eval:
  active: false
  games: 10 # Number of episodes to evaluate
  render_mode: "human" # Rendering mode ("human" or null)
```

### Evaluation Examples

**Basic evaluation:**

```bash
python main.py eval.active=true
```

**Evaluation with rendering:**

```bash
python main.py eval.active=true eval.render_mode=human eval.games=5
```

**Evaluate specific checkpoint:**

```bash
python main.py eval.active=true eval.model_path=logs/checkpoint.pt
```

### Evaluation Metrics

Logged metrics include:

- `eval/episode_reward`: Total reward per episode
- `eval/episode_steps`: Steps per episode
- `eval/mean_reward_per_step`: Average reward per step
- `eval/mean_episode_reward`: Mean reward across episodes
- `eval/mean_episode_steps`: Mean episode length

## Architecture

### Environment Architecture

```
SearchAndRescueEnv (PettingZoo ParallelEnv)
    ├── Entity Management
    │   ├── Rescuers (agents)
    │   ├── Victims (environmental entities)
    │   ├── Trees (obstacles)
    │   └── Safe Zones (goals)
    ├── Physics System
    │   ├── Movement (velocity-based)
    │   ├── Collision Detection
    │   └── Boundary Handling
    ├── Vision System
    │   ├── Distance Checking
    │   └── Occlusion Detection
    ├── Commitment System
    │   ├── Victim Assignment
    │   └── Following Behavior
    └── Reward Computation
        ├── Rescue Rewards
        ├── Distance Shaping
        └── Penalties
```

### Training Architecture

```
Training Pipeline
    ├── Environment (SearchAndRescueEnv)
    ├── Policy Network (Decentralized Actor)
    ├── Value Network (Centralized Critic)
    ├── Data Collector (SyncDataCollector)
    ├── Replay Buffer (LazyTensorStorage)
    ├── PPO Loss Module (ClipPPOLoss)
    ├── Optimizer (Adam)
    └── Logger (TensorBoard)
```

### Model Architecture

**Policy Network:**

- Input: Local observation `[batch, n_agents, obs_dim]`
- Architecture: 2-layer MLP, 64 units per layer, Tanh activation
- Output: Location and scale for TanhNormal distribution
- Parameters: Shared across agents (homogeneous)

**Value Network:**

- Input: All agent observations `[batch, n_agents, obs_dim]`
- Architecture: 2-layer MLP, 128 units per layer, Tanh activation
- Output: State value `[batch, n_agents, 1]`
- Centralized: True (sees all observations)

## Configuration

Configuration is managed using [Hydra](https://hydra.cc/). The main configuration file is `conf/config.yaml`.

### Configuration Structure

```yaml
env:
  victims: 12
  rescuers: 6
  trees: 8
  safe_zones: 4
  max_cycles: 300
  continuous_actions: false
  vision_radius: 0.4

train:
  active: false
  total_timesteps: 102400
  batch_size: 2048
  n_epochs: 20
  seed: 0
  render_mode: null
  learning_rate: 0.001

eval:
  active: false
  games: 10
  render_mode: "human"

save_folder: "search_rescue_logs/"

tensorboard:
  active: false
  port: 6006
  enabled: true
```

### Overriding Configuration

Hydra allows command-line overrides:

```bash
# Override single value
python main.py train.active=true train.total_timesteps=200000

# Override multiple values
python main.py train.active=true env.victims=8 env.rescuers=4 train.learning_rate=0.0005

# Override nested values
python main.py train.active=true tensorboard.enabled=false
```

### Configuration Files

- `conf/config.yaml`: Main configuration file
- Additional configs can be added to `conf/` directory

## Project Structure

```
.
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py               # Main entry point (Hydra CLI)
│   ├── sar_env.py            # SearchAndRescueEnv implementation
│   ├── models.py             # Neural network model factories
│   ├── train.py              # Training pipeline
│   ├── eval.py               # Evaluation pipeline
│   └── logger.py             # TensorBoard logging utilities
|   |── curriculum.py         # Curriculum learning utilities
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest configuration
│   ├── test_env_smoke.py     # Environment smoke tests
│   └── test_vision.py        # Vision system tests
├── conf/                     # Configuration files
│   └── config.yaml           # Main Hydra configuration
├── docker/                   # Docker support
│   └── Dockerfile            # Container definition
├── images/                   # Documentation images/videos
├── .github/workflows/        # CI/CD workflows
├── requirements.txt          # Python dependencies
├── Makefile                  # Build automation
├── README.md                 # This file
└── Documentation.pdf         # Detailed project documentation
```

## Docker Support

### Building the Docker Image

```bash
make build
```

Or manually:

```bash
docker build -t student-search:latest -f docker/Dockerfile .
```

### Running Training in Docker

```bash
make train
```

Or manually:

```bash
docker run --rm \
  -v $(pwd)/search_rescue_logs:/app/search_rescue_logs \
  student-search:latest \
  python main.py train.active=true
```

### Running Evaluation in Docker

```bash
make eval
```

Or manually:

```bash
docker run --rm -it \
  -v $(pwd)/search_rescue_logs:/app/search_rescue_logs \
  student-search:latest \
  python main.py eval.active=true eval.render_mode=human
```

### Docker Configuration

- Base image: Python 3.13
- Working directory: `/app`
- Volume mount: `search_rescue_logs` for persistent logs
- Entry point: `python main.py`

## Development

### Prerequisites

- Python 3.13 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/adorjannm/student-search.git
   cd student-search
   ```
2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

See `requirements.txt` for the complete list of dependencies with versions.

### Code Style

The project uses:

- **Black**: Code formatting (line length 88)
- **flake8**: Linting
- **Pre-commit hooks**: Automated checks

### Setting Up Development Environment

1. **Install pre-commit hooks:**

   ```bash
   pip install pre-commit
   pre-commit install
   ```
2. **Run code formatting:**

   ```bash
   black src/ tests/
   ```
3. **Run linting:**

   ```bash
   flake8 src/ tests/
   ```

### Project Guidelines

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Add tests for new features

## Testing

### Running Tests

Run all tests:

```bash
pytest
```

Run specific test file:

```bash
pytest tests/test_env_smoke.py
```

Run with coverage:

```bash
pytest --cov=src --cov-report=html
```

### Test Structure

- `tests/conftest.py`: Pytest fixtures and configuration
- `tests/test_env_smoke.py`: Environment smoke tests
- `tests/test_vision.py`: Vision system tests

### Writing Tests

Tests use pytest and should:

- Be placed in `tests/` directory
- Use descriptive names starting with `test_`
- Use fixtures from `conftest.py` when possible
- Test both success and failure cases

## API Documentation

### Main Entry Point

#### `main.main(cfg: DictConfig)`

Main entry point for training and evaluation. Uses Hydra for configuration management.

**Modes:**

- Training: `train.active=true`
- Evaluation: `eval.active=true`
- TensorBoard: `tensorboard.active=true`

### Environment

#### `SearchAndRescueEnv`

Multi-agent search and rescue environment.

**Key Methods:**

- `reset(seed, options)`: Reset environment to initial state
- `step(actions)`: Execute one environment step
- `render()`: Render current state
- `close()`: Clean up resources

**See `src/sar_env.py` for complete API documentation.**

### Training

#### `train.train(...)`

Train a multi-agent policy using MAPPO.

**Parameters:**

- `steps`: Total environment steps
- `batch_size`: Minibatch size
- `learning_rate`: Optimizer learning rate
- `num_epochs`: PPO update epochs
- `frames_per_batch`: Steps per batch
- `seed`: Random seed
- `save_folder`: Log directory
- `enable_logging`: Enable TensorBoard
- `**env_kwargs`: Environment parameters

**See `src/train.py` for complete API documentation.**

### Evaluation

#### `evaluate.evaluate(...)`

Evaluate a trained policy.

**Parameters:**

- `model_path`: Checkpoint path (None for auto-detect)
- `save_folder`: Log directory
- `num_games`: Number of episodes
- `enable_logging`: Enable TensorBoard
- `**env_kwargs`: Environment parameters

**See `src/eval.py` for complete API documentation.**

### Models

#### `make_policy(env, num_rescuers, device)`

Create a probabilistic policy network.

**Returns:** `ProbabilisticActor` instance

#### `make_critic(env, num_rescuers, device)`

Create a value network.

**Returns:** `ValueOperator` instance

**See `src/models.py` for complete API documentation.**

## Documentation

### Project Documentation

For detailed documentation including:

- Environment design and implementation details
- Reward system explanation
- Training pipeline architecture
- Evaluation methodology
- Performance analysis

See: **[Documentation.pdf](Documentation.pdf)**

### Code Documentation

All source code includes comprehensive docstrings following Google/NumPy style:

- Module-level docstrings explain purpose and usage
- Class docstrings describe attributes and behavior
- Function docstrings include Args, Returns, Raises, and Examples

Generate HTML documentation:

```bash
# Using pydoc
python -m pydoc -w src/

# Using Sphinx (if configured)
sphinx-build docs/ docs/_build/
```

## Team Members

- **Adorján Nagy-Mohos** - d5vd5e@inf.elte.hu
- **Máté Kovács** - u5bky4@inf.elte.hu
- **Sándor Baranyi** - ct9xfj@inf.elte.hu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Additional Resources

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Hydra Documentation](https://hydra.cc/)
- [PyTorch Documentation](https://pytorch.org/docs/)
