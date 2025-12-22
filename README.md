# Search & Rescue: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-search/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project simulates a multi-agent search-and-rescue mission using the PettingZoo MPE framework. Rescuers (cooperative agents) must guide victims to designated safe zones, navigating around obstacles and using cooperative intelligence to accomplish the task.

## Getting Started

This README covers two main workflows:

- **Run (recommended)**: Start the project via Docker (no local Python setup required).
- **Develop (optional)**: Full local development workflow using a Python virtual environment (`.venv`).

### Prerequisites

- Docker (for running in containers) or Python 3.13+ locally.
- Git for cloning the repository.

### Quickstart — Run with Docker (Recommended)

Build the image:

```bash
docker build -t student-search:latest -f docker/Dockerfile .
```

Run training (mount logs directory):

```bash
docker run --rm -v "${PWD}/search_rescue_logs:/app/search_rescue_logs" student-search:latest
```

Run evaluation (Hydra overrides exposed via environment):

```bash
docker run --rm -v "${PWD}/search_rescue_logs:/app/search_rescue_logs" student-search:latest eval.active=true
```

Open tensorboard (port 6006 exposed): _Note: If you want to change the default port, override the forwarded docker port instead of changing the config file_

```bash
docker run --rm -p 6006:6006 -v "${PWD}/search_rescue_logs:/app/search_rescue_logs" student-search:latest tensorboard.active=true
```

_If you have `make` available, targets are optional convenience wrappers (`make build`, `make train` and `make eval`)._

### Quickstart — Run Locally

```bash
# Training
python -m src.main train.active=true

# Evaluation (with human render)
python -m src.main eval.active=true eval.render_mode=human

# TensorBoard (view training metrics at http://localhost:6006)
python -m src.main tensorboard.active=true
```

## Environment Overview

This project implements a multi-agent reinforcement learning environment where:

- **Rescuers** (cooperative agents) must locate and guide victims to safe zones
- **Victims** are environmental entities that commit to rescuers when approached and follow them
- **Safe Zones** are located at the four corners of the map, each with a unique type (A, B, C, D)
- **Trees** act as obstacles that block vision and cause collisions
- **Type-based Clustering**: Each victim type must be taken to its matching safe zone type

The environment is built using [PettingZoo](https://pettingzoo.farama.org/) and integrates with [TorchRL](https://github.com/pytorch/rl) for training multi-agent policies using MAPPO (Multi-Agent Proximal Policy Optimization).

### Observation Space

Each rescuer agent receives a **local observation vector** with the following components:

1. **Self State** (4 values):
   - Velocity: `[vx, vy]`
   - Position: `[x, y]`

2. **Agent ID** (`num_rescuers` values):
   - One-hot encoding for symmetry breaking

3. **N Closest Landmarks**:
    TBD

4. **Other Agents** (`(num_rescuers - 1) * 2` values):
   - Relative positions of other rescuers: `[rel_x, rel_y]` per agent
   - Masked as `[0, 0]` if not visible (outside vision radius or occluded by trees)

5. **Victims** (`num_victims * 3` values):
   - Relative position: `[rel_x, rel_y]`
   - Type index: `[type]` (0-3)
   - Masked as `[0, 0, -1]` if not visible or already saved

**Total observation dimension**: `4 + num_rescuers + (n_closest_landmarks * 3) + ((num_rescuers - 1) * 2) + (num_victims * 3)`

**Example** (6 rescuers, 12 victims, N=3 closest landmarks):

- Self: 4
- Agent ID: 6
- Landmarks: 3 × 3 = 9
- Other agents: 5 × 2 = 10
- Victims: 12 × 3 = 36
- **Total**: 65 dimensions

### Action Space

The environment supports both discrete and continuous action spaces (configured via `continuous_actions` parameter):

**Continuous Actions (default):**

- 2D acceleration vector: `[ax, ay]` in range `[-1, 1]`
- Actions are applied as:

  ```python
  velocity = velocity * 0.8 + action * 0.1
  velocity = clip(velocity, max_speed=0.08)
  position = position + velocity
  ```

**Discrete Actions:**

- `Discrete(5)` action space with the following mapping:
  - `0`: No-op (no movement)
  - `1`: Up (acceleration `[0.0, 1.0]`)
  - `2`: Down (acceleration `[0.0, -1.0]`)
  - `3`: Left (acceleration `[-1.0, 0.0]`)
  - `4`: Right (acceleration `[1.0, 0.0]`)
- Discrete actions are internally converted to continuous accelerations and follow the same physics

### Reward Structure

The reward system uses assignment-aware shaping that dynamically switches between pickup and escort modes:

| Event                                | Reward                                   | Recipient                                                |
|--------------------------------------| ---------------------------------------- | -------------------------------------------------------- |
| **Successful rescue**                | +100.0                                   | Assigned rescuer (when victim reaches matching safe zone)|
| **Pickup mode (unassigned agents):** |                                          |                                                          |
| Distance to unassigned victim        | -0.1 × distance                          | Non-escorting rescuers (per step)                        |
| Pickup delta shaping                 | +0.2 × (prev_dist - curr_dist)           | Non-escorting rescuers (approaching unassigned victims)  |
| **Escort mode (assigned agents):**   |                                          |                                                          |
| Bounded zone proximity               | +1.0 × exp(-dist_to_zone / 0.5)          | Assigned rescuer (per step, bounded [0,1])               |
| Escort delta shaping                 | +0.5 × (prev_zone_dist - curr_zone_dist) | Assigned rescuer (bringing victim closer to zone)        |
| **Penalties:**                       |                                          |                                                          |
| Tree collision                       | -1.0                                     | Colliding rescuer                                        |
| Boundary violation                   | -0.2                                     | Rescuer near boundary (\|x\| > 0.95 or \|y\| > 0.95)     |
| Agent collision                      | -1.0                                     | Each colliding rescuer (dist < 0.15)                     |
| Idle penalty                         | -0.01                                    | Rescuer with velocity < 1e-3 (per step)                  |

### Termination Conditions

- **Success**: All victims are saved (termination)
- **Timeout**: Maximum steps (`max_cycles`) reached (truncation)

### Environment Parameters

| Parameter              | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `num_rescuers`         | Number of rescuer agents                                |
| `num_victims`          | Number of victim entities                               |
| `num_trees`            | Number of obstacle trees                                |
| `num_safe_zones`       | Number of safe zones (always at corners)                |
| `max_cycles`           | Maximum steps per episode                               |
| `vision_radius`        | Maximum observation distance                            |
| `rescue_radius`        | Distance threshold for rescuing                         |
| `follow_radius`        | Distance threshold for victim commitment                |
| `randomize_safe_zones` | Whether to randomize safe zone positions                |
| `world_size`           | World bounds ([-1, 1] range)                            |

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
- **Entropy Bonus**: Coefficient = 0.001
- **Optimizer**: Adam with learning rate = 0.0003

### Training Examples

**Basic training:**

```bash
python -m src.main train.active=true
```

**Extended training with more steps:**

```bash
python -m src.main train.active=true train.total_timesteps=500000
```

**Training with custom environment:**

```bash
python -m src.main train.active=true env.victims=12 env.rescuers=6 env.trees=8
```

**Training without TensorBoard logging:**

```bash
python -m src.main train.active=true tensorboard.enabled=false
```

### Monitoring Training

Training metrics are logged to TensorBoard. To view:

```bash
python -m src.main tensorboard.active=true
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

Key metrics logged:

- `train/loss/objective`: PPO clipped objective loss
- `train/loss/critic`: Value function loss
- `train/loss/entropy`: Entropy bonus
- `train/episode_reward`: Mean episode reward
- `train/total_frames`: Cumulative frames processed

## Evaluation

### Evaluation Examples

**Basic evaluation:**

```bash
python -m src.main eval.active=true
```

**Evaluation with rendering:**

```bash
python -m src.main eval.active=true eval.render_mode=human eval.games=5
```

**Evaluate specific checkpoint:**

```bash
python -m src.main eval.active=true eval.model_path=search_rescue_logs/checkpoint.pt
```

### Evaluation Metrics

Logged metrics include:

- `eval/episode_reward`: Total reward per episode
- `eval/episode_steps`: Steps per episode
- `eval/mean_reward_per_step`: Average reward per step
- `eval/mean_episode_reward`: Mean reward across episodes
- `eval/mean_episode_steps`: Mean episode length

## Development

This project recommends a local development environment using an isolated Python virtual environment (for example, `.venv`). The steps below are a complete developer workflow — move through them in order.

### Clone & Prepare

```bash
git clone https://github.com/elte-collective-intelligence/student-search.git
cd student-search

# create and activate a local .venv
python -m venv .venv                     # ensure Python 3.13+
source .venv/bin/activate                # On Windows: .venv\Scripts\activate

# upgrade pip and install runtime deps
pip install --upgrade pip
pip install -e .                         # install package in editable mode
pip install -e ".[dev]"                  # install dev dependencies
```

### Developer Tooling

```bash
# install pre-commit hooks used by this repo
pip install pre-commit
pre-commit install
```

### Linting & Formatting

```bash
# run all pre-commit hooks locally
pre-commit run --all-files

# or run formatters/linters directly
black src/ tests/
flake8 src/ tests/
```

### Tests

```bash
pytest -q
```

Run with coverage:

```bash
pytest --cov=src --cov-report=html
```

### Run Locally (Examples)

```bash
# training
python -m src.main train.active=true

# evaluation (human render)
python -m src.main eval.active=true eval.render_mode=human
```

**Notes:**

- The project uses Hydra for configuration; most runtime parameters can be overridden on the command line (e.g., `python -m src.main env.rescuers=8`).
- Logging/TensorBoard: by default logs are saved to the `search_rescue_logs/` directory (configurable in `conf/config.yaml`).

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
│   ├── logger.py             # TensorBoard logging utilities
│   └── curriculum.py         # Curriculum learning utilities
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest configuration
│   ├── test_env_smoke.py     # Environment smoke tests
│   ├── test_models.py        # Model tests
│   └── test_vision.py        # Vision system tests
├── conf/                     # Configuration files
│   └── config.yaml           # Main Hydra configuration
├── docker/                   # Docker support
│   └── Dockerfile            # Container definition
├── image/                    # Documentation images/videos
├── .github/workflows/        # CI/CD workflows
├── pyproject.toml            # Python project configuration
├── Makefile                  # Build automation
├── README.md                 # This file
└── Documentation.pdf         # Detailed project documentation
```

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
  continuous_actions: true
  randomize_safe_zones: true
  vision_radius: 0.4

curriculum:
  enabled: true
  min_trees: 0
  max_trees: 8
  num_stages: 5

train:
  active: false
  total_timesteps: 204800
  batch_size: 256
  n_epochs: 20
  learning_rate: 0.0003

eval:
  active: false
  games: 10
  render_mode: "human"

save_folder: "search_rescue_logs/"
seed: 0

tensorboard:
  active: false
  port: 6006
  enabled: true
```

### Overriding Configuration

Hydra allows command-line overrides:

```bash
# Override single value
python -m src.main train.active=true train.total_timesteps=200000

# Override multiple values
python -m src.main train.active=true env.victims=8 env.rescuers=4 train.learning_rate=0.0005

# Override nested values
python -m src.main train.active=true tensorboard.enabled=false
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
  train.active=true
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
  eval.active=true eval.render_mode=human
```

### Using the Published Image

```bash
# Training
docker run --rm \
  -v ./search_rescue_logs:/app/search_rescue_logs \
  ghcr.io/elte-collective-intelligence/student-search:latest

# Evaluation
docker run --rm \
  -v ./search_rescue_logs:/app/search_rescue_logs \
  ghcr.io/elte-collective-intelligence/student-search:latest \
  eval.active=true
```

### Docker Configuration

- Base image: Python 3.13-slim
- Working directory: `/app`
- Volume mount: `search_rescue_logs` for persistent logs
- Entry point: `python -m src.main`

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

## Documentation

For detailed documentation including:

- Environment design and implementation details
- Reward system explanation
- Training pipeline architecture
- Evaluation methodology
- Performance analysis

See: **[Documentation.pdf](Documentation.pdf)**

## Team Members

- Adorján Nagy-Mohos — [d5vd5e@inf.elte.hu](mailto:d5vd5e@inf.elte.hu)
- Máté Kovács — [u5bky4@inf.elte.hu](mailto:u5bky4@inf.elte.hu)
- Sándor Baranyi — [ct9xfj@inf.elte.hu](mailto:ct9xfj@inf.elte.hu)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Additional Resources

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Hydra Documentation](https://hydra.cc/)
- [PyTorch Documentation](https://pytorch.org/docs/)
