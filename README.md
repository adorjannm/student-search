# Search & Rescue: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-search/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-search)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project simulates a multi-agent search-and-rescue mission using the PettingZoo MPE framework. Rescuers (adversaries) must guide victims to designated safe zones, navigating around obstacles and using cooperative intelligence to accomplish the task.

## Getting Started

This README covers two main workflows:

- Run (recommended): start the project via Docker (no local Python setup required).
- Develop (optional): full local development workflow using a Python virtual environment (`.venv`).

### Prerequisites

- Docker (for running in containers) or Python 3.13+ locally.
- Git for cloning the repository.

### Quickstart — Run with Docker (recommended)

Build the image (only if you need a local image):

```bash
docker build -t student-search:latest .
```

Run training (mount logs directory):

```bash
docker run --rm -v "${PWD}/search_rescue_logs:/app/search_rescue_logs" student-search:latest
```

Run evaluation (Hydra overrides exposed via environment):

```bash
docker run --rm -v "${PWD}/search_rescue_logs:/app/search_rescue_logs" student-search:latest eval.active=true
```

If you have `make` available, targets are optional convenience wrappers (see `Makefile`).

## Project Overview

- **Objective**: Simulate a search-and-rescue operation where rescuers lead victims to specific safe zones based on clustering.
- **Framework**: Built using the PettingZoo MPE environment.
- **Agents**:
  - **Rescuers**: Adversarial agents guiding victims to the correct safe zones.
  - **Victims**: Agents that need to be rescued by being taken to matching safe zones.
- **Safe Zones**: Defined zones in the map’s four corners; each type of victim has a matching type of safe zone.

## Key Features

1. **Safe Zones**:
   - **Static Locations**: Safe zones are positioned at each corner of the map.
   - **Different Types**: Each victim type corresponds to a unique type of safe zone, introducing a clustering challenge.

2. **Reward System**:
   - Rescuers earn rewards for successfully moving victims to their designated safe zones.
   - Victims are incentivized to avoid capture, reinforcing the search-and-rescue dynamics.

3. **Collision Detection**:
   - Refined detection ensures rescues occur only when victims and rescuers are close.
   - Introduced obstacle collision logic to prevent straightforward rescues.

## Documentation

For a detailed description of the environment, reward system, training pipeline, and evaluation process, refer to the complete project documentation:
📄 [**Documentation.pdf**](Documentation.pdf)

The documentation includes:

- Functionalities of the search-and-rescue environment.
- Code structure and modular components.
- Training and evaluation pipelines with performance metrics.

## Usage

Run the following commands to start training or evaluating the environment:

### Training

```bash
docker run -v ./search_rescue_logs:/app/search_rescue_logs ghcr.io/elte-collective-intelligence/student-search:latest
# or
make run train # if you have Make installed
```

### Evaluation

```bash
docker run -v ./search_rescue_logs:/app/search_rescue_logs ghcr.io/elte-collective-intelligence/student-search:latest eval.active=true
# or
make run eval # if you have Make installed
```

 ![Run with 1 missing agent](images/Rescue1.mp4)

## Development

This project recommends a local development environment using an isolated Python virtual environment (for example, `.venv`). The steps below are a complete developer workflow — move through them in order.

### Clone & prepare

```bash
git clone https://github.com/elte-collective-intelligence/student-search.git
cd student-search

# create and activate a local .venv
python -m venv .venv # ensure Python 3.13+
source .venv/bin/activate

# upgrade pip and install runtime deps
pip install --upgrade pip
pip install -r requirements.txt
```

### Developer tooling

```bash
# install pre-commit hooks used by this repo
pip install pre-commit
pre-commit install
```

### Linting & formatting

```bash
# run all pre-commit hooks locally
pre-commit run --all-files

# or run formatters/linters directly
black .
flake8
```

### Tests

```bash
pytest -q
```

### Run locally (examples)

```bash
# training
python -m src.main train.active=true

# evaluation (human render)
python -m src.main eval.active=true eval.render_mode=human
```

Notes:

- The project uses Hydra for configuration; most runtime parameters can be overridden on the command line (e.g., `python -m src.main env.num_rescuers=8`).
- Logging/tensorboard: by default logs are saved to the `search_rescue_logs/` directory (configurable in `conf/config.yaml`).

## Team Members

- Adorján Nagy-Mohos — [d5vd5e@inf.elte.hu](mailto:d5vd5e@inf.elte.hu)
- Máté Kovács — [u5bky4@inf.elte.hu](mailto:u5bky4@inf.elte.hu)
- Sándor Baranyi — [ct9xfj@inf.elte.hu](mailto:ct9xfj@inf.elte.hu)

## License

This project is licensed under the MIT License. - see the [LICENSE](LICENSE) file for details.
