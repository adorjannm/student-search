"""
Training script using TorchRL with MAPPO for Multi-Agent Reinforcement Learning.

Implements CTDE (Centralized Training with Decentralized Execution):
- Actors: Use local observations for each agent (decentralized)
- Critic: Uses global state for value estimation (centralized)
- Parameter sharing: All agents share the same policy network

Based on: https://arxiv.org/abs/2103.01955 (MAPPO paper)
"""

import time
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter
from torchrl.envs.utils import check_env_specs
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from src.domain.sar_env import SearchAndRescueEnv
from src.rl.models import make_mappo_models, make_ppo_models
from src.infra.config import make_run_context
from src.infra.logging_tb import TensorboardLogger
from src.infra.checkpointing import save_checkpoint


def make_env(env_kwargs, device="cpu"):
    """Create and wrap the environment."""
    env = SearchAndRescueEnv(**env_kwargs, device=device)
    env = TransformedEnv(
        env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    return env


def train(
    steps: int = 100000,
    batch_size: int = 256,
    seed: int = 0,
    save_folder: str = "search_rescue_logs/",
    algorithm: str = "mappo",
    tb_enabled: bool = True,
    **env_kwargs,
):
    """Train a PPO/MAPPO agent on the search and rescue environment.

    Implements MARL with CTDE:
    - All agents share the same policy (parameter sharing)
    - Critic uses global state for centralized training
    - Actors execute using only local observations

    Args:
        steps: Total training steps.
        batch_size: Batch size for training.
        seed: Random seed.
        save_folder: Folder to save logs and models.
        algorithm: "ppo" (local critic) or "mappo" (CTDE with global state critic).
        tb_enabled: Enable tensorboard logging.
        **env_kwargs: Environment configuration.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env_kwargs["seed"] = seed
    env = make_env(env_kwargs, device=device)

    # Check environment specs
    check_env_specs(env)

    env_name = getattr(env.base_env, "metadata", {}).get("name", type(env).__name__)

    num_agents = getattr(env, "base_env", env).num_rescuers
    print(f"Starting MARL training on {env_name}.")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Number of agents: {num_agents}")
    print(f"Observation shape (per-agent): {env.observation_spec['observation'].shape}")
    print(f"Global state shape: {env.observation_spec['state'].shape}")
    print(f"Action spec: {env.action_spec}")

    # Create models based on algorithm choice
    if algorithm.lower() == "mappo":
        # MAPPO: actor uses local obs, critic uses global state (CTDE)
        actor, critic = make_mappo_models(env, device=device)
        print("Critic using: global state (CTDE - Centralized Training)")
        print("Actor using: local observations (Decentralized Execution)")
    else:
        # PPO: both actor and critic use local observation
        actor, critic = make_ppo_models(env, device=device)
        print("Critic using: local observation (Independent PPO)")

    # Create GAE module for advantage estimation
    adv_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
    )

    # Create PPO loss module
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=0.5,
        loss_critic_type="smooth_l1",
    )

    # Create optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

    # Create data collector
    frames_per_batch = min(batch_size, steps // 4)
    print(f"Batch size (frames per batch): {frames_per_batch}")
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=steps,
        split_trajs=False,
        device=device,
    )

    # Run context + centralized TB logger
    ctx = make_run_context(
        save_folder=save_folder,
        env_name=env_name,
        algorithm=algorithm,
        seed=seed,
    )
    logger = TensorboardLogger.create(ctx, enabled=tb_enabled)

    logger.log_text(
        "run/banner",
        r"""```text
      ____        ____
     / ___|  __ _|  _ \
     \___ \ / _` | |_) |
      ___) | (_| |  _ <
     |____/ \__,_|_| \_\

    SEARCH AND RESCUE — Multi-Agent RL (MAPPO)
    ```""",
        step=0,
    )

    # Optional: log a short run header once
    logger.log_text(
        "run/info",
        f"env={env_name}\nalgorithm={algorithm}\nseed={seed}\nnum_agents={num_agents}",
        step=0,
    )

    # Training loop
    ppo_epochs = 4
    total_frames = 0
    start_time = time.time()

    pbar = tqdm(total=steps, desc="Training", unit="frames")

    for i, batch in enumerate(collector):
        total_frames += batch.numel()

        # Compute advantage using GAE
        with torch.no_grad():
            adv_module(batch)

        # Flatten batch for training
        batch_flat = batch.reshape(-1)

        # PPO update - iterate through batch with mini-batches
        minibatch_size = min(64, len(batch_flat))
        loss_sum = torch.tensor(0.0)
        loss_objective_sum = 0.0
        loss_critic_sum = 0.0
        loss_entropy_sum = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(len(batch_flat))

            for start_idx in range(0, len(batch_flat), minibatch_size):
                end_idx = min(start_idx + minibatch_size, len(batch_flat))
                mb_indices = indices[start_idx:end_idx]
                mb = batch_flat[mb_indices].to(device)

                # Compute PPO loss
                loss = loss_module(mb)

                # Aggregate losses
                loss_sum = (
                    loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                )

                # Track individual losses for logging
                loss_objective_sum += loss["loss_objective"].item()
                loss_critic_sum += loss["loss_critic"].item()
                loss_entropy_sum += loss["loss_entropy"].item()
                num_updates += 1

                optim.zero_grad()
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()

        # Logging
        reward = batch["next", "reward"].mean().item()
        done_rate = batch["next", "done"].float().mean().item()

        logs = {
            "reward_mean": reward,
            "done_rate": done_rate,
            "loss_total": loss_sum.item(),
        }

        if num_updates > 0:
            logs.update(
                {
                    "loss/objective": loss_objective_sum / num_updates,
                    "loss/critic": loss_critic_sum / num_updates,
                    "loss/entropy": loss_entropy_sum / num_updates,
                }
            )

        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0.0
        logs["fps"] = fps

        logger.log_dict("train", logs, total_frames)

        # keep tqdm behavior
        pbar.update(batch.numel())
        pbar.set_postfix(
            reward=f"{reward:.2f}",
            fps=f"{fps:.0f}",
            L=f"{logs['loss_total']:.3f}",
            Lo=f"{(loss_objective_sum / num_updates) if num_updates else 0.0:.3f}",
            Lc=f"{(loss_critic_sum / num_updates) if num_updates else 0.0:.3f}",
            Le=f"{(loss_entropy_sum / num_updates) if num_updates else 0.0:.3f}",
        )

        # Optional: flush sometimes (helps if you crash mid-run)
        if i % 10 == 0:
            logger.flush()

    pbar.close()

    # Save model
    ckpt_path = save_checkpoint(
        ctx=ctx,
        actor=actor,
        critic=critic,
        meta={
            "algorithm": algorithm,
            "num_agents": num_agents,
            "seed": seed,
            "total_frames": total_frames,
            "env_name": env_name,
        },
    )
    print(f"Model saved to {ckpt_path}")

    logger.close()
    collector.shutdown()
    env.close()

    print(f"Finished MARL training on {env_name}.")
