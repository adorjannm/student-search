"""
Evaluation pipeline for trained multi-agent policies.

This module provides functionality to evaluate trained policies on the
Search and Rescue environment. It supports:

- Automatic model discovery (finds latest checkpoint)
- Environment configuration loading from checkpoints
- Episode-by-episode evaluation with rendering
- Metric logging to TensorBoard
- Summary statistics computation

The evaluation runs episodes with the trained policy and logs performance
metrics including rewards, episode lengths, and per-step rewards.
"""

from time import sleep

import numpy as np

from src.sar_env import make_env
from src.models import make_policy
from src.logger import RunContext, TensorboardLogger
import glob
import os
import torch
from torchrl.envs.utils import step_mdp


def find_latest_model(save_folder: str, env_name: str) -> str:
    """
    Find the latest model checkpoint file in the save folder.

    This function searches for .pt checkpoint files in the save_folder directory,
    including recursive search in timestamped subdirectories. It returns the
    file with the most recent modification time.

    Args:
        save_folder: Base directory to search for checkpoint files.
        env_name: Environment name (currently unused, kept for API compatibility).

    Returns:
        Path to the latest checkpoint file (absolute or relative path).

    Raises:
        FileNotFoundError: If no .pt files are found in save_folder or its
            subdirectories.

    Search Strategy:
        1. First searches recursively in subdirectories (e.g., save_folder/20251220-000020/*.pt)
        2. Falls back to flat search in save_folder if no files found
        3. Sorts by modification time and returns the newest file

    Note:
        The function prints the detected model path for user confirmation.
    """
    # First try recursive search in dated subdirectories
    search_pattern = os.path.join(save_folder, "**", "*.pt")
    files = glob.glob(search_pattern, recursive=True)

    # Fallback to flat files in save_folder
    if not files:
        search_pattern = os.path.join(save_folder, "*.pt")
        files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(
            f"No model files (*.pt) found in {save_folder} or its subdirectories"
        )

    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Auto-detected latest model: {latest_file}")
    return latest_file


def _get_metrics_env(env):
    base = getattr(env, "base_env", None)
    # Walk down wrappers until we find the environment exposing metrics
    while base is not None and not hasattr(base, "pop_episode_metrics"):
        base = getattr(base, "base_env", getattr(base, "_env", None))
    return base if base is not None else env


def evaluate(
    model_path: str = None,
    save_folder: str = "search_rescue_logs",
    num_games: int = 3,
    enable_logging: bool = True,
    **env_kwargs,
):
    """
    Evaluate a trained policy on the Search and Rescue environment.

    This function loads a trained policy checkpoint and evaluates it by running
    multiple episodes. It supports automatic model discovery (if model_path is None)
    and loads environment configuration from the checkpoint to ensure consistency.

    Evaluation Process:
        1. Resolve model path (auto-detect latest if not provided)
        2. Load checkpoint and extract environment configuration
        3. Create environment with saved configuration
        4. Load policy network weights
        5. Run num_games episodes with rendering (if enabled)
        6. Log metrics and compute summary statistics

    Args:
        model_path: Path to the checkpoint file (.pt). If None, automatically
            searches for the latest checkpoint in save_folder.
        save_folder: Base directory containing checkpoints and logs. Used for
            model discovery if model_path is None, and for creating evaluation logs.
        num_games: Number of episodes to run for evaluation.
        enable_logging: Whether to enable TensorBoard logging for evaluation
            metrics. If False, uses NoOpLogger.
        **env_kwargs: Environment configuration arguments. These are used if
            the checkpoint doesn't contain env_config. If env_config exists
            in the checkpoint, these are overridden (except render_mode which
            is always taken from env_kwargs). Common arguments:
            - num_rescuers: Number of rescuer agents
            - num_victims: Number of victim entities
            - num_trees: Number of obstacle trees
            - num_safe_zones: Number of safe zones
            - max_cycles: Maximum steps per episode
            - vision_radius: Vision/observation radius
            - render_mode: Rendering mode ("human" for visualization, None for headless)
            - continuous_actions: Whether to use continuous actions

    Evaluation Metrics:
        The function logs the following metrics to TensorBoard:
        - eval/episode_{i}: Per-step reward for episode i
        - eval/episode_reward: Total reward for each episode
        - eval/episode_steps: Number of steps for each episode
        - eval/mean_reward_per_step: Average reward per step for each episode
        - eval/mean_episode_reward: Mean reward across all episodes
        - eval/mean_episode_steps: Mean episode length across all episodes
        - eval/total_episodes: Total number of episodes evaluated

    Checkpoint Format:
        The function supports multiple checkpoint formats:
        1. Full checkpoint dict with "policy_state_dict" key (preferred)
        2. Checkpoint dict with "actor" key (legacy format)
        3. Raw state dict (fallback)

    Note:
        - The policy is set to evaluation mode (no gradient computation)
        - Rendering introduces a 0.1s delay per step for visualization
        - Environment configuration from checkpoint takes precedence over
          env_kwargs (except render_mode)
        - The function automatically handles different TensorDict structures
          returned by different TorchRL wrapper versions

    Example:
        >>> evaluate(
        ...     model_path="checkpoints/model.pt",
        ...     num_games=10,
        ...     render_mode="human",
        ...     enable_logging=True
        ... )

        >>> evaluate(
        ...     save_folder="logs/",
        ...     num_games=5,
        ...     render_mode=None
        ... )  # Auto-detects latest model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup logging
    run_ctx = RunContext(base_dir=save_folder, run_name="eval", create_subdirs=True)
    logger = TensorboardLogger.create(ctx=run_ctx, enabled=enable_logging)
    if enable_logging:
        print(f"TensorBoard logging enabled. Log directory: {run_ctx.tb_run_dir}")

    # 1. Resolve Model Path first (we need to load config from checkpoint)
    if not model_path:
        print(f"No model path provided. Searching in '{save_folder}'...")
        # Create temporary env to get metadata name
        temp_env = make_env(device=device, **env_kwargs)
        try:
            env_name = temp_env.base_env.metadata["name"]
            model_path = find_latest_model(save_folder, env_name)
        except Exception as e:
            print(f"Error finding model: {e}")
            return
        finally:
            temp_env.close()

    # 2. Load Model and Config
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 3. Use environment config from checkpoint if available
    if "env_config" in checkpoint:
        print("Using environment configuration from checkpoint:")
        saved_config = checkpoint["env_config"]
        for key, value in saved_config.items():
            print(f"  {key}: {value}")

        # Override env_kwargs with saved config (but keep render_mode from args)
        render_mode = env_kwargs.get("render_mode", "human")
        env_kwargs = saved_config.copy()
        env_kwargs["render_mode"] = render_mode
    else:
        print("Warning: No env_config in checkpoint, using provided arguments")

    # 4. Create Environment with correct configuration
    print("Initializing environment...")
    env = make_env(device=device, **env_kwargs)

    # 5. Create and Load Policy
    num_agents = env.action_spec["agents", "action"].shape[0]

    # Determine if we're using discrete or continuous actions
    is_discrete = not env.base_env.is_continuous
    print(f"Action type: {'Discrete' if is_discrete else 'Continuous'}")

    policy = make_policy(
        env, num_rescuers=num_agents, device=device, discrete=is_discrete
    )

    # Handle different saving formats
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        # Format from the robust training script
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(
            f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'N/A')} "
            f"(total frames: {checkpoint.get('total_frames', 'N/A')})"
        )
    elif isinstance(checkpoint, dict) and "actor" in checkpoint:
        # Format from older simple script
        policy.load_state_dict(checkpoint["actor"])
    else:
        # Raw state dict
        policy.load_state_dict(checkpoint)

    policy.eval()  # Set to evaluation mode

    # 6. Evaluation Loop
    print(f"Starting evaluation for {num_games} episodes...")

    # Track evaluation metrics
    episode_rewards = []
    episode_steps = []
    rescues_pct_log = []
    collisions_log = []
    coverage_log = []
    metrics_env = _get_metrics_env(env)

    for i in range(num_games):
        td = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0

        print(f"--- Episode {i + 1} ---")

        while not done:
            with torch.no_grad():
                td = policy(td)

            td = env.step(td)
            env.render()

            if "next" in td.keys():
                # Standard TorchRL behavior
                if td["next", "done"].any():
                    done = True
                td = step_mdp(td)
            else:
                # Flat behavior (PettingZooWrapper sometimes does this)
                # The 'td' returned IS the next state
                # Check for "done", "terminated", or "agents/done"
                if "done" in td.keys() and td["done"].any():
                    done = True
                elif "terminated" in td.keys() and td["terminated"].any():
                    done = True
                elif ("agents", "done") in td.keys(include_nested=True) and td[
                    "agents", "done"
                ].any():
                    done = True

            step_count += 1

            # Collect rewards for logging
            if ("agents", "reward") in td.keys(include_nested=True):
                rewards = td["agents", "reward"].detach().cpu().numpy()
                logger.log_scalar(
                    f"eval/episode_{i+1}", rewards.mean(), step=step_count
                )
                episode_reward += float(rewards.sum())

            sleep(0.1)

        # Log episode metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)

        logger.log_scalar("eval/episode_reward", episode_reward, step=i + 1)
        logger.log_scalar("eval/episode_steps", step_count, step=i + 1)
        logger.log_scalar(
            "eval/mean_reward_per_step", episode_reward / max(step_count, 1), step=i + 1
        )

        # Log environment metrics (rescues %, collisions, coverage)
        metrics = metrics_env.pop_episode_metrics()
        if metrics:
            m = metrics[-1]
            rescues_pct_log.append(m["rescues_pct"])
            collisions_log.append(m["collisions"])
            coverage_log.append(m["coverage_cells"])
            logger.log_scalar("eval/rescues_pct", m["rescues_pct"], step=i + 1)
            logger.log_scalar("eval/collisions", m["collisions"], step=i + 1)
            logger.log_scalar("eval/coverage_cells", m["coverage_cells"], step=i + 1)

        print(
            f"Episode {i + 1} finished in {step_count} steps. Total reward: {episode_reward:.2f}"
        )

    mean_reward = 0
    mean_steps = 0

    # Log summary statistics
    if episode_rewards:
        mean_reward = float(np.mean(episode_rewards))
        mean_steps = float(np.mean(episode_steps))
        logger.log_scalar("eval/mean_episode_reward", mean_reward, step=num_games)
        logger.log_scalar("eval/mean_episode_steps", mean_steps, step=num_games)
        logger.log_scalar("eval/total_episodes", num_games, step=num_games)

    if rescues_pct_log:
        mean_rescues_pct = float(np.mean(rescues_pct_log))
        mean_collisions = float(np.mean(collisions_log))
        mean_coverage = float(np.mean(coverage_log))
        logger.log_dict(
            "eval/summary",
            {
                "rescues_pct": mean_rescues_pct,
                "collisions": mean_collisions,
                "coverage_cells": mean_coverage,
            },
            step=num_games,
        )

    logger.close()
    print("Evaluation finished.")
    if episode_rewards:
        print(f"Mean episode reward: {mean_reward:.2f}")
        print(f"Mean episode steps: {mean_steps:.1f}")
    if enable_logging:
        print(f"TensorBoard logs available at: {run_ctx.tb_run_dir}")
    env.close()
