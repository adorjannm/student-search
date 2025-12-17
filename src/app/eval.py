"""
Evaluation script for the Search and Rescue environment.
Supports visualization and testing trained MARL models.

CTDE evaluation: Uses trained actors that rely only on local observations
(decentralized execution) while the policy was trained with centralized critics.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from src.domain.sar_env import SearchAndRescueEnv
from src.infra.checkpointing import find_latest_checkpoint, load_checkpoint
from src.infra.config import make_run_context
from src.infra.logging_tb import TensorboardLogger
from src.metrics.metrics import (
    EpisodeTracker,
    aggregate_logs,
    compute_summary,
    plot_core_metrics,
)
from src.rl.models import make_actor


def _random_action(env, device: str):
    """
    Robust random action sampler.
    Works for both discrete and continuous action specs (TorchRL style).
    """
    try:
        a = env.action_spec.rand()
        if isinstance(a, torch.Tensor):
            return a.to(device)
        return a
    except (AttributeError, TypeError, RuntimeError, ValueError):
        a = torch.tensor(env.sample_discrete_action(), dtype=torch.int64, device=device)
        return a


def evaluate(
    num_games: int = 100,
    save_folder: str = "search_rescue_logs/",
    render_mode=None,
    tb_enabled: bool = True,
    seed: int = 0,
    algorithm: str = "eval",
    **env_kwargs,
):
    """Evaluate trained MARL agents with visualization support.

    CTDE evaluation: The policy trained with a centralized critic
    executes using only local observations (decentralized execution).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment with rendering
    env_kwargs["seed"] = seed
    env = SearchAndRescueEnv(render_mode=render_mode, **env_kwargs)
    env_name = getattr(env, "metadata", {}).get("name", type(env).__name__)
    num_agents = getattr(env, "base_env", env).num_rescuers
    victims_total = len(env.victims)

    ctx = make_run_context(
        save_folder=save_folder,
        env_name=env_name,
        algorithm=algorithm,
        seed=seed,
    )
    logger = TensorboardLogger.create(ctx, enabled=tb_enabled)

    print(
        f"\nStarting MARL evaluation on {env_name} "
        f"(num_games={num_games}, render_mode={render_mode}, num_agents={num_agents})"
    )

    # Create actor for loading weights (no log prob needed for eval)
    actor = make_actor(env, device=device, return_log_prob=False)
    if actor is None:
        raise RuntimeError(
            "make_actor returned None; cannot load checkpoint into actor"
        )

    # checkpoint selection (single prompt)
    manual = input(
        "Enter checkpoint path (or press Enter to load the latest from save_folder/checkpoints): "
    ).strip()

    ckpt_path: Path | None = (
        Path(manual)
        if manual
        else find_latest_checkpoint(save_folder=save_folder, env_name=None)
    )

    # load checkpoint or fall back to random
    if ckpt_path is None or not ckpt_path.exists():
        print("No checkpoint found. Running with random actions.")
        logger.log_text("eval/checkpoint", "None (random policy)", step=0)
        actor = None
    else:
        try:
            ckpt = load_checkpoint(ckpt_path, device=device)
            actor.load_state_dict(ckpt.actor_state)
            print(f"Loaded checkpoint: {ckpt_path}")
            logger.log_text("eval/checkpoint", str(ckpt_path), step=0)

            if ckpt.meta:
                safe_meta = {k: str(v) for k, v in ckpt.meta.items()}
                logger.log_text(
                    "eval/checkpoint_meta",
                    "\n".join(f"{k}={v}" for k, v in safe_meta.items()),
                    step=0,
                )
        except Exception as e:
            print(f"Error loading checkpoint ({ckpt_path}): {e}")
            print("Running with random actions instead.")
            logger.log_text("eval/checkpoint", f"Failed to load: {ckpt_path}", step=0)
            actor = None

    # Evaluation loop
    episode_logs = []

    for game in range(num_games):
        tracker = EpisodeTracker(game + 1)
        td = env.reset()

        while True:
            # Render
            if render_mode == "human":
                env.render()
                time.sleep(0.05)  # Slow down for visualization

            if actor is not None:
                # Use trained policy (decentralized execution)
                with torch.no_grad():
                    td = actor(td)  # writes td["action"]
            else:
                td["action"] = _random_action(env, device=device)

            # Step environment
            td = env.step(td)

            # Extract results (shared team reward in cooperative MARL)
            reward = td["next", "reward"].item()
            done = td["next", "done"].item()

            tracker.record(env, reward)

            if done:
                break

            # Update td for next iteration
            td = td["next"].clone()

        # Count rescued victims
        log = tracker.finalize(env)
        episode_logs.append(log)

        print(
            f"Game {game + 1}/{num_games}: "
            f"Reward={log.rewards:.2f}, "
            f"Rescued={log.rescues}/{victims_total}"
        )

    df = aggregate_logs(episode_logs)
    summary = compute_summary(df, victims_total=victims_total)

    # Statistics
    avg_reward = float(df["rewards"].mean()) if not df.empty else 0.0
    avg_rescues = float(df["rescues"].mean()) if not df.empty else 0.0

    plots_dir = ctx.plots_root / ctx.run_id
    plots = plot_core_metrics(df, str(plots_dir))

    print("\n" + "=" * 50)
    print("MARL Evaluation Results:")
    print(f"  Number of agents: {num_agents}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average rescues: {avg_rescues:.2f}/{victims_total}")
    print(f"  Rescues completed: {summary['rescues_pct']:.1f}%")
    print(f"  Average collisions: {summary['avg_collisions']:.2f}")
    print(f"  Average coverage cells: {summary['avg_coverage_cells']:.2f}")
    print(
        f"  Avg time to first rescue: {summary['avg_time_to_first_rescue']:.2f} steps"
    )
    print("  Plots saved:")
    for label, path in plots.items():
        print(f"    {label}: {path}")
    print(f"  Total games: {num_games}")
    print("=" * 50)

    # --- TensorBoard logging ---
    logger.log_dict(
        "eval",
        {
            "avg_reward": avg_reward,
            "avg_rescues": avg_rescues,
            **summary,
        },
        step=0,
    )
    logger.log_text("eval/plots_dir", str(plots_dir), step=0)
    logger.flush()
    logger.close()

    env.close()
    return avg_reward


def visualize_random(env_kwargs, num_steps=500):
    """Run visualization with random actions for testing."""
    env = SearchAndRescueEnv(render_mode="human", **env_kwargs)
    num_agents = env.num_rescuers

    print(
        f"Running random visualization for {num_steps} steps with {num_agents} agents..."
    )

    td = env.reset()

    for step in range(num_steps):
        env.render()
        time.sleep(0.05)

        # Random action
        action = torch.tensor(
            env.sample_discrete_action(),
            dtype=torch.int64,
        )
        td["action"] = action

        td = env.step(td)

        if td["next", "done"].item():
            print(f"Episode finished at step {step}")
            td = env.reset()

    env.close()
