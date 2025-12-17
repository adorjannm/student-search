from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from src.infra.config import RunContext, to_abs_path


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    actor_state: dict[str, Any]
    critic_state: dict[str, Any]
    meta: dict[str, Any]


def save_checkpoint(
    ctx: RunContext,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save a checkpoint under ctx.ckpt_root with the ctx.run_id name.
    """
    ctx.ensure_dirs()
    path = ctx.ckpt_root / f"{ctx.run_id}.pt"

    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "meta": meta or {},
        "run_id": ctx.run_id,
    }
    torch.save(payload, str(path))
    return path


def find_latest_checkpoint(
    save_folder: str,
    env_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Find the latest checkpoint under <save_folder>/checkpoints.

    If env_name is provided, only match files starting with env_name.
    """
    root = to_abs_path(save_folder)
    ckpt_root = root / "checkpoints"
    if not ckpt_root.exists():
        return None

    pattern = f"{env_name}*.pt" if env_name else "*.pt"
    candidates = list(ckpt_root.glob(pattern))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_checkpoint(path: Path, device: str = "cpu") -> Checkpoint:
    """
    Load a checkpoint from disk.
    """
    payload = torch.load(str(path), map_location=device)
    return Checkpoint(
        path=path,
        actor_state=payload["actor"],
        critic_state=payload["critic"],
        meta=payload.get("meta", {}),
    )
