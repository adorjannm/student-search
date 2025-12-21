"""
Neural network model definitions for multi-agent reinforcement learning.

This module provides factory functions to create policy (actor) and value
(critic) networks for the Search and Rescue multi-agent environment.

The models use:
- MultiAgentMLP: Shared parameter networks for homogeneous agents
- ProbabilisticActor: Policy network with TanhNormal distribution
- ValueOperator: Centralized value function for MAPPO

Architecture:
    - Policy: Decentralized (each agent uses local observations)
    - Critic: Centralized (uses all agent observations for value estimation)
    - Both networks share parameters across agents (homogeneous agents)
"""

from typing import Union

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, ValueOperator


class SplitLayer(nn.Module):
    """
    Helper layer to split network output into location and scale parameters.

    This layer takes a tensor with shape [..., 2, action_dim] and splits it
    into two tensors: location (mean) and scale (standard deviation). The
    scale is exponentiated to ensure it's positive.

    The input is expected to come from a network that outputs both mean and
    log-scale parameters concatenated along the second-to-last dimension.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split input tensor into location and scale.

        Args:
            x: Input tensor of shape [..., 2, action_dim] where:
                - x[..., 0, :] contains location (mean) parameters
                - x[..., 1, :] contains log-scale parameters

        Returns:
            Tuple of (location, scale) tensors:
            - location: Shape [..., action_dim], mean parameters
            - scale: Shape [..., action_dim], positive scale parameters
                (exponentiated from log-scale)
        """
        # x shape: [..., 2, action_dim]
        return x[..., 0, :], x[..., 1, :].exp()  # loc, scale (positive)


def make_policy(
    env, num_rescuers: int, device: Union[torch.device, str] = "cpu"
) -> ProbabilisticActor:
    """
    Create a probabilistic policy network (actor) for multi-agent training.

    The policy network is decentralized: each agent uses only its local
    observations to make decisions. All agents share the same network
    parameters (homogeneous agents).

    Architecture:
        - Input: Local observation for each agent [batch, n_agents, obs_dim]
        - Hidden layers: 2 layers with 64 units each, Tanh activation
        - Output: Location and scale parameters for TanhNormal distribution
        - Distribution: TanhNormal (bounded actions in [-1, 1])

    The network outputs both mean (location) and standard deviation (scale)
    parameters, allowing for stochastic policies with exploration.

    Args:
        env: TorchRL environment instance. Must have observation_spec and
            action_spec attributes with ("agents", "observation") and
            ("agents", "action") keys respectively.
        num_rescuers: Number of rescuer agents in the environment.
        device: Device to place the network on ("cpu" or "cuda").

    Returns:
        ProbabilisticActor instance that:
        - Takes observations from ("agents", "observation")
        - Outputs actions to env.action_key (typically ("agents", "action"))
        - Outputs log probabilities for PPO training
        - Uses TanhNormal distribution for bounded continuous actions

    Note:
        The policy is strictly decentralized (centralised=False) meaning
        each agent's action depends only on its own observation. This
        encourages local decision-making without explicit communication.
    """
    policy_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=env.action_spec["agents", "action"].shape[-1]
            * 2,  # Mean + Std
            n_agents=num_rescuers,
            centralised=False,  # strictly local
            share_params=True,  # Homogenous agents share weights
            device=device,
            depth=2,
            num_cells=64,
            activation_class=torch.nn.Tanh,
        ),
        # Helper to split output into mean and log_std for sampling
        nn.Unflatten(-1, (2, env.action_spec["agents", "action"].shape[-1])),
    ).to(device)

    policy_module = TensorDictModule(
        module=nn.Sequential(policy_net, SplitLayer()),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    return ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )


def make_critic(
    env, num_rescuers: int, device: Union[torch.device, str] = "cpu"
) -> ValueOperator:
    """
    Create a value network (critic) for multi-agent training.

    The value network is centralized: it uses observations from all agents
    to estimate state values. This follows the MAPPO (Multi-Agent PPO) approach
    where the critic has access to global information while the actor is
    decentralized.

    Architecture:
        - Input: All agent observations concatenated [batch, n_agents, obs_dim]
        - Hidden layers: 2 layers with 128 units each, Tanh activation
        - Output: State value estimate for each agent [batch, n_agents, 1]
        - Centralized: True (critic sees all observations)

    The centralized critic helps with credit assignment and value estimation
    in multi-agent settings, while the decentralized policy maintains
    decentralized execution.

    Args:
        env: TorchRL environment instance. Must have observation_spec with
            ("agents", "observation") key.
        num_rescuers: Number of rescuer agents in the environment.
        device: Device to place the network on ("cpu" or "cuda").

    Returns:
        ValueOperator instance that:
        - Takes observations from ("agents", "observation")
        - Outputs state values to ("agents", "state_value")
        - Uses centralized observations (all agents' observations)

    Note:
        The critic is centralized (centralised=True) which is standard for
        MAPPO. This allows better value estimation while maintaining
        decentralized policy execution.
    """

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=num_rescuers,
        centralised=True,  # MAPPO: Critic sees all
        share_params=True,
        device=device,
        depth=2,
        num_cells=128,
        activation_class=torch.nn.Tanh,
    )

    return ValueOperator(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
