"""
Search and Rescue Multi-Agent Environment.

This module implements a multi-agent reinforcement learning environment for
search and rescue operations. Rescuers (cooperative agents) must guide victims
to their designated safe zones while navigating around obstacles (trees).

The environment features:
- Multi-agent coordination: Multiple rescuers work together to rescue victims
- Victim commitment system: Victims commit to rescuers when approached
- Vision and occlusion: Limited vision radius with tree occlusion
- Type-based clustering: Victims must be taken to matching safe zone types
- Physics simulation: Velocity-based movement with collision handling

The environment follows the PettingZoo ParallelEnv API and is compatible with
TorchRL for training multi-agent reinforcement learning policies.
"""

from typing import Optional, Union

import numpy as np
import pygame
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from torchrl.envs import PettingZooWrapper, TransformedEnv, RewardSum

from src.seed_utils import set_seed


class SearchAndRescueEnv(ParallelEnv):
    """
    Multi-agent Search and Rescue environment.

    This environment simulates a search-and-rescue operation where rescuers
    (cooperative agents) must guide victims to designated safe zones. The
    environment includes obstacles (trees) that block vision and movement,
    and implements a commitment system where victims follow rescuers when
    approached.

    Environment Dynamics:
        - Rescuers move using continuous acceleration actions
        - Victims commit to rescuers when within follow_radius
        - Victims are saved when they reach their matching safe zone
        - Trees block vision and cause collisions
        - Safe zones are located at the four corners of the map

    Observation Space:
        For each rescuer agent, the observation includes:
        - Self velocity (2D)
        - Self position (2D)
        - Agent ID one-hot encoding (num_rescuers)
        - Safe zone relative positions and types (num_safe_zones * 3)
        - Tree relative positions (num_trees * 2, masked if not visible)
        - Victim relative positions and types (num_victims * 3, masked if not visible)

    Action Space:
        Continuous 2D acceleration vector in range [-1, 1] for each rescuer.

    Reward Structure:
        - +100.0: Successfully rescuing a victim (to assigned rescuer)
        - +10.0: Assisting in a rescue (to nearby rescuers)
        - +1.0 / (distance_to_zone + eps): Escorting victim toward safe zone
        - +0.1 * delta_distance: Distance shaping (getting closer to victims)
        - -1.0: Tree collision
        - -1.0: Boundary violation
        - -5.0: Agent-agent collision

    Termination Conditions:
        - All victims are saved (success)
        - Maximum number of steps reached (truncation)

    Attributes:
        num_rescuers: Number of rescuer agents in the environment
        num_victims: Number of victim entities to rescue
        num_trees: Number of obstacle trees in the environment
        num_safe_zones: Number of safe zones (always 4, at corners)
        max_steps: Maximum number of steps per episode
        is_continuous: Whether actions are continuous (always True)
        world_size: Size of the world (2.0, corresponding to [-1, 1] range)
        vision_radius: Maximum distance for vision and observation
        rescue_radius: Distance threshold for rescuing victims
        agent_size: Radius of agent entities
        tree_radius: Radius of tree obstacles
        safe_zone_radius: Radius of safe zones
        follow_radius: Distance threshold for victim commitment
        agents: List of active agent names
        possible_agents: List of all possible agent names
        victim_names: List of victim entity names
        victim_types: List of victim type indices (0-3, cyclically assigned)
        safe_zone_types: List of safe zone type indices (0-3, one per corner)
        type_colors: Mapping from type index to RGB color tuple
        action_spaces: Dictionary mapping agent names to action spaces
        observation_spaces: Dictionary mapping agent names to observation spaces
        rescuer_pos: Array of rescuer positions [num_rescuers, 2]
        rescuer_vel: Array of rescuer velocities [num_rescuers, 2]
        victim_pos: Array of victim positions [num_victims, 2]
        victim_vel: Array of victim velocities [num_victims, 2]
        victim_saved: Boolean array indicating which victims are saved
        victim_assignments: Array mapping victims to assigned rescuer indices (-1 if none)
        tree_pos: Array of tree positions [num_trees, 2]
        safezone_pos: Array of safe zone positions [4, 2] (corners)
        prev_agent_victim_dists: Previous distances for reward shaping
        steps: Current step count in the episode
        screen: Pygame screen surface for rendering (None if not rendering)
        clock: Pygame clock for frame rate control (None if not rendering)
        font: Pygame font for rendering text labels (None if not rendering)

    Examples:
        Basic usage:
            >>> env = SearchAndRescueEnv(num_rescuers=2, num_victims=4)
            >>> obs, info = env.reset()
            >>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            >>> obs, rewards, dones, truncs, infos = env.step(actions)

        With rendering:
            >>> env = SearchAndRescueEnv(render_mode="human")
            >>> obs, info = env.reset()
            >>> env.render()  # Display the environment
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "search_rescue_v2"}

    def __init__(
        self,
        num_rescuers: int = 2,
        num_victims: int = 2,
        num_trees: int = 5,
        num_safe_zones: int = 4,
        max_cycles: int = 200,
        continuous_actions: bool = True,
        vision_radius: float = 0.5,
        randomize_safe_zones: bool = False,
        render_mode: Optional[str] = None,
        max_trees: Optional[
            int
        ] = None,  # For curriculum learning - fixes obs space size
    ):
        """
        Initialize the Search and Rescue environment.

        Args:
            num_rescuers: Number of rescuer agents. Must be >= 1.
            num_victims: Number of victim entities to rescue. Must be >= 1.
            num_trees: Number of obstacle trees. Must be >= 0.
            num_safe_zones: Number of safe zones. Should be 4 (corners), but
                can be configured. Each safe zone has a unique type (0-3).
            max_cycles: Maximum number of steps per episode before truncation.
            continuous_actions: Whether to use continuous actions.
            vision_radius: Maximum distance at which agents can observe entities.
                Entities beyond this radius or occluded by trees are masked.
            seed: Random seed for environment initialization. If None, uses
                system random seed.
            render_mode: Rendering mode. Options:
                - None: No rendering
                - "human": Render to a window using pygame
                - "rgb_array": Return RGB array (not implemented)

        Note:
            Victim types are assigned cyclically (victim i gets type i % 4).
            Safe zones are always positioned at the four corners:
            - Type 0: Top-left (-0.9, 0.9)
            - Type 1: Top-right (0.9, 0.9)
            - Type 2: Bottom-left (-0.9, -0.9)
            - Type 3: Bottom-right (0.9, -0.9)
        """
        self.num_rescuers = num_rescuers
        self.num_victims = num_victims
        self.num_trees = num_trees
        self.num_safe_zones = num_safe_zones
        self.render_mode = render_mode
        self.max_steps = max_cycles
        self.is_continuous = continuous_actions
        self.randomize_safe_zones = randomize_safe_zones

        # For curriculum learning: use max_trees for obs space size, pad observations
        self.max_trees = max_trees if max_trees is not None else num_trees

        # Parameters
        self.world_size = 2.0  # [-1, 1] range
        self.vision_radius = vision_radius
        self.rescue_radius = 0.15
        self.agent_size = 0.03
        self.tree_radius = 0.05
        self.safe_zone_radius = 0.15
        self.follow_radius = 0.2

        self.agents = [f"rescuer_{i}" for i in range(num_rescuers)]
        self.possible_agents = self.agents[:]
        self.victim_names = [f"victim_{i}" for i in range(num_victims)]

        # Colors for rendering
        self.type_colors = {
            0: (255, 50, 50),  # Red (A)
            1: (50, 255, 50),  # Green (B)
            2: (50, 50, 255),  # Blue (C)
            3: (255, 255, 50),  # Yellow (D)
        }

        # Action Space: Either Discrete or Continuous
        if self.is_continuous:
            # Continuous acceleration (dx, dy)
            self.action_spaces = {
                agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                for agent in self.agents
            }
        else:
            # Discrete actions: 0=noop, 1=up, 2=down, 3=left, 4=right
            self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}

        # Observation Space Calculation (Partial Observability with Vision Masking)
        # Layout per agent:
        # [self_vel(2), self_pos(2), agent_id(num_rescuers),
        #  safe_zones(max_safe_zones * 3: rel_x, rel_y, type_idx or -1 if masked),
        #  trees(max_trees * 3: rel_x, rel_y, visible_bit),
        #  victims(num_victims * 3: rel_x, rel_y, type_idx or -1 if masked)]
        self.max_safe_zones = self.num_safe_zones  # kept for clarity

        self.obs_dim = (
            2  # self_vel
            + 2  # self_pos
            + self.num_rescuers  # agent_id one-hot
            + (self.max_safe_zones * 3)
            + (self.max_trees * 3)
            + (self.num_victims * 3)
        )

        # Build bounded observation space
        rel_low, rel_high = -3.0, 3.0  # generous bound for relative deltas
        obs_low = []
        obs_high = []

        # Self velocity (2)
        obs_low.extend([-0.1, -0.1])
        obs_high.extend([0.1, 0.1])

        # Self position (2)
        obs_low.extend([-1.0, -1.0])
        obs_high.extend([1.0, 1.0])

        # Agent ID one-hot
        obs_low.extend([0.0] * self.num_rescuers)
        obs_high.extend([1.0] * self.num_rescuers)

        # Safe zones: rel_x, rel_y, type (or -1 when masked)
        for _ in range(self.max_safe_zones):
            obs_low.extend([rel_low, rel_low, -1.0])
            obs_high.extend([rel_high, rel_high, float(self.num_safe_zones - 1)])

        # Trees: rel_x, rel_y (masked to 0), visible bit
        for _ in range(self.max_trees):
            obs_low.extend([rel_low, rel_low, 0.0])
            obs_high.extend([rel_high, rel_high, 1.0])

        # Victims: rel_x, rel_y, type (or -1 when masked)
        for _ in range(self.num_victims):
            obs_low.extend([rel_low, rel_low, -1.0])
            obs_high.extend([rel_high, rel_high, float(self.num_safe_zones - 1)])

        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array(obs_low, dtype=np.float32),
                high=np.array(obs_high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        self.screen = None
        self.clock = None

        # Metrics state (reset each episode)
        self._cell_size = 0.05
        self._episode_counter = 0
        self._visited_cells = []
        self._collision_events = 0
        self._completed_episode_metrics = []

    def _reset_metrics(self) -> None:
        """Reset per-episode metric trackers."""
        self._visited_cells = [set() for _ in range(self.num_rescuers)]
        self._collision_events = 0

    def _hash_pos(self, pos: np.ndarray) -> tuple[int, int]:
        return tuple(np.floor(pos / self._cell_size).astype(int))

    def pop_episode_metrics(self) -> list[dict]:
        """Return and clear metrics for episodes that completed since last call."""
        metrics = self._completed_episode_metrics
        self._completed_episode_metrics = []
        return metrics

    def set_num_trees(self, num_trees: int) -> None:
        """
        Update the number of trees (occluders) in the environment.

        This is useful for curriculum learning where difficulty increases
        by adding more occluders over time. Changes take effect on next reset().

        Note: This only updates the target tree count. The actual tree positions
        are created during reset(), so changes won't take effect until the
        current episode ends and a new one begins.

        Args:
            num_trees: New number of trees to use
        """
        # Store the target tree count
        # We don't update obs_dim or observation_spaces here because that would
        # cause dimension mismatch errors during the current episode.
        # Instead, we wait for the next reset() to apply the changes.
        self._pending_num_trees = num_trees

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.

        This method initializes all entities (rescuers, victims, trees, safe zones)
        to random positions within the world bounds. All velocities are set to zero,
        and victim states are reset (not saved, no assignments).

        Args:
            seed: Random seed for reproducible initialization. If provided,
                sets numpy random seed. If None, uses current random state.
            options: Optional dictionary of reset options (currently unused).

        Returns:
            A tuple containing:
            - observations: Dictionary mapping agent names to observation arrays.
                Each observation is a numpy array of shape (obs_dim,).
            - infos: Dictionary mapping agent names to info dictionaries.
                Currently empty for all agents.

        Note:
            The agents list is reset to include all possible agents. This is
            required by the PettingZoo API to handle agent removal on termination.
        """
        self.steps = 0

        if seed is not None:
            set_seed(seed)

        # Apply pending tree count update (from curriculum learning)
        # Note: We don't change obs_dim since it's fixed to max_trees
        if hasattr(self, "_pending_num_trees"):
            self.num_trees = self._pending_num_trees
            delattr(self, "_pending_num_trees")

        # Reset Agents list (required by PettingZoo API)
        self.agents = self.possible_agents[:]

        # Metrics state (reset each episode)
        self._episode_counter += 1
        self._reset_metrics()

        # Positions: Rescuers, Victims, Trees, SafeZones (using global np.random)
        self.rescuer_pos = np.random.uniform(-0.8, 0.8, (self.num_rescuers, 2))
        self.rescuer_vel = np.zeros((self.num_rescuers, 2))

        self.victim_pos = np.random.uniform(-0.8, 0.8, (self.num_victims, 2))
        self.victim_vel = np.zeros((self.num_victims, 2))
        self.victim_saved = np.zeros(self.num_victims, dtype=bool)

        self.victim_types = np.random.randint(
            0, self.num_safe_zones, size=self.num_victims
        )

        # Track which agent each victim is committed to (-1 = none)
        self.victim_assignments = np.full(self.num_victims, -1, dtype=int)

        self.tree_pos = np.random.uniform(-0.8, 0.8, (self.num_trees, 2))

        # Safe zones: randomized or at fixed corners
        if self.randomize_safe_zones:
            # Randomize positions within the world bounds
            self.safezone_pos = np.random.uniform(-0.95, 0.95, (self.num_safe_zones, 2))

            # Shuffle types to randomize which zone accepts which victim type
            # Keep types as 0,1,2,3 but in random order
            self.safe_zone_types = list(range(self.num_safe_zones))
            np.random.shuffle(self.safe_zone_types)
        else:
            # Fixed positions at corners (default behavior)
            self.safezone_pos = np.array(
                [[-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]]
            )
            # Types stay in original order
            self.safe_zone_types = [0, 1, 2, 3]

        # Track previous distances for delta-based shaping
        self.prev_agent_victim_dists = self._compute_agent_victim_dists()

        return self._get_obs(), {a: {} for a in self.agents}

    def _is_visible(
        self, observer_pos, target_pos, target_radius, exclude_tree_idx=None
    ):
        """Checks whether a target is within vision range and not occluded by trees.
        Args:
            observer_pos: Position of the observer.
            target_pos: Position of the target to check visibility of.
            target_radius: Radius of the target. Currently unused, kept for API compatibility
                and potential future use in more detailed visibility calculations.
            exclude_tree_idx: Optional tree index to exclude from the occlusion check
                (e.g., when checking if a tree itself is visible). Occlusion is currently
                determined using the environment's ``tree_radius`` for all trees.
        """
        # If vision radius is zero, nothing is visible
        if self.vision_radius == 0.0:
            return False

        dist = np.linalg.norm(target_pos - observer_pos)
        if dist > self.vision_radius:
            return False

        # Check line of sight against all trees (excluding the tree being checked if specified)
        for t_idx in range(self.num_trees):
            if exclude_tree_idx is not None and t_idx == exclude_tree_idx:
                continue  # Skip the tree being checked

            tree_c = self.tree_pos[t_idx]

            # Skip trees that are at the target position (target tree should not block itself)
            tree_to_target_dist = np.linalg.norm(tree_c - target_pos)
            if tree_to_target_dist < 1e-6:
                continue  # Tree is at target position, skip it

            # Vector from observer to target
            d_vec = target_pos - observer_pos
            # Vector from observer to tree center
            f_vec = observer_pos - tree_c

            a = np.dot(d_vec, d_vec)
            # Handle edge case: observer and target at same position
            if a < 1e-10:
                # If observer and target are at same position, check if tree is at that position
                tree_dist = np.linalg.norm(tree_c - observer_pos)
                if tree_dist < self.tree_radius:
                    return False  # Tree is blocking (at same position)
                continue  # No blocking if tree is not at same position

            b = 2 * np.dot(f_vec, d_vec)
            c = np.dot(f_vec, f_vec) - self.tree_radius**2

            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                discriminant = np.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a)
                t2 = (-b + discriminant) / (2 * a)

                if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                    return False  # Blocked
        return True

    def _get_obs(self) -> dict:
        """
          Compute observations for all active agents.

          Observations are computed relative to each agent's position and include:
          - Self state (velocity, position)
          - Agent ID (one-hot encoding for symmetry breaking)
          - Safe zones (global knowledge, always visible)
          - Trees (masked if not visible or occluded)
          - Victims (masked if not visible, occluded, or already saved)

        Masking is done by setting tree relative positions to [0.0, 0.0] and
          victim relative positions and types to [0.0, 0.0, -1.0] (indicating not visible).

          Returns:
              Dictionary mapping agent names to observation arrays.
              Each observation is a numpy array of shape (obs_dim,) containing:
              - [0:2]: Self velocity (vx, vy)
              - [2:4]: Self position (x, y)
              - [4:4+num_rescuers]: Agent ID one-hot encoding
              - [4+num_rescuers:4+num_rescuers+num_safe_zones*3]: Safe zones
                  (relative_x, relative_y, type) for each safe zone
              - [safe_zones_end:safe_zones_end+num_trees*2]: Trees
                  (relative_x, relative_y) for each tree (masked if not visible)
              - [trees_end:trees_end+num_victims*3]: Victims
                  (relative_x, relative_y, type) for each victim
                  (masked as [0, 0, -1] if not visible or saved)

          Note:
              Other rescuers are intentionally excluded from observations to
              focus learning on the rescue task rather than agent coordination
              through explicit observation.
        """
        observations = {}
        for i, agent in enumerate(self.agents):
            # If agent was removed (e.g. done), skip
            if agent not in self.agents:
                continue

            obs_vec = []
            my_pos = self.rescuer_pos[i]

            # 1. Self State (Vel, Pos)
            obs_vec.extend(self.rescuer_vel[i])
            obs_vec.extend(my_pos)

            # 2. Agent ID (one-hot encoding to break symmetry)
            agent_id_onehot = np.zeros(self.num_rescuers)
            agent_id_onehot[i] = 1.0
            obs_vec.extend(agent_id_onehot)

            # 3. Safe Zones (always visible landmarks)
            for sz_i in range(self.num_safe_zones):
                rel_pos = self.safezone_pos[sz_i] - my_pos
                obs_vec.extend(
                    [rel_pos[0], rel_pos[1], float(self.safe_zone_types[sz_i])]
                )

            # 4. Trees (partial observability): mask rel_pos when not visible
            for t_i in range(self.max_trees):
                if t_i < self.num_trees and self._is_visible(
                    my_pos, self.tree_pos[t_i], self.tree_radius, exclude_tree_idx=t_i
                ):
                    rel_pos = self.tree_pos[t_i] - my_pos
                    obs_vec.extend([rel_pos[0], rel_pos[1], 1.0])
                else:
                    obs_vec.extend([0.0, 0.0, 0.0])  # unused slot

            # 5. Victims
            for v_i in range(self.num_victims):
                # If visible and not saved
                if not self.victim_saved[v_i] and self._is_visible(
                    my_pos, self.victim_pos[v_i], self.agent_size
                ):
                    rel = self.victim_pos[v_i] - my_pos
                    # Use numeric type (0-3) for observation
                    obs_vec.extend([rel[0], rel[1], float(self.victim_types[v_i])])
                else:
                    obs_vec.extend(
                        [0.0, 0.0, -1.0]
                    )  # Masked (Type -1 indicates not visible)

            # Other rescuers removed from observation to focus learning on task

            observations[agent] = np.array(obs_vec, dtype=np.float32)
        return observations

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        """
        Execute one environment step.

        This method processes actions for all agents, updates physics (movement,
        collisions), handles victim dynamics (commitment and following), computes
        rewards, and checks termination conditions.

        Step Process:
        1. Apply rescuer actions (acceleration updates)
        2. Handle wall collisions (reflection with damping)
        3. Handle tree collisions (reflection with damping, penalty)
        4. Handle agent-agent collisions (soft repulsion)
        5. Update victim dynamics (commitment logic, following behavior)
        6. Compute rewards (rescues, distance shaping, penalties)
        7. Check termination conditions

        Args:
            actions: Dictionary mapping agent names to action arrays.
                Each action is a 2D numpy array representing acceleration
                in range [-1, 1] for (x, y) directions.

        Returns:
            A tuple containing:
            - observations: Dictionary mapping agent names to observation arrays.
            - rewards: Dictionary mapping agent names to reward values (float).
            - terminations: Dictionary mapping agent names to termination flags (bool).
                True if all victims are saved (success).
            - truncations: Dictionary mapping agent names to truncation flags (bool).
                True if max_steps reached.
            - infos: Dictionary mapping agent names to info dictionaries.
                Currently empty for all agents.

        Note:
            On termination or truncation, the agents list is emptied to comply
            with PettingZoo API requirements. This prevents further steps.
        """
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # 1. Apply Actions
        for i, agent in enumerate(self.agents):
            if agent not in actions:
                continue

            action = actions[agent]

            # Convert discrete action to continuous if needed
            if not self.is_continuous:
                # Discrete actions: 0=noop, 1=up, 2=down, 3=left, 4=right
                if isinstance(action, np.ndarray):
                    action = int(action)

                action_map = {
                    0: np.array([0.0, 0.0]),  # noop
                    1: np.array([0.0, 1.0]),  # up
                    2: np.array([0.0, -1.0]),  # down
                    3: np.array([-1.0, 0.0]),  # left
                    4: np.array([1.0, 0.0]),  # right
                }
                action = action_map.get(action, np.array([0.0, 0.0]))

            # Physics
            self.rescuer_vel[i] = self.rescuer_vel[i] * 0.8 + action * 0.1
            speed = np.linalg.norm(self.rescuer_vel[i])
            if speed > 0.08:
                self.rescuer_vel[i] = (self.rescuer_vel[i] / speed) * 0.08

            # Inward push if we’re hugging the boundary to avoid sticking
            margin = 0.85
            inward_k = 0.02
            if self.rescuer_pos[i][0] > margin:
                self.rescuer_vel[i][0] -= inward_k
            elif self.rescuer_pos[i][0] < -margin:
                self.rescuer_vel[i][0] += inward_k
            if self.rescuer_pos[i][1] > margin:
                self.rescuer_vel[i][1] -= inward_k
            elif self.rescuer_pos[i][1] < -margin:
                self.rescuer_vel[i][1] += inward_k

            self.rescuer_pos[i] += self.rescuer_vel[i]

            # --- Wall collision handling (reflect and damp) ---
            # If we cross the world bounds, clamp position to boundary and
            # reflect the corresponding velocity component with damping.
            for axis in range(2):
                if self.rescuer_pos[i][axis] > 1.0:
                    self.rescuer_pos[i][axis] = 1.0
                    # invert normal component (axis) and damp
                    self.rescuer_vel[i][axis] *= -0.5
                elif self.rescuer_pos[i][axis] < -1.0:
                    self.rescuer_pos[i][axis] = -1.0
                    self.rescuer_vel[i][axis] *= -0.5

            # Tree Collision (reflect away from tree center and damp)
            for t_pos in self.tree_pos:
                to_tree = self.rescuer_pos[i] - t_pos
                dist = np.linalg.norm(to_tree)
                min_dist = self.agent_size + self.tree_radius
                if dist < min_dist:
                    # Track collision metric
                    self._collision_events += 1
                    # Tree collision penalty
                    rewards[agent] -= 1
                    # Compute penetration depth and normal
                    if dist > 1e-6:
                        n = to_tree / dist
                    else:
                        # Degenerate case: pick any normal (e.g., x-axis)
                        n = np.array([1.0, 0.0], dtype=float)
                    # Push the agent to the surface of the tree (no interpenetration)
                    self.rescuer_pos[i] = t_pos + n * min_dist
                    # Reflect velocity about the normal with damping
                    v = self.rescuer_vel[i]
                    vn = np.dot(v, n)
                    self.rescuer_vel[i] = (
                        v - (1.5 * vn) * n
                    )  # reflect and damp (~0.5 after reflection)

        # Agent-Agent Collision Physics (soft repulsion to prevent clustering)
        agent_repulsion_radius = 0.15
        repulsion_strength = 0.005
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                to_other = self.rescuer_pos[i] - self.rescuer_pos[j]
                dist = np.linalg.norm(to_other)

                if dist < agent_repulsion_radius and dist > 1e-6:
                    # Track collision between agents (count once per pair)
                    self._collision_events += 1
                    # Apply soft repulsion force inversely proportional to distance
                    repulsion_force = (
                        repulsion_strength * (agent_repulsion_radius - dist) / dist
                    )
                    direction = to_other / dist

                    # Apply equal and opposite forces
                    self.rescuer_vel[i] += direction * repulsion_force
                    self.rescuer_vel[j] -= direction * repulsion_force

        # 2. Victim Dynamics with Commitment System
        # Victims commit to an agent when approached, and maintain commitment
        # until the agent leaves or another agent is significantly closer
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                self.victim_vel[v_i] = 0
                continue

            # Find closest agent and their distance
            min_dist = float("inf")
            closest_agent_idx = -1
            for a_i, a_pos in enumerate(self.rescuer_pos):
                dist = np.linalg.norm(a_pos - self.victim_pos[v_i])
                if dist < min_dist:
                    min_dist = dist
                    closest_agent_idx = a_i

            current_assignment = self.victim_assignments[v_i]

            # Commitment logic with hysteresis
            if current_assignment == -1:
                # Not assigned - assign if agent is close enough
                if min_dist < self.follow_radius:
                    self.victim_assignments[v_i] = closest_agent_idx
                    current_assignment = closest_agent_idx
            else:
                # Already assigned - check if we should switch or release
                assigned_agent_pos = self.rescuer_pos[current_assignment]
                dist_to_assigned = np.linalg.norm(
                    assigned_agent_pos - self.victim_pos[v_i]
                )

                # Release if assigned agent is too far (1.5x follow radius = hysteresis)
                if dist_to_assigned > self.follow_radius * 1.5:
                    self.victim_assignments[v_i] = -1
                    current_assignment = -1
                    # Reassign if another agent is close
                    if min_dist < self.follow_radius:
                        self.victim_assignments[v_i] = closest_agent_idx
                        current_assignment = closest_agent_idx
                # Switch only if another agent is significantly closer (0.6x distance)
                elif (
                    closest_agent_idx != current_assignment
                    and min_dist < dist_to_assigned * 0.6
                ):
                    self.victim_assignments[v_i] = closest_agent_idx
                    current_assignment = closest_agent_idx

            # Movement based on assignment
            if current_assignment != -1:
                # Follow assigned agent
                assigned_pos = self.rescuer_pos[current_assignment]
                direction = (assigned_pos - self.victim_pos[v_i]) / (
                    np.linalg.norm(assigned_pos - self.victim_pos[v_i]) + 1e-6
                )
                follow_force = 0.03
                self.victim_vel[v_i] = (
                    self.victim_vel[v_i] * 0.8 + direction * follow_force
                )
            else:
                # Simple Brownian motion (using global np.random)
                noise = np.random.randn(2) * 0.0075
                self.victim_vel[v_i] = self.victim_vel[v_i] * 0.8 + noise

            self.victim_pos[v_i] += self.victim_vel[v_i]
            self.victim_pos[v_i] = np.clip(self.victim_pos[v_i], -1, 1)

        # Gentle exploration if an agent sees no victims (breaks idling)
        for i, agent in enumerate(self.agents):
            sees_victim = any(
                not self.victim_saved[v_i]
                and self._is_visible(
                    self.rescuer_pos[i], self.victim_pos[v_i], self.agent_size
                )
                for v_i in range(self.num_victims)
            )
            if (not sees_victim) and np.linalg.norm(self.rescuer_vel[i]) < 0.01:
                self.rescuer_vel[i] += np.random.uniform(-0.02, 0.02, size=2)

        # 3. Logic: Rescues & Rewards
        saved_count = self._compute_rewards(rewards)

        # Track coverage each step (after physics updates)
        for i, agent in enumerate(self.agents):
            self._visited_cells[i].add(self._hash_pos(self.rescuer_pos[i].copy()))

        self.steps += 1

        # Termination conditions
        if saved_count == self.num_victims:
            terminations = {a: True for a in self.agents}
            self.agents = []  # PettingZoo requires emptying agents list on termination
        elif self.steps >= self.max_steps:
            truncations = {a: True for a in self.agents}
            self.agents = []

        if not self.agents:
            # Episode ended; aggregate metrics
            rescues_pct = (
                100.0
                * float(np.count_nonzero(self.victim_saved))
                / max(1, self.num_victims)
            )
            coverage_cells = len(set().union(*self._visited_cells))
            collisions = self._collision_events
            self._completed_episode_metrics.append(
                {
                    "episode": self._episode_counter,
                    "rescues_pct": rescues_pct,
                    "collisions": collisions,
                    "coverage_cells": coverage_cells,
                }
            )
            self._reset_metrics()

        return self._get_obs(), rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment state.

        This method renders the current state of the environment using pygame.
        It displays:
        - Safe zones (colored circles with transparency)
        - Trees (gray circles)
        - Victims (colored circles matching their type)
        - Rescuers (white circles with outlines)
        - Vision circles (white outlines around rescuers)

        The rendering uses a 600x600 pixel window with coordinate transformation
        from world coordinates [-1, 1] to screen coordinates [0, 600].

        Returns:
            None if render_mode is "human" or None.
            RGB array if render_mode is "rgb_array" (not currently implemented).

        Note:
            Pygame is initialized on first render call. The screen and clock
            are stored as instance variables for subsequent render calls.
            Rendering runs at approximately 30 FPS.
        """
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()
            # Font for A/B/C/D labels
            self.font = pygame.font.SysFont("Arial", 18, bold=True)

        self.screen.fill((30, 30, 30))

        def to_screen(pos):
            x = (pos[0] + 1) / 2 * 600
            y = (1 - (pos[1] + 1) / 2) * 600
            return int(x), int(y)

        # Draw Safe Zones
        for i, pos in enumerate(self.safezone_pos):
            s_pos = to_screen(pos)
            r = int(self.safe_zone_radius * 300)

            type_idx = self.safe_zone_types[i]
            color = self.type_colors[type_idx]

            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, 50), (r, r), r)
            self.screen.blit(s, (s_pos[0] - r, s_pos[1] - r))
            pygame.draw.circle(self.screen, color, s_pos, r, 2)

        # Draw Trees
        for pos in self.tree_pos:
            pygame.draw.circle(
                self.screen,
                (100, 100, 100),
                to_screen(pos),
                int(self.tree_radius * 300),
            )

        # Draw Victims
        for i, pos in enumerate(self.victim_pos):
            if self.victim_saved[i]:
                continue

            type_idx = self.victim_types[i]
            color = self.type_colors[type_idx]

            screen_pos = to_screen(pos)
            pygame.draw.circle(
                self.screen, color, screen_pos, int(self.agent_size * 300)
            )

        # Draw Rescuers
        for i, pos in enumerate(self.rescuer_pos):
            pygame.draw.circle(
                self.screen, (200, 200, 200), to_screen(pos), int(self.agent_size * 300)
            )
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                to_screen(pos),
                int(self.agent_size * 300),
                2,
            )

        # Draw Vision Circles
        for i, pos in enumerate(self.rescuer_pos):
            screen_pos = to_screen(pos)
            vision_r = int(self.vision_radius * 300)
            pygame.draw.circle(
                self.screen, (255, 255, 255, 100), screen_pos, vision_r, 2
            )

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        """
        Clean up rendering resources.

        Closes the pygame display if it was initialized. This should be called
        when the environment is no longer needed to free system resources.
        """
        if self.screen:
            pygame.quit()

    def _get_matching_zone_idx(self, victim_type: int) -> Optional[int]:
        """Find the safe zone index that accepts the given victim type."""
        for zone_idx, zone_type in enumerate(self.safe_zone_types):
            if zone_type == victim_type:
                return zone_idx
        return None

    def _compute_agent_victim_dists(self) -> list[float]:
        """
        Compute minimum distance from each agent to the nearest unsaved victim.

        This is used for reward shaping to encourage agents to approach victims.
        Only unsaved victims are considered in the distance calculation.

        Returns:
            List of minimum distances, one per rescuer agent. Each value is
            the Euclidean distance to the closest unsaved victim. If no unsaved
            victims exist, returns 0.0 for all agents.
        """
        dists = []
        unsaved_indices = [k for k, saved in enumerate(self.victim_saved) if not saved]
        for i in range(self.num_rescuers):
            if unsaved_indices:
                min_dist = min(
                    np.linalg.norm(self.rescuer_pos[i] - self.victim_pos[k])
                    for k in unsaved_indices
                )
            else:
                min_dist = 0.0
            dists.append(min_dist)
        return dists

    def _bounded_zone_shaping(self, dist_to_zone: float) -> float:
        """
        Bounded shaping for escorting a victim to its safe zone.
        Returns a value in [0, 1], higher when closer.
        """
        # Smooth, bounded, stable (no explosion near 0)
        scale = 0.5  # adjust: smaller => steeper near zone
        return float(np.exp(-dist_to_zone / scale))

    def _nearest_unassigned_victim_dist(self, agent_idx: int) -> float:
        """
        Distance from an agent to the nearest *unassigned* and *unsaved* victim.
        If none exist, returns 0.0
        """
        best = float("inf")
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                continue
            if self.victim_assignments[v_i] != -1:
                continue  # already assigned to someone
            d = np.linalg.norm(self.rescuer_pos[agent_idx] - self.victim_pos[v_i])
            if d < best:
                best = d
        return 0.0 if best == float("inf") else float(best)

    def _compute_rewards(self, rewards) -> int:
        """
        Compute multi-component shaped rewards for the rescue task.

        This method implements assignment-aware reward shaping that guides agents through
        the victim rescue workflow: approach unassigned victims, escort assigned victims
        to matching safe zones, and receive sparse success bonuses upon rescue completion.

        Reward Components:
        ------------------
        1. **Pickup Shaping** (for non-escorting agents):
           - Distance penalty: -0.1 × distance_to_nearest_unassigned_victim
           - Delta shaping: +0.2 × (prev_dist - curr_dist)
           - Encourages agents without assignments to approach available victims

        2. **Sparse Success Reward**:
           - +100.0 for each victim successfully delivered to matching safe zone
           - Awarded to the assigned agent when victim enters zone radius
           - Only one reward per victim per episode

        3. **Escort Shaping** (for agents with assigned victims):
           - Bounded dense shaping: +1.0 × bounded_zone_proximity
             (mapped to [0,1] via exponential decay, prevents reward blowup)
           - Delta shaping: +0.5 × (prev_zone_dist - curr_zone_dist)
           - Encourages agents to bring their assigned victims closer to correct safe zones

        4. **Boundary Penalties**:
           - -0.2 for approaching arena boundaries (|x| > 0.95 or |y| > 0.95)
           - Discourages agents from leaving the operational area

        5. **Agent Collision Penalty**:
           - -1.0 to each agent when inter-agent distance < 0.15
           - Promotes spatial diversity and prevents excessive clustering

        6. **Idle Penalty**:
           - -0.01 for agents with velocity magnitude < 1e-3
           - Encourages active movement and exploration

        State Tracking:
        ---------------
        - `prev_agent_pickup_dists`: Per-agent distance to nearest unassigned victim
        - `prev_victim_zone_dists`: Per-victim distance to matching safe zone
        - Both used for delta-based progress shaping across timesteps

        Args:
            rewards: Dictionary mapping agent names to float rewards (modified in-place)

        Returns:
            int: Number of victims successfully saved this step (newly saved only)

        Notes:
        ------
        - Agents dynamically switch between pickup and escort modes based on assignment state
        - Escort shaping only activates when victim is assigned and not yet saved
        - All distance-based rewards use Euclidean L2 norm
        - Reward weights are tuned for balance between exploration, coordination, and task completion
        """

        # -----------------------------
        # 1) PICKUP shaping (unassigned agents -> nearest unassigned victim)
        # -----------------------------
        # Track previous distances for delta shaping (per agent) for pickup mode
        if not hasattr(self, "prev_agent_pickup_dists"):
            self.prev_agent_pickup_dists = [0.0 for _ in range(self.num_rescuers)]

        # Determine which agents are currently escorting at least one victim
        agent_is_escorting = [False for _ in range(self.num_rescuers)]
        for v_i in range(self.num_victims):
            a = int(self.victim_assignments[v_i])
            if a != -1 and not self.victim_saved[v_i] and 0 <= a < self.num_rescuers:
                agent_is_escorting[a] = True

        # Pickup shaping for non-escorting agents only
        for i, agent in enumerate(self.agents):
            if agent_is_escorting[i]:
                continue  # escort shaping happens below

            d = self._nearest_unassigned_victim_dist(i)

            # Distance penalty (encourage getting close to a victim to start follow/commit)
            rewards[agent] -= 0.1 * d

            # Delta shaping (reward improvement)
            prev = self.prev_agent_pickup_dists[i]
            if prev > 0.0:
                rewards[agent] += 0.2 * (prev - d)  # positive if closer
            self.prev_agent_pickup_dists[i] = d

        # -----------------------------
        # 2) SAVE events (sparse success reward +100)
        # -----------------------------
        saved_count = 0
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                saved_count += 1
                continue

            v_pos = self.victim_pos[v_i]
            v_type = self.victim_types[v_i]

            target_zone_idx = self._get_matching_zone_idx(v_type)
            if target_zone_idx is None:
                continue

            target_zone_pos = self.safezone_pos[target_zone_idx]
            dist_to_zone = float(np.linalg.norm(v_pos - target_zone_pos))

            # Victim saved
            if dist_to_zone < self.safe_zone_radius:
                self.victim_saved[v_i] = True
                saved_count += 1

                assigned_agent_idx = int(self.victim_assignments[v_i])
                if 0 <= assigned_agent_idx < len(self.agents):
                    rewards[self.agents[assigned_agent_idx]] += 100.0

        # -----------------------------
        # 3) ESCORT shaping (assigned agent -> bring victim closer to its zone)
        # -----------------------------
        # Track previous zone distances for delta shaping (per victim)
        if not hasattr(self, "prev_victim_zone_dists"):
            self.prev_victim_zone_dists = [None for _ in range(self.num_victims)]

        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                continue

            assigned_agent_idx = int(self.victim_assignments[v_i])
            if assigned_agent_idx == -1 or assigned_agent_idx >= len(self.agents):
                continue

            agent = self.agents[assigned_agent_idx]

            # Find correct safe zone by type
            v_type = self.victim_types[v_i]
            target_zone_idx = self._get_matching_zone_idx(v_type)
            if target_zone_idx is None:
                continue

            dist_to_zone = float(
                np.linalg.norm(
                    self.victim_pos[v_i] - self.safezone_pos[target_zone_idx]
                )
            )

            # Bounded dense shaping: in [0,1]
            # Encourages staying closer to the target zone (stable, no blow-ups)
            shaped = self._bounded_zone_shaping(dist_to_zone)
            rewards[agent] += 1.0 * shaped  # weight = 1.0, tune if needed

            # Delta shaping: reward reduction in distance (progress)
            prev = self.prev_victim_zone_dists[v_i]
            if prev is not None:
                rewards[agent] += 0.5 * (prev - dist_to_zone)  # positive if closer
            self.prev_victim_zone_dists[v_i] = dist_to_zone

        # -----------------------------
        # 4) Boundary penalties
        # -----------------------------
        for i, agent in enumerate(self.agents):
            pos = self.rescuer_pos[i]
            if abs(pos[0]) > 0.95 or abs(pos[1]) > 0.95:
                rewards[agent] -= 0.2  # softened from -1

        # -----------------------------
        # 5) Agent collision penalty (discourage clustering)
        # -----------------------------
        num_agents = len(self.agents)
        if num_agents > 1:
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(self.rescuer_pos[i] - self.rescuer_pos[j])
                    if dist < 0.15:
                        rewards[self.agents[i]] -= 1.0  # softened from -5
                        rewards[self.agents[j]] -= 1.0

        # -----------------------------
        # 6) Small penalty for idling
        # -----------------------------
        for i, agent in enumerate(self.agents):
            if np.linalg.norm(self.rescuer_vel[i]) < 1e-3:
                rewards[agent] -= 0.01

        return saved_count


def make_env(device: Union[torch.device, str] = "cpu", **kwargs) -> TransformedEnv:
    """
    Create a wrapped Search and Rescue environment for TorchRL.

    This factory function creates a SearchAndRescueEnv instance and wraps it
    with the necessary TorchRL transformations:
    1. PettingZooWrapper: Converts PettingZoo ParallelEnv to TorchRL format
    2. TransformedEnv with RewardSum: Adds episode reward tracking

    The wrapped environment is moved to the specified device (CPU or GPU)
    and is ready for use with TorchRL collectors and training pipelines.

    Args:
        device: Device to run the environment on. Can be "cpu", "cuda",
            or a torch.device object.
        **kwargs: Additional keyword arguments passed to SearchAndRescueEnv
            constructor. Common arguments include:
            - num_rescuers: Number of rescuer agents
            - num_victims: Number of victim entities
            - num_trees: Number of obstacle trees
            - num_safe_zones: Number of safe zones (typically 4)
            - max_cycles: Maximum steps per episode
            - vision_radius: Vision/observation radius
            - render_mode: Rendering mode ("human", None, etc.)
            - seed: Random seed

    Returns:
        TransformedEnv instance ready for TorchRL training/evaluation.
        The environment includes:
        - Observation key: ("agents", "observation")
        - Action key: ("agents", "action")
        - Reward key: ("agents", "reward")
        - Episode reward key: ("agents", "episode_reward")
        - Done key: ("agents", "done")
        - Terminated key: ("agents", "terminated")

    Example:
        >>> env = make_env(device="cuda", num_rescuers=4, num_victims=8)
        >>> td = env.reset()
        >>> print(td.keys())  # Shows available keys
    """
    env = SearchAndRescueEnv(**kwargs)
    group_map = {"agents": env.possible_agents}
    env = PettingZooWrapper(env, group_map=group_map, use_mask=True, device=device)
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env.to(device)
