"""Gymnasium environment for joint UAV deployment, clustering and resource allocation.

The scenario
------------
``N`` rotary-wing UAVs serve ``M`` ground users inside a square area.  At every
decision step each UAV simultaneously chooses three things:

1. **Deployment** - a continuous 2-D displacement (the UAV flies at a fixed
   altitude, so only ``x`` and ``y`` are controlled).
2. **Clustering** - a preference score for every user.  Each user is then
   attached to the UAV that scores it highest, which partitions the users
   into one cluster per UAV.
3. **Resource allocation** - how to divide its bandwidth among the users it
   ended up serving, and what fraction of its power budget to radiate.

All UAVs reuse the whole band, so a UAV's transmission is co-channel
interference for every user it does not serve.  This is what couples the
agents: moving closer to your own cluster also means polluting somebody
else's, which is precisely the trade-off a multi-agent policy has to learn.

Multi-agent interface
---------------------
The class follows the single-agent Gymnasium API (``reset``/``step`` with a
5-tuple), but observations and actions are stacked per agent with shape
``(n_uav, ...)``.  A centralised critic can call :meth:`global_state` to get
the full system state.  This "stacked Box" convention keeps the environment
usable with plain Gymnasium tooling while still being MAPPO-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs import channel_model as cm


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class UAVEnvConfig:
    """All tunable knobs of the scenario, with course-project-friendly defaults."""

    # --- scenario geometry ---
    n_uav: int = 3
    n_users: int = 20
    area_size_m: float = 1000.0          # side length of the square service area
    altitude_m: float = 100.0            # fixed UAV flight altitude
    max_speed_mps: float = 20.0          # maximum horizontal speed
    dt_s: float = 1.0                    # duration of one decision step

    # --- radio ---
    carrier_hz: float = 2.0e9
    bandwidth_hz: float = 20.0e6         # total band, fully reused by every UAV
    tx_power_max_dbm: float = 30.0       # per-UAV radiated power budget
    tx_power_min_dbm: float = 10.0
    noise_figure_db: float = 7.0
    antenna_gain_db: float = 0.0
    environment: str = "urban"           # key into channel_model.ENVIRONMENTS
    min_rate_bps: float = 1.0e6          # a user below this counts as uncovered

    # --- energy ---
    initial_energy_j: float = 300_000.0
    hover_power_w: float = 150.0         # power needed just to stay airborne
    move_power_coeff: float = 0.5        # extra propulsion power per (m/s)^2

    # --- reward shaping ---
    w_throughput: float = 1.0
    w_coverage: float = 1.0
    w_energy: float = 0.2
    w_waste: float = 0.2
    rate_norm_bps: float = 10.0e6        # per-user rate that scores 1.0
    # Potential-based positioning shaping (Ng et al., 1999), added on top of
    # compute_reward().  The potential is minus the mean user-to-nearest-UAV
    # distance (normalised by area size), so flying past your users - which
    # pulls them further from every UAV - is penalised the step it happens,
    # instead of only via a slow drop in coverage.  Being potential-based, it
    # does not change which policy is optimal.  0 disables it.
    w_shaping: float = 0.0
    shaping_gamma: float = 0.99          # match the agent's per-step gamma
    # Penalty for UAV pairs closer than separation_m, discouraging UAVs from
    # stacking on the same users.  Each pair costs up to 1 (when coincident),
    # averaged over pairs.  0 disables it.
    w_separation: float = 0.0
    separation_m: float = 250.0

    # --- episode ---
    max_steps: int = 200
    user_mobility_mps: float = 0.0       # 0 keeps users static
    # Probability that reset() scatters the UAVs uniformly over the area
    # instead of bunching them at the centre.  A training aid: starting only
    # from the centre, "fly outwards" always pays, and learned UAVs overshoot
    # their users and park at the walls.  Scenarios that pin ``uav_xy`` (the
    # fixed evaluation benchmark) are never affected.
    random_uav_start_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.n_uav < 1 or self.n_users < 1:
            raise ValueError("n_uav and n_users must both be >= 1")
        if self.tx_power_min_dbm > self.tx_power_max_dbm:
            raise ValueError("tx_power_min_dbm must not exceed tx_power_max_dbm")
        if not 0.0 <= self.random_uav_start_prob <= 1.0:
            raise ValueError("random_uav_start_prob must lie in [0, 1]")

    @classmethod
    def from_yaml(cls, path: str, section: str | None = "env") -> "UAVEnvConfig":
        """Build a config from a YAML file, ignoring unknown keys.

        Parameters
        ----------
        path : str
            Path to the YAML file.
        section : str or None
            Top-level key to read from, e.g. ``"env"`` in the sectioned
            ``configs/default.yaml``.  Pass ``None`` for a flat file.  A
            missing section falls back to the top level, so both layouts work.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if section is not None and isinstance(raw.get(section), dict):
            raw = raw[section]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


# --------------------------------------------------------------------------
# Shared helpers (used by both the environment and the K-means baseline)
# --------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def allocate_bandwidth(
    weights: np.ndarray, assignment: np.ndarray, n_uav: int, total_bw_hz: float
) -> np.ndarray:
    """Split each UAV's band among the users in its own cluster.

    Parameters
    ----------
    weights : ndarray, shape (n_uav, n_users)
        Raw (unnormalised) allocation scores produced by the policy.
    assignment : ndarray of int, shape (n_users,)
        Index of the UAV serving each user.
    n_uav : int
        Number of UAVs.
    total_bw_hz : float
        Band available to *each* UAV (full frequency reuse).

    Returns
    -------
    ndarray, shape (n_users,)
        Bandwidth in hertz granted to each user.  Users in an empty or
        unreachable cluster receive 0.
    """
    bw = np.zeros(assignment.shape[0], dtype=float)
    for i in range(n_uav):
        members = np.flatnonzero(assignment == i)
        if members.size == 0:
            continue
        bw[members] = softmax(weights[i, members]) * total_bw_hz
    return bw


def compute_reward(
    rates_bps: np.ndarray,
    bandwidth_hz: np.ndarray,
    energy_used_j: float,
    cfg: UAVEnvConfig,
) -> tuple[float, dict[str, float]]:
    """Team reward shared by every agent, plus its individual terms.

    The reward balances four objectives::

        r = w_thr * throughput  +  w_cov * coverage
            - w_energy * energy  -  w_waste * wasted spectrum

    * **throughput** - mean per-user rate normalised by ``rate_norm_bps`` and
      capped at 1 per user.  Without the cap the mean is dominated by the one
      or two users right under a UAV, and the policy is paid more for
      starving everyone else to feed them (~2.3/step) than for the K-means
      layout (~1.8/step).  Capped, every term lies in ``[0, 1]``.
    * **coverage** - fraction of users whose rate clears ``min_rate_bps``.
    * **energy** - energy drawn this step, normalised by the worst case of
      every UAV flying flat out for one step.
    * **waste** - share of the allocated spectrum handed to users that still
      failed to reach ``min_rate_bps``; that bandwidth bought nothing.

    Both the environment and the baselines call this function, so their
    numbers are directly comparable.

    Returns
    -------
    reward : float
    terms : dict
        The four normalised components, handy for logging and plots.
    """
    rates_bps = np.asarray(rates_bps, dtype=float)
    bandwidth_hz = np.asarray(bandwidth_hz, dtype=float)

    covered = rates_bps >= cfg.min_rate_bps
    throughput = float(np.mean(np.minimum(rates_bps / cfg.rate_norm_bps, 1.0)))
    coverage = float(np.mean(covered))

    max_speed_power = cfg.hover_power_w + cfg.move_power_coeff * cfg.max_speed_mps**2
    worst_case_energy = max_speed_power * cfg.dt_s * cfg.n_uav
    energy = float(energy_used_j / max(worst_case_energy, 1e-9))

    total_bw = float(np.sum(bandwidth_hz))
    wasted_bw = float(np.sum(bandwidth_hz[~covered]))
    waste = wasted_bw / total_bw if total_bw > 0 else 1.0

    reward = (
        cfg.w_throughput * throughput
        + cfg.w_coverage * coverage
        - cfg.w_energy * energy
        - cfg.w_waste * waste
    )
    terms = {
        "throughput_term": throughput,
        "coverage_term": coverage,
        "energy_term": energy,
        "waste_term": waste,
    }
    return float(reward), terms


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class UAVEnv(gym.Env):
    """Joint UAV deployment, user clustering and resource allocation.

    Observation (per UAV, shape ``(n_uav, obs_dim)``)
        * own position, normalised to ``[-1, 1]``            -> 2
        * own residual energy fraction                       -> 1
        * relative position of every other UAV               -> 2 * (n_uav - 1)
        * relative position of every user                    -> 2 * n_users
        * normalised path loss to every user (channel state) -> n_users

    Action (per UAV, shape ``(n_uav, action_dim)``, all entries in ``[-1, 1]``)
        * ``[0:2]``                     movement direction, scaled by max speed
        * ``[2 : 2+M]``                 clustering preference for each user
        * ``[2+M : 2+2M]``              bandwidth weight for each user
        * ``[-1]``                      transmit power level

    Reward
        A single team reward (see :func:`compute_reward`) shared by all
        agents, which is the usual setting for MAPPO with a centralised
        critic, plus ``w_shaping`` times a potential-based positioning term
        (see :meth:`_positioning_potential`).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: UAVEnvConfig | dict | None = None, render_mode: str | None = None):
        super().__init__()
        if isinstance(config, dict):
            config = UAVEnvConfig(**config)
        self.cfg = config or UAVEnvConfig()
        self.render_mode = render_mode

        n, m = self.cfg.n_uav, self.cfg.n_users
        self.n_uav, self.n_users = n, m

        self.obs_dim = 2 + 1 + 2 * (n - 1) + 2 * m + m
        self.action_dim = 2 + 2 * m + 1

        # Per-agent stacked spaces.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n, self.obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n, self.action_dim), dtype=np.float32
        )
        # Centralised-critic state: UAV positions + energies + user positions.
        self.state_dim = 3 * n + 2 * m
        self.share_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Filled in by reset().
        self.uav_xy = np.zeros((n, 2), dtype=float)
        self.user_xy = np.zeros((m, 2), dtype=float)
        self.energy_j = np.zeros(n, dtype=float)
        self.step_count = 0
        self._last_rates = np.zeros(m, dtype=float)
        self._potential = 0.0

    # ---------------------------------------------------------------- reset
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode, optionally replaying a fixed scenario.

        By default users are drawn uniformly over the service area and UAVs
        start bunched near the centre, so the policy has to learn to spread
        out.  With probability ``random_uav_start_prob`` the UAVs are instead
        scattered uniformly over the area.

        Parameters
        ----------
        seed : int, optional
            Seeds the episode's random draws.
        options : dict, optional
            May contain ``"user_xy"`` of shape ``(n_users, 2)`` and/or
            ``"uav_xy"`` of shape ``(n_uav, 2)`` to pin the layout instead of
            sampling it.  This is how a saved evaluation scenario is replayed,
            so that every method is scored on identical user layouts.
        """
        super().reset(seed=seed)
        cfg = self.cfg
        size = cfg.area_size_m
        options = options or {}

        if "user_xy" in options:
            user_xy = np.asarray(options["user_xy"], dtype=float)
            if user_xy.shape != (self.n_users, 2):
                raise ValueError(
                    f"options['user_xy'] has shape {user_xy.shape}, "
                    f"expected {(self.n_users, 2)}"
                )
            self.user_xy = np.clip(user_xy.copy(), 0.0, size)
        else:
            self.user_xy = self.np_random.uniform(0.0, size, size=(self.n_users, 2))

        if "uav_xy" in options:
            uav_xy = np.asarray(options["uav_xy"], dtype=float)
            if uav_xy.shape != (self.n_uav, 2):
                raise ValueError(
                    f"options['uav_xy'] has shape {uav_xy.shape}, "
                    f"expected {(self.n_uav, 2)}"
                )
            self.uav_xy = np.clip(uav_xy.copy(), 0.0, size)
        elif self.np_random.random() < cfg.random_uav_start_prob:
            self.uav_xy = self.np_random.uniform(0.0, size, size=(self.n_uav, 2))
        else:
            centre = np.full((self.n_uav, 2), size / 2.0)
            jitter = self.np_random.uniform(-0.05 * size, 0.05 * size, size=(self.n_uav, 2))
            self.uav_xy = np.clip(centre + jitter, 0.0, size)

        self.energy_j = np.full(self.n_uav, cfg.initial_energy_j, dtype=float)
        self.step_count = 0
        self._last_rates = np.zeros(self.n_users, dtype=float)
        self._potential = self._positioning_potential()

        obs = self._build_observations()
        info = {"state": self.global_state()}
        return obs, info

    # ----------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one decision step.

        Order of operations: move the UAVs (paying the propulsion energy),
        let users drift, form clusters, allocate bandwidth and power,
        evaluate the channel, then score the result.
        """
        cfg = self.cfg
        action = np.asarray(action, dtype=float).reshape(self.n_uav, self.action_dim)
        action = np.clip(action, -1.0, 1.0)

        alive = self.energy_j > 0.0

        # --- 1. deployment -------------------------------------------------
        velocity = action[:, 0:2] * cfg.max_speed_mps          # (N, 2) m/s
        # The two components are bounded independently, so a diagonal command
        # would otherwise reach sqrt(2) * max_speed.  Rescale any over-fast
        # vector back onto the speed limit, preserving its direction.
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        excess = speed > cfg.max_speed_mps
        velocity = np.where(excess, velocity * cfg.max_speed_mps / np.maximum(speed, 1e-9), velocity)
        velocity[~alive] = 0.0                                  # dead UAVs ground out
        self.uav_xy = np.clip(
            self.uav_xy + velocity * cfg.dt_s, 0.0, cfg.area_size_m
        )

        # --- 2. user mobility (optional random walk) -----------------------
        if cfg.user_mobility_mps > 0.0:
            drift = self.np_random.uniform(
                -cfg.user_mobility_mps, cfg.user_mobility_mps, size=self.user_xy.shape
            )
            self.user_xy = np.clip(self.user_xy + drift * cfg.dt_s, 0.0, cfg.area_size_m)

        # --- 3. clustering: each user joins its highest-scoring live UAV ----
        preference = action[:, 2 : 2 + self.n_users].copy()      # (N, M)
        preference[~alive] = -np.inf
        if not np.any(alive):
            # Every UAV is out of battery: nothing can be served any more.
            obs = self._build_observations()
            reward, terms = compute_reward(
                np.zeros(self.n_users), np.zeros(self.n_users), 0.0, cfg
            )
            self.step_count += 1
            info = self._build_info(
                np.zeros(self.n_users), np.zeros(self.n_users),
                np.zeros(self.n_users, dtype=int), np.zeros(self.n_users), 0.0, terms,
            )
            return obs, reward, True, False, info

        assignment = np.argmax(preference, axis=0)               # (M,)

        # --- 4. resource allocation ---------------------------------------
        bw_weights = action[:, 2 + self.n_users : 2 + 2 * self.n_users]
        bandwidth = allocate_bandwidth(
            bw_weights, assignment, self.n_uav, cfg.bandwidth_hz
        )

        # Power level in [-1, 1] -> [min, max] dBm, per UAV.
        power_level = (action[:, -1] + 1.0) / 2.0
        uav_power_dbm = cfg.tx_power_min_dbm + power_level * (
            cfg.tx_power_max_dbm - cfg.tx_power_min_dbm
        )
        uav_power_dbm = np.where(alive, uav_power_dbm, -np.inf)

        tx_matrix = self._build_tx_matrix(uav_power_dbm, assignment, bandwidth)

        # --- 5. channel evaluation ----------------------------------------
        rates, sinr = cm.downlink_rates(
            uav_xy=self.uav_xy,
            altitudes=np.full(self.n_uav, cfg.altitude_m),
            user_xy=self.user_xy,
            assignment=assignment,
            bandwidth_hz=bandwidth,
            tx_power_dbm=tx_matrix,
            carrier_hz=cfg.carrier_hz,
            environment=cfg.environment,
            noise_figure_db=cfg.noise_figure_db,
            antenna_gain_db=cfg.antenna_gain_db,
        )
        rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_rates = rates

        # --- 6. energy bookkeeping ----------------------------------------
        speed = np.linalg.norm(velocity, axis=1)
        flight_power = cfg.hover_power_w + cfg.move_power_coeff * speed**2
        comm_power = np.where(alive, cm.dbm_to_watt(uav_power_dbm), 0.0)
        step_energy = np.where(alive, (flight_power + comm_power) * cfg.dt_s, 0.0)
        step_energy = np.minimum(step_energy, self.energy_j)     # cannot overdraw
        self.energy_j = np.maximum(self.energy_j - step_energy, 0.0)

        # --- 7. reward and episode bookkeeping -----------------------------
        reward, terms = compute_reward(rates, bandwidth, float(step_energy.sum()), cfg)
        potential = self._positioning_potential()
        shaping = cfg.shaping_gamma * potential - self._potential
        self._potential = potential
        terms["shaping_term"] = float(shaping)
        reward += cfg.w_shaping * shaping
        terms["separation_term"] = self._crowding()
        reward -= cfg.w_separation * terms["separation_term"]
        self.step_count += 1

        terminated = bool(np.all(self.energy_j <= 0.0))
        truncated = bool(self.step_count >= cfg.max_steps)

        obs = self._build_observations()
        info = self._build_info(
            rates, sinr, assignment, bandwidth, float(step_energy.sum()), terms
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- internals
    def _build_tx_matrix(
        self, uav_power_dbm: np.ndarray, assignment: np.ndarray, bandwidth: np.ndarray
    ) -> np.ndarray:
        """Per-pair transmit powers in dBm, shape ``(n_uav, n_users)``.

        For a UAV that does *not* serve user ``j`` the entry is the UAV's full
        radiated power, because all of it lands on ``j`` as co-channel
        interference.  For the serving UAV the entry is only the slice of
        power devoted to that user's stream, split in proportion to the
        bandwidth it was granted.
        """
        n, m = self.n_uav, self.n_users
        tx = np.tile(uav_power_dbm[:, None], (1, m))

        for i in range(n):
            members = np.flatnonzero(assignment == i)
            if members.size == 0:
                continue
            share = bandwidth[members]
            total = share.sum()
            fraction = share / total if total > 0 else np.full(members.size, 1.0 / members.size)
            with np.errstate(divide="ignore"):
                tx[i, members] = uav_power_dbm[i] + 10.0 * np.log10(
                    np.maximum(fraction, 1e-12)
                )
        return tx

    def _crowding(self) -> float:
        """Mean over UAV pairs of ``max(0, 1 - distance / separation_m)``."""
        if self.n_uav < 2:
            return 0.0
        i, j = np.triu_indices(self.n_uav, k=1)
        d = np.linalg.norm(self.uav_xy[i] - self.uav_xy[j], axis=1)
        return float(np.mean(np.maximum(0.0, 1.0 - d / self.cfg.separation_m)))

    def _positioning_potential(self) -> float:
        """Minus the mean distance from each user to its nearest UAV, in area units.

        Only live UAVs count; with none left the potential is that of the
        farthest possible placement, so no shaping reward is left to collect.
        """
        alive = self.energy_j > 0.0
        if not np.any(alive):
            return -np.sqrt(2.0)
        d = np.linalg.norm(
            self.uav_xy[alive][:, None, :] - self.user_xy[None, :, :], axis=-1
        )
        return -float(np.mean(np.min(d, axis=0))) / self.cfg.area_size_m

    def _build_observations(self) -> np.ndarray:
        """Assemble the per-agent observation matrix of shape ``(n_uav, obs_dim)``."""
        cfg = self.cfg
        size = cfg.area_size_m

        # Normalise coordinates to [-1, 1] so the network sees well-scaled inputs.
        uav_norm = self.uav_xy / size * 2.0 - 1.0                    # (N, 2)
        user_norm = self.user_xy / size * 2.0 - 1.0                  # (M, 2)
        energy_frac = (self.energy_j / cfg.initial_energy_j)[:, None]  # (N, 1)

        # Channel state: path loss from each UAV to each user, roughly centred.
        pl = cm.path_loss_db(
            self.uav_xy[:, None, :],
            np.full((self.n_uav, 1), cfg.altitude_m),
            self.user_xy[None, :, :],
            cfg.carrier_hz,
            cfg.environment,
        )                                                             # (N, M)
        pl_norm = (pl - 100.0) / 50.0

        rows = []
        for i in range(self.n_uav):
            others = [j for j in range(self.n_uav) if j != i]
            rel_uav = (uav_norm[others] - uav_norm[i]).reshape(-1) if others else np.zeros(0)
            rel_user = (user_norm - uav_norm[i]).reshape(-1)
            rows.append(
                np.concatenate([uav_norm[i], energy_frac[i], rel_uav, rel_user, pl_norm[i]])
            )
        return np.asarray(rows, dtype=np.float32)

    def _build_info(
        self,
        rates: np.ndarray,
        sinr: np.ndarray,
        assignment: np.ndarray,
        bandwidth: np.ndarray,
        energy_used: float,
        terms: dict[str, float],
    ) -> dict[str, Any]:
        """Collect raw per-step quantities for logging and metric computation."""
        info: dict[str, Any] = {
            "rates_bps": rates,
            "sinr": sinr,
            "assignment": assignment,
            "bandwidth_hz": bandwidth,
            "energy_used_j": energy_used,
            "energy_remaining_j": self.energy_j.copy(),
            "uav_xy": self.uav_xy.copy(),
            "user_xy": self.user_xy.copy(),
            "state": self.global_state(),
        }
        info.update(terms)
        return info

    def global_state(self) -> np.ndarray:
        """Flat global state for a centralised critic.

        Concatenates normalised UAV positions, residual energy fractions and
        normalised user positions into a single vector of length
        ``3 * n_uav + 2 * n_users``.
        """
        size = self.cfg.area_size_m
        return np.concatenate(
            [
                (self.uav_xy / size * 2.0 - 1.0).reshape(-1),
                self.energy_j / self.cfg.initial_energy_j,
                (self.user_xy / size * 2.0 - 1.0).reshape(-1),
            ]
        ).astype(np.float32)

    # --------------------------------------------------------------- render
    def render(self) -> None:
        """Print a compact text summary of the current step."""
        if self.render_mode != "human":
            return
        print(
            f"step {self.step_count:3d} | "
            f"mean rate {np.mean(self._last_rates) / 1e6:6.2f} Mbps | "
            f"energy {np.round(self.energy_j / 1e3, 1)} kJ"
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        """No resources to free; present for API completeness."""
        return None
