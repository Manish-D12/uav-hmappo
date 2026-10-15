"""Two-timescale hierarchical wrapper around :class:`~agents.mappo.MAPPO`.

Why a hierarchy
---------------
The three decisions in this problem do not change at the same natural rate.
Where to fly and which users to serve are *strategic*: a UAV that re-picks its
cluster every second thrashes, and a 20 m/s platform cannot meaningfully
reposition in one step anyway.  How to split bandwidth and power, by contrast,
is *tactical* and should react immediately to the channel.

So the agent is split in two, each level being an ordinary MAPPO instance
over a subset of the action heads:

===========  ==================================  ==============  =============
level        action heads                        decides every   reward
===========  ==================================  ==============  =============
high         ``move`` + ``cluster``              ``interval``    window sum
low          ``resource``                        1 step          per step
===========  ==================================  ==============  =============

The high level's choice is *held* for the whole window, so the low level
learns to make the best of whatever cluster and trajectory it was handed.
This is the standard credit split: the high level is scored on the total
reward accumulated across its window, the low level on the immediate reward.

Goal conditioning
-----------------
The low level cannot allocate bandwidth sensibly without knowing which users
it is serving, so its observation is the environment observation concatenated
with its own cluster bids, and its critic state is the global state
concatenated with every UAV's bids.  That is what makes this a real hierarchy
rather than two policies sharing an environment by accident.

A note on the high-level discount
---------------------------------
One high-level transition spans ``interval`` environment steps, so its
discount factor is not the environment's.  If you want an effective per-step
discount of ``gamma``, set the high level's ``gamma`` to ``gamma ** interval``
(with the defaults: ``0.99 ** 5 ~= 0.95``).
"""

from __future__ import annotations

import numpy as np

from agents.mappo import MAPPO, MAPPOConfig, heads_to_env_action


class HierarchicalMAPPO:
    """Hierarchical MAPPO with a slow strategic level and a fast tactical one.

    Exposes the same ``begin_episode`` / ``act`` / ``record`` / ``finish``
    interface as :class:`~agents.mappo.MAPPO`, so the training loop does not
    need to know which of the two it is driving.

    Parameters
    ----------
    obs_dim : int
        Per-agent observation size reported by the environment.
    state_dim : int
        Global state size reported by the environment.
    n_uav, n_users : int
        Scenario dimensions.
    interval : int
        Number of environment steps a high-level decision is held for.
    high_config, low_config : MAPPOConfig, optional
        Separate hyperparameters per level.  The low level usually wants a
        slightly larger entropy bonus, since its action space is wider.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_uav: int,
        n_users: int,
        interval: int = 5,
        high_config: MAPPOConfig | None = None,
        low_config: MAPPOConfig | None = None,
    ):
        if interval < 1:
            raise ValueError("interval must be >= 1")

        self.n_uav, self.n_users = n_uav, n_users
        self.interval = int(interval)

        # High level: strategy. Sees the raw environment observation.
        self.high = MAPPO(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_uav=n_uav,
            n_users=n_users,
            heads=("move", "cluster"),
            config=high_config or MAPPOConfig(),
        )

        # Low level: tactics. Sees the observation plus the cluster it was given.
        self.low = MAPPO(
            obs_dim=obs_dim + n_users,
            state_dim=state_dim + n_uav * n_users,
            n_uav=n_uav,
            n_users=n_users,
            heads=("resource",),
            config=low_config or MAPPOConfig(),
        )

        self._reset_scratch()

    # ------------------------------------------------------------ internals
    def _reset_scratch(self) -> None:
        """Clear the per-episode bookkeeping."""
        self.t = 0
        self._cluster_bids = np.zeros((self.n_uav, self.n_users), dtype=np.float32)
        self._cluster_probs: np.ndarray | None = None
        self._high_actions: dict[str, np.ndarray] = {}
        self._high_pending: dict | None = None
        self._window_reward = 0.0
        self._window_len = 0

    def _low_obs(self, obs: np.ndarray) -> np.ndarray:
        """Condition each UAV's observation on the cluster it was assigned."""
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._cluster_bids], axis=1)

    def _low_state(self, state: np.ndarray) -> np.ndarray:
        """Condition the centralised state on the full clustering decision."""
        return np.concatenate(
            [np.asarray(state, dtype=np.float32), self._cluster_bids.reshape(-1)]
        )

    def _commit_high(
        self,
        reward: float,
        terminated: bool = False,
        truncated: bool = False,
        boot_value: float = 0.0,
    ) -> None:
        """Push the open high-level transition into its buffer.

        A window that closes mid-episode is an ordinary transition: the next
        high-level value already sits in the buffer.  A window closed by the
        episode ending is a boundary and needs the terminated/truncated
        distinction, exactly as at the low level.
        """
        if self._high_pending is None:
            return
        p = self._high_pending
        self.high.store(
            p["obs"], p["state"], p["actions"], p["log_prob"], p["value"],
            reward, terminated, truncated, boot_value,
        )
        self._high_pending = None
        self._window_reward = 0.0
        self._window_len = 0

    # ---------------------------------------------------- episode interface
    def begin_episode(self) -> None:
        """Reset timers and the held high-level decision."""
        self._reset_scratch()

    def act(self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Produce one environment action from both levels.

        The high level only fires at the start of each window; in between,
        its previous decision is reused verbatim.
        """
        obs = np.asarray(obs, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)

        # --- high level: strategic decision, refreshed once per window ---
        if self.t % self.interval == 0:
            actions, log_prob, value, cluster_probs = self.high.select_action(
                obs, state, deterministic
            )
            self._high_pending = {
                "obs": obs,
                "state": state,
                "actions": actions,
                "log_prob": log_prob,
                "value": value,
            }
            self._high_actions = actions
            self._cluster_probs = cluster_probs
            self._cluster_bids = np.asarray(actions["cluster"], dtype=np.float32)

        # --- low level: tactical decision, every single step ---
        low_actions, low_log_prob, low_value, _ = self.low.select_action(
            self._low_obs(obs), self._low_state(state), deterministic
        )
        self._low_pending = {
            "obs": self._low_obs(obs),
            "state": self._low_state(state),
            "actions": low_actions,
            "log_prob": low_log_prob,
            "value": low_value,
        }

        return heads_to_env_action(
            {**self._high_actions, **low_actions},
            self.n_uav,
            self.n_users,
            self._cluster_probs,
            obs,
        )

    def record(
        self,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        next_state: np.ndarray | None = None,
    ) -> None:
        """Route the reward to both levels.

        The low level books it immediately.  The high level accumulates it and
        only books the total when its window closes - or early, if the episode
        ends mid-window.
        """
        if getattr(self, "_low_pending", None) is None:
            raise RuntimeError("record() called before act()")

        episode_over = terminated or truncated
        needs_boot = truncated and not terminated and next_state is not None

        # --- low level: one transition per environment step ---
        boot_low = (
            self.low.get_value(self._low_state(next_state)) if needs_boot else 0.0
        )
        p = self._low_pending
        self.low.store(
            p["obs"], p["state"], p["actions"], p["log_prob"], p["value"],
            reward, terminated, truncated, boot_low,
        )
        self._low_pending = None

        self._window_reward += float(reward)
        self._window_len += 1
        self.t += 1

        # --- high level: one transition per window ---
        if self._window_len >= self.interval or episode_over:
            boot_high = self.high.get_value(next_state) if needs_boot else 0.0
            self._commit_high(
                self._window_reward,
                terminated=terminated,
                # Only an episode boundary counts as an end for the high level;
                # a window closing mid-episode is an ordinary transition.
                truncated=truncated if episode_over else False,
                boot_value=boot_high,
            )

    def set_lr_fraction(self, fraction: float) -> None:
        """Anneal both levels' learning rates (no-op unless lr_decay is set)."""
        self.high.set_lr_fraction(fraction)
        self.low.set_lr_fraction(fraction)

    def finish(self, last_obs: np.ndarray, last_state: np.ndarray) -> dict[str, float]:
        """Close any open window, then update both levels.

        Returns
        -------
        dict
            PPO diagnostics with ``high_``/``low_`` prefixes so the two levels
            can be logged and plotted separately.
        """
        last_obs = np.asarray(last_obs, dtype=np.float32)
        last_state = np.asarray(last_state, dtype=np.float32)

        # Defensive: if the caller never reported the episode end, a partial
        # window may still be open.  Book it as truncated so its tail is
        # bootstrapped rather than silently dropped.
        if self._high_pending is not None:
            self._commit_high(
                self._window_reward,
                terminated=False,
                truncated=True,
                boot_value=self.high.get_value(last_state),
            )

        high_stats = self.high.update(self.high.get_value(last_state))
        low_stats = self.low.update(self.low.get_value(self._low_state(last_state)))

        stats = {f"high_{k}": v for k, v in high_stats.items()}
        stats.update({f"low_{k}": v for k, v in low_stats.items()})
        return stats

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        """Write both levels to ``path`` with ``.high``/``.low`` suffixes."""
        self.high.save(f"{path}.high")
        self.low.save(f"{path}.low")

    def load(self, path: str) -> None:
        """Restore both levels from checkpoints written by :meth:`save`."""
        self.high.load(f"{path}.high")
        self.low.load(f"{path}.low")
