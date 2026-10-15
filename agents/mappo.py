"""Multi-Agent PPO (MAPPO) with centralised training and decentralised execution.

Structure
---------
* **Actor** - one network, shared by every UAV (the agents are homogeneous, so
  parameter sharing is both cheaper and better-conditioned than separate
  networks).  It reads only a UAV's *local* observation, which is what makes
  execution decentralised.  Three heads emit the three kinds of decision:

  ==============  ===============  ==========================================
  head            distribution     meaning
  ==============  ===============  ==========================================
  ``move``        Gaussian (2)     continuous 2-D displacement
  ``cluster``     Bernoulli (M)    one bid per user: "do I want to serve it?"
  ``resource``    Gaussian (M+1)   bandwidth weights per user, plus power
  ==============  ===============  ==========================================

* **Critic** - a single centralised network that sees the *global* state
  (every UAV position, every residual energy, every user position).  It is
  used only during training, so it never breaks the decentralised execution
  guarantee.  Because :class:`~envs.uav_env.UAVEnv` hands out one shared team
  reward, one scalar value per timestep is all that is needed.

* **Update** - standard PPO: GAE(lambda) advantages, a clipped surrogate
  objective, a clipped value loss, and an entropy bonus.

Note on the clustering head
---------------------------
The environment expects a *continuous* preference score per user and settles
contention with an ``argmax`` across UAVs.  The policy's discrete Bernoulli
bid is mapped onto that interface by :func:`heads_to_env_action`.

When the agent's observations are available, the score is *channel-anchored*:
each UAV starts from its own path loss to the user (read from its local
observation, so execution stays decentralised) and its bid nudges that by
``BID_MARGIN_DB`` either way.  With no bids the argmax is therefore
"strongest channel wins", and the policy only has to learn where to deviate
from it - e.g. shedding boundary users to balance load.

This matters a great deal.  Letting 60 independent coin flips decide the
clustering from scratch means each user lands on a random UAV, usually not
the one it can hear best; with UAVs parked at the ideal K-means positions
that alone caps coverage at ~32%, and one shared team reward is far too weak
a signal for PPO to untangle 60 binary decisions.

Without observations the older scheme is used: a bid of 1 lands in the upper
band of the score range and a bid of 0 in the lower one, with the Bernoulli
probability breaking ties between UAVs that bid for the same user.

Note on continuous actions
--------------------------
The Gaussian heads are unbounded, and the raw sample is what gets stored and
scored by ``log_prob``.  The action is clipped to ``[-1, 1]`` only on its way
into the environment.  This is the usual "Gaussian + clip" formulation; it
avoids the change-of-variables correction a tanh squash would require, at the
cost of a little probability mass sitting on the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

# The three action heads, in the canonical order used everywhere below.
ALL_HEADS: tuple[str, ...] = ("move", "cluster", "resource")

# Channel-anchored clustering (see the module docstring).  The environment
# observation ends with the per-user path loss normalised as (PL - 100) / 50,
# so one dB is 1/50 of a unit.  A bid shifts a UAV's score by this many dB
# either way: enough to move users near a cluster boundary, small enough that
# exploratory coin flips cannot hand a user to a UAV it barely hears.
PATH_LOSS_NORM_DB = 50.0
BID_MARGIN_DB = 3.0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class MAPPOConfig:
    """Hyperparameters for the PPO update."""

    hidden_dim: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2        # <= 0 disables value clipping
    entropy_coef: float = 0.01
    reward_scale: float = 1.0            # rewards are multiplied by this before GAE
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    minibatch_size: int = 64
    init_log_std: float = -0.5
    # Linearly anneal both learning rates to 0 over training (train.py calls
    # set_lr_fraction before each update).  Off by default.
    lr_decay: bool = False
    device: str = "cpu"


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
def mlp(in_dim: int, hidden_dim: int, out_dim: int | None = None) -> nn.Sequential:
    """Two-layer tanh MLP, optionally with a linear output layer.

    Tanh activations are the usual choice for PPO: they keep activations
    bounded, which pairs well with the small, noisy batches of on-policy RL.
    """
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
    ]
    if out_dim is not None:
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Orthogonal weight init, the standard recipe for stable PPO training."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)
    return module


class Actor(nn.Module):
    """Per-agent policy with one head per action type.

    Parameters
    ----------
    obs_dim : int
        Size of a single UAV's local observation.
    n_users : int
        Number of ground users, which sizes the clustering and bandwidth heads.
    heads : sequence of str
        Which of ``("move", "cluster", "resource")`` this actor controls.  The
        hierarchical agent uses this to split the action space across two
        policies operating at different timescales.
    """

    def __init__(
        self,
        obs_dim: int,
        n_users: int,
        heads: Sequence[str] = ALL_HEADS,
        hidden_dim: int = 128,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        unknown = set(heads) - set(ALL_HEADS)
        if unknown:
            raise ValueError(f"unknown head(s) {sorted(unknown)}; choose from {ALL_HEADS}")
        if not heads:
            raise ValueError("an actor needs at least one head")

        self.heads = tuple(heads)
        self.n_users = n_users
        self.body = _orthogonal_init(mlp(obs_dim, hidden_dim))

        # Policy heads get a small gain so the initial policy is near-uniform
        # and does not commit to an arbitrary behaviour before learning starts.
        if "move" in self.heads:
            self.move_mean = _orthogonal_init(nn.Linear(hidden_dim, 2), gain=0.01)
            self.move_log_std = nn.Parameter(torch.full((2,), init_log_std))
        if "cluster" in self.heads:
            self.cluster_logits = _orthogonal_init(nn.Linear(hidden_dim, n_users), gain=0.01)
        if "resource" in self.heads:
            self.resource_mean = _orthogonal_init(nn.Linear(hidden_dim, n_users + 1), gain=0.01)
            self.resource_log_std = nn.Parameter(torch.full((n_users + 1,), init_log_std))

    # ------------------------------------------------------------------
    def head_dims(self) -> dict[str, int]:
        """Number of scalar action components produced by each active head."""
        sizes = {"move": 2, "cluster": self.n_users, "resource": self.n_users + 1}
        return {h: sizes[h] for h in self.heads}

    def distributions(self, obs: torch.Tensor) -> dict[str, torch.distributions.Distribution]:
        """Build the per-head action distributions for a batch of observations.

        Parameters
        ----------
        obs : Tensor, shape (batch, obs_dim)

        Returns
        -------
        dict
            Head name to distribution.  ``log_std`` is clamped to keep the
            Gaussians from collapsing to a delta or exploding early in
            training.
        """
        z = self.body(obs)
        dists: dict[str, torch.distributions.Distribution] = {}

        if "move" in self.heads:
            std = self.move_log_std.clamp(-5.0, 2.0).exp().expand_as(self.move_mean(z))
            dists["move"] = Normal(self.move_mean(z), std)
        if "cluster" in self.heads:
            dists["cluster"] = Bernoulli(logits=self.cluster_logits(z))
        if "resource" in self.heads:
            mean = self.resource_mean(z)
            std = self.resource_log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
            dists["resource"] = Normal(mean, std)
        return dists

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Sample an action and return it together with its log-probability.

        Parameters
        ----------
        obs : Tensor, shape (batch, obs_dim)
        deterministic : bool
            If True, take the distribution mode instead of sampling - used at
            evaluation time.

        Returns
        -------
        actions : dict of Tensor
            Head name to sampled action, shape ``(batch, head_dim)``.
        log_prob : Tensor, shape (batch,)
            Joint log-probability, summed over heads and action dimensions.
        """
        dists = self.distributions(obs)
        actions, log_prob = {}, torch.zeros(obs.shape[0], device=obs.device)

        for name, dist in dists.items():
            if deterministic:
                sample = dist.mean if isinstance(dist, Normal) else (dist.probs > 0.5).float()
            else:
                sample = dist.sample()
            actions[name] = sample
            log_prob = log_prob + dist.log_prob(sample).sum(dim=-1)
        return actions, log_prob

    def evaluate(
        self, obs: torch.Tensor, actions: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-score stored actions under the current policy.

        This is what makes PPO's importance ratio possible: the same actions
        are evaluated again after the parameters have moved.

        Returns
        -------
        log_prob : Tensor, shape (batch,)
        entropy : Tensor, shape (batch,)
            Summed over heads; used for the entropy bonus.
        """
        dists = self.distributions(obs)
        log_prob = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)

        for name, dist in dists.items():
            log_prob = log_prob + dist.log_prob(actions[name]).sum(dim=-1)
            entropy = entropy + dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Centralised value function over the global state.

    Sees everything - all UAVs and all users - which gives it a stationary
    learning target even though each agent's own view of the world shifts as
    the others change their policies.  Only ever used during training.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = _orthogonal_init(mlp(state_dim, hidden_dim))
        self.head = _orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate ``V(s)`` for a batch of global states, shape ``(batch,)``."""
        return self.head(self.net(state)).squeeze(-1)


# --------------------------------------------------------------------------
# Action conversion
# --------------------------------------------------------------------------
def heads_to_env_action(
    actions: dict[str, np.ndarray],
    n_uav: int,
    n_users: int,
    cluster_probs: np.ndarray | None = None,
    obs: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the environment's ``(n_uav, action_dim)`` array from head outputs.

    Missing heads default to a neutral value, which is what lets the
    hierarchical agent fill in the slots its two policies own separately:
    a zero move means "hold position", a zero bandwidth weight means "split
    evenly", and a cluster bid of 0 means "not interested".

    Parameters
    ----------
    actions : dict
        Head name to array of shape ``(n_uav, head_dim)``.
    n_uav, n_users : int
        Scenario dimensions.
    cluster_probs : ndarray, optional
        Bernoulli probabilities of shape ``(n_uav, n_users)``.  Used to break
        ties between UAVs that bid for the same user; without it, tied bids
        are resolved by UAV index, which biases clustering towards UAV 0.
    obs : ndarray, optional
        Environment observations of shape ``(n_uav, obs_dim)``.  When given,
        clustering scores are anchored on each UAV's own path loss to every
        user (the last ``n_users`` entries of its observation).

    Returns
    -------
    ndarray, shape (n_uav, 2 + 2 * n_users + 1)
        A valid environment action, clipped to ``[-1, 1]``.
    """
    action = np.zeros((n_uav, 2 + 2 * n_users + 1), dtype=np.float32)

    if "move" in actions:
        action[:, 0:2] = actions["move"]

    if "cluster" in actions:
        bids = np.asarray(actions["cluster"], dtype=float)
        if obs is not None:
            # Higher score = stronger channel.  Stays inside [-1, 1]: the
            # normalised path loss spans roughly [-0.45, 0.45] over the area.
            pl_norm = np.asarray(obs, dtype=float)[:, -n_users:]
            margin = BID_MARGIN_DB / PATH_LOSS_NORM_DB
            score = -pl_norm + margin * (2.0 * bids - 1.0)
            if cluster_probs is not None:
                score = score + 0.2 * margin * (np.asarray(cluster_probs, dtype=float) - 0.5)
            action[:, 2 : 2 + n_users] = score
        else:
            _legacy_cluster_scores(action, bids, n_users, cluster_probs)

    if "resource" in actions:
        resource = np.asarray(actions["resource"], dtype=float)
        action[:, 2 + n_users : 2 + 2 * n_users] = resource[:, :n_users]
        action[:, -1] = resource[:, n_users]

    return np.clip(action, -1.0, 1.0)


def _legacy_cluster_scores(
    action: np.ndarray, bids: np.ndarray, n_users: int, cluster_probs: np.ndarray | None
) -> None:
    """Channel-agnostic bid-to-score mapping, used when no observation is given."""
    # Bid 1 -> upper band [0.25, 0.75]; bid 0 -> lower band [-0.75, -0.25].
    # Any bidder therefore outranks any non-bidder under the env's argmax.
    score = 0.5 * (2.0 * bids - 1.0)
    if cluster_probs is not None:
        score = score + 0.5 * (np.asarray(cluster_probs, dtype=float) - 0.5)
    action[:, 2 : 2 + n_users] = score


# --------------------------------------------------------------------------
# Rollout storage
# --------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """On-policy storage for one batch of experience.

    Observations and actions are per-agent (leading shape ``(T, n_uav)``)
    while states, rewards and values are per-timestep (shape ``(T,)``),
    mirroring the centralised-critic / shared-reward setup.
    """

    obs: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[dict[str, np.ndarray]] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    ends: list[bool] = field(default_factory=list)
    boot_values: list[float] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: dict[str, np.ndarray],
        log_prob: np.ndarray,
        value: float,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        boot_value: float = 0.0,
    ) -> None:
        """Append one transition.

        Parameters
        ----------
        terminated : bool
            The episode reached a genuine terminal state (flat batteries).
            No future reward exists, so no bootstrap is applied.
        truncated : bool
            The episode was cut by the time limit.  The future still exists,
            so ``boot_value`` is used - but the GAE trace must not run on
            into the next episode.
        boot_value : float
            ``V(s_next)``, only consulted when this step ends an episode.
        """
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append({k: np.asarray(v, dtype=np.float32) for k, v in actions.items()})
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.ends.append(bool(terminated or truncated))
        self.boot_values.append(float(boot_value))

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        """Drop everything; PPO is on-policy, so data is used exactly once."""
        for store in (self.obs, self.states, self.actions, self.log_probs,
                      self.values, self.rewards, self.terminated, self.ends,
                      self.boot_values):
            store.clear()

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float, reward_scale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation over the stored trajectory.

        The buffer may hold several concatenated episodes.  At every episode
        boundary two things happen: the bootstrap value switches from
        ``values[t+1]`` to the stored ``boot_value`` (or 0 for a true
        terminal), and the advantage trace is reset so that credit never
        leaks backwards from one episode into its predecessor.

        Parameters
        ----------
        last_value : float
            ``V(s_T)`` for the state after the final stored step.  Only used
            when the buffer ends *mid*-episode, e.g. on a partial rollout.
        gamma, gae_lambda : float
            Discount and GAE trace-decay parameters.
        reward_scale : float
            Multiplier applied to the stored rewards.  Keeps the critic's
            regression targets O(10) rather than O(100); the stored values and
            bootstraps are already on the scaled scale because the critic was
            trained on it.

        Returns
        -------
        advantages, returns : ndarray, shape (T,)
        """
        rewards = np.asarray(self.rewards, dtype=np.float32) * np.float32(reward_scale)
        values = np.asarray(self.values, dtype=np.float32)
        terminated = np.asarray(self.terminated, dtype=bool)
        ends = np.asarray(self.ends, dtype=bool)
        boots = np.asarray(self.boot_values, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            if ends[t]:
                # A terminal state has no future; a truncated one does.
                next_value = 0.0 if terminated[t] else boots[t]
                trace = 0.0            # cut: the next episode is unrelated
            else:
                next_value = values[t + 1] if t + 1 < n else last_value
                trace = 1.0
            delta = rewards[t] + gamma * next_value - values[t]
            running = delta + gamma * gae_lambda * trace * running
            advantages[t] = running
        return advantages, advantages + values


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class MAPPO:
    """MAPPO agent: shared actor, centralised critic, PPO update.

    Parameters
    ----------
    obs_dim : int
        Per-agent observation size.
    state_dim : int
        Global state size for the centralised critic.
    n_uav, n_users : int
        Scenario dimensions.
    heads : sequence of str
        Which action heads this agent controls.
    config : MAPPOConfig, optional
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_uav: int,
        n_users: int,
        heads: Sequence[str] = ALL_HEADS,
        config: MAPPOConfig | None = None,
    ):
        self.cfg = config or MAPPOConfig()
        self.device = torch.device(self.cfg.device)
        self.n_uav, self.n_users = n_uav, n_users
        self.heads = tuple(heads)

        self.actor = Actor(
            obs_dim, n_users, heads, self.cfg.hidden_dim, self.cfg.init_log_std
        ).to(self.device)
        self.critic = Critic(state_dim, self.cfg.hidden_dim).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------- acting
    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float, np.ndarray | None]:
        """Choose an action for every UAV from their local observations.

        Parameters
        ----------
        obs : ndarray, shape (n_uav, obs_dim)
        state : ndarray, shape (state_dim,)
            Global state, consumed by the critic only.
        deterministic : bool
            Use distribution modes instead of sampling.

        Returns
        -------
        actions : dict of ndarray
            Head name to array of shape ``(n_uav, head_dim)``.
        log_prob : ndarray, shape (n_uav,)
        value : float
            Centralised value estimate of the current state.
        cluster_probs : ndarray or None
            Bernoulli probabilities, passed on for tie-breaking.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        actions_t, log_prob_t = self.actor.act(obs_t, deterministic=deterministic)
        value = float(self.critic(state_t).item())

        cluster_probs = None
        if "cluster" in self.heads:
            cluster_probs = self.actor.distributions(obs_t)["cluster"].probs.cpu().numpy()

        actions = {k: v.cpu().numpy() for k, v in actions_t.items()}
        return actions, log_prob_t.cpu().numpy(), value, cluster_probs

    def to_env_action(
        self,
        actions: dict[str, np.ndarray],
        cluster_probs: np.ndarray | None = None,
        obs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience wrapper around :func:`heads_to_env_action`."""
        return heads_to_env_action(actions, self.n_uav, self.n_users, cluster_probs, obs)

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """Centralised value estimate of a single global state."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(state_t).item())

    def store(self, *args, **kwargs) -> None:
        """Append a transition to the rollout buffer (see :meth:`RolloutBuffer.add`)."""
        self.buffer.add(*args, **kwargs)

    # --------------------------------------------------- episode interface
    # The four methods below (begin_episode / act / record / finish) are the
    # contract the training loop talks to.  HierarchicalMAPPO implements the
    # same four, so scripts/train.py can drive either agent unchanged.
    def begin_episode(self) -> None:
        """Reset the per-episode scratch state before the first action."""
        self._pending: dict | None = None

    def act(self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Pick an environment action and remember what it takes to score it.

        The transition is only committed to the buffer once :meth:`record` is
        called with the resulting reward.
        """
        actions, log_prob, value, cluster_probs = self.select_action(obs, state, deterministic)
        self._pending = {
            "obs": obs,
            "state": state,
            "actions": actions,
            "log_prob": log_prob,
            "value": value,
        }
        return self.to_env_action(actions, cluster_probs, obs)

    def record(
        self,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        next_state: np.ndarray | None = None,
    ) -> None:
        """Complete the transition opened by the last :meth:`act` call.

        Parameters
        ----------
        reward : float
        terminated, truncated : bool
            Kept distinct on purpose - see :meth:`RolloutBuffer.add`.
        next_state : ndarray, optional
            Global state reached after this step.  Only needed when the step
            truncates an episode, where ``V(next_state)`` is the bootstrap.
        """
        if getattr(self, "_pending", None) is None:
            raise RuntimeError("record() called before act()")

        boot = 0.0
        if truncated and not terminated and next_state is not None:
            boot = self.get_value(next_state)

        p = self._pending
        self.store(
            p["obs"], p["state"], p["actions"], p["log_prob"], p["value"],
            reward, terminated, truncated, boot,
        )
        self._pending = None

    def finish(self, last_obs: np.ndarray, last_state: np.ndarray) -> dict[str, float]:
        """Bootstrap from the final state and run the PPO update.

        ``last_obs`` is unused here but is part of the shared interface, since
        the hierarchical agent needs it to build its low-level observation.
        """
        del last_obs  # required by the shared interface, not by this agent
        return self.update(self.get_value(last_state))

    def set_lr_fraction(self, fraction: float) -> None:
        """Scale both learning rates to ``fraction`` of their configured values.

        Does nothing unless ``lr_decay`` is enabled.
        """
        if not self.cfg.lr_decay:
            return
        fraction = float(np.clip(fraction, 0.0, 1.0))
        for optim, base in ((self.actor_optim, self.cfg.lr_actor),
                            (self.critic_optim, self.cfg.lr_critic)):
            for group in optim.param_groups:
                group["lr"] = base * fraction

    # ------------------------------------------------------------- update
    def update(self, last_value: float = 0.0) -> dict[str, float]:
        """Run the PPO update on everything in the buffer, then clear it.

        Parameters
        ----------
        last_value : float
            Bootstrap value for the state after the last stored transition.

        Returns
        -------
        dict
            Mean policy loss, value loss, entropy, approximate KL and clip
            fraction over the update - the usual PPO diagnostics.  An empty
            dict if there was nothing to learn from.
        """
        if len(self.buffer) == 0:
            return {}

        cfg = self.cfg
        advantages, returns = self.buffer.compute_gae(
            last_value, cfg.gamma, cfg.gae_lambda, cfg.reward_scale
        )

        obs = torch.as_tensor(np.asarray(self.buffer.obs), device=self.device)          # (T,N,O)
        states = torch.as_tensor(np.asarray(self.buffer.states), device=self.device)    # (T,S)
        old_log_probs = torch.as_tensor(np.asarray(self.buffer.log_probs), device=self.device)
        old_values = torch.as_tensor(np.asarray(self.buffer.values, dtype=np.float32), device=self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)

        actions = {
            head: torch.as_tensor(
                np.stack([a[head] for a in self.buffer.actions]), device=self.device
            )
            for head in self.heads
        }

        # One shared reward means one advantage per timestep, applied to every
        # agent's own action - the standard MAPPO credit assignment.
        adv_per_agent = adv_t.unsqueeze(1).expand(-1, self.n_uav)

        n_steps = len(self.buffer)
        batch_size = min(cfg.minibatch_size, n_steps)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

        for _ in range(cfg.ppo_epochs):
            order = torch.randperm(n_steps, device=self.device)
            for start in range(0, n_steps, batch_size):
                idx = order[start : start + batch_size]
                if idx.numel() < 2:
                    continue  # a single-sample minibatch makes normalisation undefined

                mb_obs = obs[idx].reshape(-1, obs.shape[-1])
                mb_actions = {h: actions[h][idx].reshape(-1, actions[h].shape[-1]) for h in self.heads}
                mb_old_log = old_log_probs[idx].reshape(-1)
                mb_adv = adv_per_agent[idx].reshape(-1)

                # Normalise advantages inside the minibatch for scale stability.
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                log_prob, entropy = self.actor.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(log_prob - mb_old_log)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy_loss = -entropy.mean()

                self.actor_optim.zero_grad()
                (policy_loss + cfg.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optim.step()

                # --- centralised critic, one value per timestep ---
                value_pred = self.critic(states[idx])
                value_loss = (value_pred - ret_t[idx]) ** 2
                if cfg.value_clip_range > 0:
                    # Clipping the value update the same way as the policy
                    # keeps the critic from lurching after a noisy batch.  The
                    # range is absolute, though: it caps how far V can move per
                    # update, so it must be small relative to the returns.
                    clipped_pred = old_values[idx] + torch.clamp(
                        value_pred - old_values[idx], -cfg.value_clip_range, cfg.value_clip_range
                    )
                    value_loss = torch.max(value_loss, (clipped_pred - ret_t[idx]) ** 2)
                value_loss = cfg.value_loss_coef * value_loss.mean()

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optim.step()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append((mb_old_log - log_prob).mean().item())
                    stats["clip_frac"].append(
                        ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                    )

        self.buffer.clear()
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    # ------------------------------------------------------------- io
    def save(self, path: str) -> None:
        """Persist actor, critic and optimiser state to ``path``."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "heads": self.heads,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore from a checkpoint written by :meth:`save`."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
