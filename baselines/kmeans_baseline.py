"""K-means reference policy for UAV clustering and deployment.

This is the classical, non-learning approach that the hierarchical MAPPO
agent has to beat:

1. Cluster the users geographically with K-means.
2. Fly each UAV towards its cluster centroid.
3. Serve exactly the users of that cluster, splitting bandwidth equally and
   transmitting at a fixed power level.

The baseline is written as a *policy* - :func:`kmeans_action` turns an
environment state into an action array - rather than as a separate
simulation.  Running it through :class:`~envs.uav_env.UAVEnv` means it is
scored by the very same channel model, reward function and metrics as the
learned policy, so the comparison is apples to apples.

Known limitations (which is the point of a baseline): K-means optimises
geometric compactness only.  It ignores interference between UAVs, the
residual energy of each platform, and the fact that a large cluster gets a
thinner slice of bandwidth per user.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from envs.uav_env import UAVEnv, UAVEnvConfig, compute_reward


# --------------------------------------------------------------------------
# Clustering and UAV-to-cluster matching
# --------------------------------------------------------------------------
def kmeans_clusters(
    user_xy: np.ndarray, n_clusters: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Partition users into geographic clusters.

    Parameters
    ----------
    user_xy : ndarray, shape (M, 2)
        User positions in metres.
    n_clusters : int
        Desired number of clusters; automatically reduced if there are fewer
        users than clusters.
    seed : int, optional
        Seed for K-means initialisation, for reproducibility.

    Returns
    -------
    centroids : ndarray, shape (k, 2)
        Cluster centres, i.e. the "ideal" UAV hover points.
    labels : ndarray of int, shape (M,)
        Cluster index of each user.
    """
    user_xy = np.asarray(user_xy, dtype=float)
    k = int(min(n_clusters, user_xy.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(user_xy)
    return km.cluster_centers_, labels


def match_uavs_to_centroids(uav_xy: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Greedily assign each centroid to its nearest still-unclaimed UAV.

    Without a matching step two UAVs can chase the same centroid and leave
    another cluster unserved.  Pairs are claimed in order of increasing
    distance, which is a cheap stand-in for the optimal Hungarian assignment
    and is more than adequate at course-project scale.

    Returns
    -------
    ndarray of int, shape (n_uav,)
        Centroid index served by each UAV, or ``-1`` for a UAV left over
        because there were more UAVs than clusters.
    """
    uav_xy = np.asarray(uav_xy, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    n_uav, k = uav_xy.shape[0], centroids.shape[0]

    dist = np.linalg.norm(uav_xy[:, None, :] - centroids[None, :, :], axis=-1)
    target = np.full(n_uav, -1, dtype=int)
    free_uav = set(range(n_uav))
    free_centroid = set(range(k))

    # Visit all (uav, centroid) pairs from closest to furthest.
    for flat in np.argsort(dist, axis=None):
        i, c = np.unravel_index(flat, dist.shape)
        if i in free_uav and c in free_centroid:
            target[i] = c
            free_uav.discard(int(i))
            free_centroid.discard(int(c))
        if not free_centroid or not free_uav:
            break
    return target


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def kmeans_action(
    env: UAVEnv, seed: int | None = None, power_level: float = 1.0
) -> np.ndarray:
    """Build the environment action implementing the K-means heuristic.

    Parameters
    ----------
    env : UAVEnv
        Environment whose *current* state is used; nothing is mutated.
    seed : int, optional
        Seed passed to K-means.
    power_level : float
        Transmit power in ``[0, 1]``, mapped onto the configured dBm range.
        The default of 1.0 is deliberately naive: the heuristic always
        transmits at full power and therefore generates maximum co-channel
        interference.

    Returns
    -------
    ndarray, shape (n_uav, action_dim)
        A valid action in ``[-1, 1]`` for :meth:`UAVEnv.step`.
    """
    cfg = env.cfg
    n, m = env.n_uav, env.n_users

    centroids, labels = kmeans_clusters(env.user_xy, n, seed=seed)
    target = match_uavs_to_centroids(env.uav_xy, centroids)

    action = np.zeros((n, env.action_dim), dtype=np.float32)

    for i in range(n):
        # --- deployment: head for the centroid, capped at one step of flight.
        if target[i] >= 0:
            delta = centroids[target[i]] - env.uav_xy[i]
            reachable = cfg.max_speed_mps * cfg.dt_s
            # Normalising by the reachable distance puts a full-speed move at
            # +/-1 and a short hop proportionally below it.
            move = np.clip(delta / max(reachable, 1e-9), -1.0, 1.0)
        else:
            move = np.zeros(2)                       # surplus UAV: hold position
        action[i, 0:2] = move

        # --- clustering: strongly prefer this UAV's own cluster members.
        preference = np.full(m, -1.0)
        if target[i] >= 0:
            preference[labels == target[i]] = 1.0
        action[i, 2 : 2 + m] = preference

        # --- resource allocation: equal bandwidth split, fixed power.
        # Equal weights make the softmax inside the env uniform over whichever
        # users end up in this UAV's cluster.
        action[i, 2 + m : 2 + 2 * m] = 0.0
        action[i, -1] = np.clip(2.0 * power_level - 1.0, -1.0, 1.0)

    return action


# --------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------
def run_episode(
    env: UAVEnv,
    seed: int | None = None,
    power_level: float = 1.0,
    options: dict | None = None,
) -> dict[str, np.ndarray | float]:
    """Run one full episode under the K-means policy.

    Parameters
    ----------
    env : UAVEnv
    seed : int, optional
        Episode seed, also used to seed K-means.
    power_level : float
        Transmit power in ``[0, 1]``.
    options : dict, optional
        Passed straight to :meth:`UAVEnv.reset`, which is how a fixed
        evaluation scenario is replayed.

    Returns
    -------
    dict
        ``return`` (undiscounted episode return), ``steps``, and the
        per-step history of rates, bandwidth, reward and reward terms.  The
        history is what :mod:`utils.metrics` consumes.
    """
    obs, _ = env.reset(seed=seed, options=options)
    rewards: list[float] = []
    rates_hist: list[np.ndarray] = []
    bw_hist: list[np.ndarray] = []
    energy_hist: list[float] = []

    terminated = truncated = False
    while not (terminated or truncated):
        action = kmeans_action(env, seed=seed, power_level=power_level)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        rates_hist.append(info["rates_bps"])
        bw_hist.append(info["bandwidth_hz"])
        energy_hist.append(info["energy_used_j"])

    return {
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "rewards": np.asarray(rewards),
        "rates_bps": np.asarray(rates_hist),        # (T, M)
        "bandwidth_hz": np.asarray(bw_hist),        # (T, M)
        "energy_used_j": np.asarray(energy_hist),   # (T,)
    }


def evaluate(
    config: UAVEnvConfig | None = None,
    n_episodes: int = 5,
    seed: int = 0,
    power_level: float = 1.0,
) -> dict[str, float]:
    """Average the K-means baseline over several episodes.

    Each episode uses a different seed, so the UAVs face a fresh random user
    layout every time.

    Returns
    -------
    dict
        Mean episode return, mean per-user rate in Mbps, and mean coverage.
    """
    cfg = config or UAVEnvConfig()
    env = UAVEnv(cfg)

    returns, mean_rates, coverages = [], [], []
    for ep in range(n_episodes):
        result = run_episode(env, seed=seed + ep, power_level=power_level)
        rates = result["rates_bps"]
        returns.append(result["return"])
        mean_rates.append(float(np.mean(rates)))
        coverages.append(float(np.mean(rates >= cfg.min_rate_bps)))

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "mean_rate_mbps": float(np.mean(mean_rates) / 1e6),
        "mean_coverage": float(np.mean(coverages)),
    }


def reward_of_static_deployment(
    user_xy: np.ndarray, cfg: UAVEnvConfig, seed: int | None = None
) -> tuple[float, dict[str, float]]:
    """Score a one-shot K-means deployment without stepping the environment.

    Useful for quick sanity checks and for plots that only need the quality
    of the placement itself.  UAVs are teleported to the centroids, bandwidth
    is shared equally inside each cluster, and the result is passed through
    the environment's own :func:`~envs.uav_env.compute_reward`.
    """
    from envs import channel_model as cm

    user_xy = np.asarray(user_xy, dtype=float)
    centroids, labels = kmeans_clusters(user_xy, cfg.n_uav, seed=seed)
    n_uav = centroids.shape[0]

    bandwidth = np.zeros(user_xy.shape[0])
    for i in range(n_uav):
        members = np.flatnonzero(labels == i)
        if members.size:
            bandwidth[members] = cfg.bandwidth_hz / members.size

    # Serving UAV splits power across its cluster; interferers use full power.
    tx = np.tile(np.full(n_uav, cfg.tx_power_max_dbm)[:, None], (1, user_xy.shape[0]))
    for i in range(n_uav):
        members = np.flatnonzero(labels == i)
        if members.size:
            tx[i, members] = cfg.tx_power_max_dbm - 10.0 * np.log10(members.size)

    rates, _ = cm.downlink_rates(
        uav_xy=centroids,
        altitudes=np.full(n_uav, cfg.altitude_m),
        user_xy=user_xy,
        assignment=labels,
        bandwidth_hz=bandwidth,
        tx_power_dbm=tx,
        carrier_hz=cfg.carrier_hz,
        environment=cfg.environment,
        noise_figure_db=cfg.noise_figure_db,
        antenna_gain_db=cfg.antenna_gain_db,
    )
    hover_energy = cfg.hover_power_w * cfg.dt_s * cfg.n_uav
    return compute_reward(rates, bandwidth, hover_energy, cfg)


# --------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------
def main(n_episodes: int = 5, seed: int = 0, power_level: float = 1.0) -> None:
    """Evaluate the baseline, print its metrics and save them to CSV.

    Writes ``results/baseline_metrics.csv`` at the project root with one row
    per episode followed by a ``mean`` row.
    """
    import csv
    import os

    from utils import metrics

    cfg = UAVEnvConfig()
    env = UAVEnv(cfg)

    columns = [
        ("total_throughput_mbps", "Throughput", "Mbps"),
        ("avg_rate_mbps", "Average data rate", "Mbps"),
        ("jains_fairness", "Jain's fairness", ""),
        ("energy_efficiency_kbit_per_j", "Energy efficiency", "kbit/J"),
        ("coverage", "Coverage", ""),
    ]

    rows = []
    for ep in range(n_episodes):
        result = run_episode(env, seed=seed + ep, power_level=power_level)
        summary = metrics.summarize(
            result["rates_bps"], cfg.min_rate_bps, energy_j=result["energy_used_j"]
        )
        rows.append({"episode": ep, **{key: summary[key] for key, _, _ in columns}})
    env.close()

    mean_row = {"episode": "mean"}
    for key, _, _ in columns:
        mean_row[key] = float(np.mean([row[key] for row in rows]))

    print(f"K-means baseline ({n_episodes} episodes, power level {power_level})")
    print("-" * 44)
    for key, label, unit in columns:
        value = mean_row[key]
        text = f"{value:.2%}" if key == "coverage" else f"{value:.4f}"
        print(f"{label + ':':<22}{text} {unit}".rstrip())

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode"] + [key for key, _, _ in columns])
        writer.writeheader()
        writer.writerows(rows + [mean_row])
    print("-" * 44)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
