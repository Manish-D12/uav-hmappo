"""Sanity tests for the UAV environment, channel model, metrics and baseline.

These are contract tests rather than numerical-accuracy tests: they check
that shapes line up, that values stay finite and inside their theoretical
bounds, and that the physics moves in the direction it should.  Run them
with::

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.kmeans_baseline import kmeans_action, kmeans_clusters, match_uavs_to_centroids
from envs import channel_model as cm
from envs.uav_env import UAVEnv, UAVEnvConfig, compute_reward
from utils import metrics


@pytest.fixture
def cfg() -> UAVEnvConfig:
    """A small, fast configuration used by most tests."""
    return UAVEnvConfig(n_uav=3, n_users=12, max_steps=10)


@pytest.fixture
def env(cfg: UAVEnvConfig) -> UAVEnv:
    """A freshly constructed environment."""
    return UAVEnv(cfg)


# --------------------------------------------------------------------------
# Environment: reset
# --------------------------------------------------------------------------
def test_reset_returns_valid_observation(env: UAVEnv):
    """reset() yields an in-space observation and the centralised state."""
    obs, info = env.reset(seed=0)

    assert obs.shape == (env.n_uav, env.obs_dim)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.all(np.isfinite(obs))
    assert info["state"].shape == (env.state_dim,)


def test_reset_places_everything_inside_the_service_area(env: UAVEnv, cfg: UAVEnvConfig):
    """UAVs and users start within the square area, at full battery."""
    env.reset(seed=1)

    assert env.uav_xy.shape == (cfg.n_uav, 2)
    assert env.user_xy.shape == (cfg.n_users, 2)
    assert np.all((env.uav_xy >= 0) & (env.uav_xy <= cfg.area_size_m))
    assert np.all((env.user_xy >= 0) & (env.user_xy <= cfg.area_size_m))
    assert np.allclose(env.energy_j, cfg.initial_energy_j)


def test_reset_is_reproducible(cfg: UAVEnvConfig):
    """The same seed reproduces the same scenario; a different one does not."""
    a, b, c = UAVEnv(cfg), UAVEnv(cfg), UAVEnv(cfg)
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    obs_c, _ = c.reset(seed=43)

    assert np.array_equal(obs_a, obs_b)
    assert not np.array_equal(obs_a, obs_c)


# --------------------------------------------------------------------------
# Environment: step
# --------------------------------------------------------------------------
def test_step_returns_valid_shapes_and_finite_reward(env: UAVEnv):
    """step() honours the Gymnasium 5-tuple contract."""
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert obs.shape == (env.n_uav, env.obs_dim)
    assert np.all(np.isfinite(obs))
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)

    assert info["rates_bps"].shape == (env.n_users,)
    assert info["assignment"].shape == (env.n_users,)
    assert info["bandwidth_hz"].shape == (env.n_users,)


def test_reward_is_finite_for_many_random_actions(env: UAVEnv):
    """Random play never produces NaN/inf rewards, rates or observations."""
    env.reset(seed=0)
    for _ in range(env.cfg.max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(info["rates_bps"]))
        assert np.all(info["rates_bps"] >= 0.0)
        if terminated or truncated:
            break


def test_step_respects_physical_limits(env: UAVEnv, cfg: UAVEnvConfig):
    """UAVs stay in bounds, never exceed max speed, and only lose energy."""
    env.reset(seed=2)
    previous_xy = env.uav_xy.copy()
    previous_energy = env.energy_j.copy()
    max_step_distance = cfg.max_speed_mps * cfg.dt_s + 1e-6

    for _ in range(5):
        env.step(env.action_space.sample())
        travelled = np.linalg.norm(env.uav_xy - previous_xy, axis=1)

        assert np.all(travelled <= max_step_distance)
        assert np.all((env.uav_xy >= 0) & (env.uav_xy <= cfg.area_size_m))
        assert np.all(env.energy_j <= previous_energy)
        assert np.all(env.energy_j >= 0.0)

        previous_xy = env.uav_xy.copy()
        previous_energy = env.energy_j.copy()


def test_bandwidth_budget_is_respected(env: UAVEnv, cfg: UAVEnvConfig):
    """No UAV hands out more than its own band to its cluster."""
    env.reset(seed=3)
    _, _, _, _, info = env.step(env.action_space.sample())

    for i in range(cfg.n_uav):
        members = info["assignment"] == i
        allocated = info["bandwidth_hz"][members].sum()
        assert allocated <= cfg.bandwidth_hz + 1e-3


def test_episode_truncates_at_max_steps(cfg: UAVEnvConfig):
    """The episode ends after exactly max_steps when energy is plentiful."""
    env = UAVEnv(cfg)
    env.reset(seed=0)

    for step in range(1, cfg.max_steps + 1):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if step < cfg.max_steps:
            assert not (terminated or truncated)
    assert truncated


def test_episode_terminates_when_batteries_die():
    """A UAV fleet that runs out of energy ends the episode."""
    cfg = UAVEnvConfig(n_uav=2, n_users=6, initial_energy_j=200.0, max_steps=50)
    env = UAVEnv(cfg)
    env.reset(seed=0)

    terminated = False
    for _ in range(cfg.max_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    assert terminated
    assert np.all(env.energy_j <= 0.0)


# --------------------------------------------------------------------------
# Channel model
# --------------------------------------------------------------------------
def test_los_probability_is_bounded_and_increasing():
    """LoS probability lives in [0, 1] and grows with elevation angle."""
    angles = np.linspace(0.0, 90.0, 50)
    p = cm.los_probability(angles, **{k: cm.ENVIRONMENTS["urban"][k] for k in ("a", "b")})

    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) > 0.0)


def test_path_loss_grows_with_distance():
    """Moving a user away from the UAV can only increase path loss."""
    uav = np.array([0.0, 0.0])
    near = np.array([50.0, 0.0])
    far = np.array([900.0, 0.0])

    pl_near = cm.path_loss_db(uav, 100.0, near, 2e9)
    pl_far = cm.path_loss_db(uav, 100.0, far, 2e9)

    assert pl_far > pl_near


def test_capacity_is_monotone_in_bandwidth_and_sinr():
    """Shannon capacity increases with both bandwidth and SINR, and is 0 at B=0."""
    assert cm.shannon_capacity_bps(0.0, 10.0) == 0.0
    assert cm.shannon_capacity_bps(2e6, 10.0) > cm.shannon_capacity_bps(1e6, 10.0)
    assert cm.shannon_capacity_bps(1e6, 20.0) > cm.shannon_capacity_bps(1e6, 10.0)


def test_downlink_rates_shapes_and_positivity():
    """downlink_rates returns one finite, non-negative rate per user."""
    uav = np.array([[200.0, 200.0], [800.0, 800.0]])
    users = np.array([[150.0, 150.0], [820.0, 780.0], [500.0, 500.0]])

    rates, sinr = cm.downlink_rates(
        uav_xy=uav,
        altitudes=np.full(2, 100.0),
        user_xy=users,
        assignment=np.array([0, 1, 0]),
        bandwidth_hz=np.full(3, 5e6),
        tx_power_dbm=np.full(3, 30.0),
        carrier_hz=2e9,
    )

    assert rates.shape == (3,) and sinr.shape == (3,)
    assert np.all(np.isfinite(rates)) and np.all(rates >= 0.0)
    assert np.all(sinr > 0.0)


def test_unknown_environment_is_rejected():
    """A typo in the environment name fails loudly rather than silently."""
    with pytest.raises(KeyError):
        cm.path_loss_db(np.zeros(2), 100.0, np.ones(2), 2e9, environment="martian")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_jains_fairness_bounds():
    """Fairness is 1 for an equal split and 1/n when one user takes everything."""
    equal = np.full(8, 3e6)
    hogged = np.array([1e7, 0.0, 0.0, 0.0])

    assert metrics.jains_fairness(equal) == pytest.approx(1.0)
    assert metrics.jains_fairness(hogged) == pytest.approx(0.25)
    assert metrics.jains_fairness(np.zeros(5)) == pytest.approx(1.0)


def test_coverage_and_efficiency():
    """Coverage counts users above threshold; efficiency is bits per joule."""
    rates = np.array([2e6, 5e5, 4e6, 1e6])

    assert metrics.coverage_ratio(rates, 1e6) == pytest.approx(0.75)
    assert metrics.energy_efficiency(rates, 100.0) == pytest.approx(rates.sum() / 100.0)
    assert metrics.energy_efficiency(rates, 0.0) == 0.0


def test_summarize_reports_all_metrics(env: UAVEnv):
    """summarize() produces finite values for a real episode rollout."""
    env.reset(seed=0)
    rates, energy = [], []
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        rates.append(info["rates_bps"])
        energy.append(info["energy_used_j"])

    summary = metrics.summarize(np.asarray(rates), env.cfg.min_rate_bps, energy_j=energy)

    expected = {
        "total_throughput_mbps",
        "avg_rate_mbps",
        "jains_fairness",
        "coverage",
        "energy_efficiency_kbit_per_j",
    }
    assert expected <= set(summary)
    assert all(np.isfinite(v) for v in summary.values())


def test_metrics_reject_non_finite_input():
    """A NaN slipping into the rates is an error, not a silent NaN metric."""
    with pytest.raises(ValueError):
        metrics.average_data_rate(np.array([1e6, np.nan]))


# --------------------------------------------------------------------------
# K-means baseline
# --------------------------------------------------------------------------
def test_kmeans_action_is_within_action_space(env: UAVEnv):
    """The heuristic emits a legal action the environment accepts."""
    env.reset(seed=0)
    action = kmeans_action(env, seed=0)

    assert action.shape == (env.n_uav, env.action_dim)
    assert env.action_space.contains(action)

    _, reward, _, _, _ = env.step(action)
    assert np.isfinite(reward)


def test_uav_centroid_matching_is_one_to_one():
    """Every centroid is claimed by a distinct UAV."""
    uav = np.array([[0.0, 0.0], [100.0, 100.0], [500.0, 500.0]])
    centroids = np.array([[490.0, 495.0], [10.0, 5.0], [110.0, 90.0]])

    target = match_uavs_to_centroids(uav, centroids)

    assert sorted(target.tolist()) == [0, 1, 2]
    assert target.tolist() == [1, 2, 0]  # each UAV takes its nearest centroid


def test_kmeans_handles_fewer_users_than_uavs():
    """With more UAVs than users the clustering degrades gracefully."""
    env = UAVEnv(UAVEnvConfig(n_uav=4, n_users=2, max_steps=5))
    env.reset(seed=0)

    centroids, labels = kmeans_clusters(env.user_xy, env.n_uav, seed=0)
    assert centroids.shape[0] == 2
    assert set(labels.tolist()) == {0, 1}

    _, reward, _, _, _ = env.step(kmeans_action(env, seed=0))
    assert np.isfinite(reward)


def test_kmeans_beats_random_play(cfg: UAVEnvConfig):
    """The heuristic must outscore random actions, or the reward is degenerate."""
    env = UAVEnv(cfg)

    env.reset(seed=7)
    kmeans_return = 0.0
    for _ in range(cfg.max_steps):
        _, reward, terminated, truncated, _ = env.step(kmeans_action(env, seed=7))
        kmeans_return += reward
        if terminated or truncated:
            break

    env.reset(seed=7)
    random_return = 0.0
    for _ in range(cfg.max_steps):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        random_return += reward
        if terminated or truncated:
            break

    assert kmeans_return > random_return


# --------------------------------------------------------------------------
# Reward function
# --------------------------------------------------------------------------
def test_reward_terms_are_normalised(cfg: UAVEnvConfig):
    """Coverage and waste terms are fractions; the reward stays finite."""
    rates = np.array([5e6, 2e6, 1e5, 0.0])
    bandwidth = np.full(4, 5e6)

    reward, terms = compute_reward(rates, bandwidth, 500.0, cfg)

    assert np.isfinite(reward)
    assert 0.0 <= terms["coverage_term"] <= 1.0
    assert 0.0 <= terms["waste_term"] <= 1.0
    assert terms["coverage_term"] == pytest.approx(0.5)


def test_reward_prefers_better_service(cfg: UAVEnvConfig):
    """Serving everyone well beats serving them badly at equal cost."""
    bandwidth = np.full(6, 3e6)
    good, _ = compute_reward(np.full(6, 8e6), bandwidth, 500.0, cfg)
    bad, _ = compute_reward(np.full(6, 1e5), bandwidth, 500.0, cfg)

    assert good > bad


def test_shaping_penalises_flying_away_from_users():
    """Moving off the users costs shaping reward; moving onto them earns it."""
    cfg = UAVEnvConfig(n_uav=1, n_users=1, w_shaping=10.0, shaping_gamma=1.0)
    env = UAVEnv(cfg)
    action = np.zeros((1, env.action_dim), dtype=np.float32)

    env.reset(seed=0, options={"user_xy": [[500.0, 500.0]], "uav_xy": [[500.0, 500.0]]})
    action[0, 0] = 1.0
    *_, info_away = env.step(action)
    assert info_away["shaping_term"] < 0.0

    action[0, 0] = -1.0
    *_, info_back = env.step(action)
    assert info_back["shaping_term"] > 0.0
    assert np.isclose(info_away["shaping_term"], -info_back["shaping_term"])

