"""A fixed benchmark of evaluation scenarios.

Comparing a learned policy against K-means is only meaningful if both face
the *same* problems.  Sampling fresh random user layouts at evaluation time
adds variance that can easily exceed the difference between the methods - a
method can look better simply because it drew easier layouts.

This module pins the benchmark down: a set of user layouts (and matching UAV
start positions) is generated once, written to an ``.npz`` file, and replayed
by every method through :meth:`envs.uav_env.UAVEnv.reset`'s ``options``
argument.  The file also stores the scenario dimensions it was built for, so
that silently evaluating on a mismatched scenario set is impossible.

Typical use::

    scenarios = ensure_scenarios("results/eval_scenarios.npz", cfg, n=50)
    for options in iter_options(scenarios):
        obs, info = env.reset(options=options)
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np

from envs.uav_env import UAVEnvConfig


def generate_scenarios(
    cfg: UAVEnvConfig, n_scenarios: int = 50, seed: int = 12345
) -> dict[str, np.ndarray]:
    """Sample a fresh benchmark of user layouts and UAV start positions.

    The sampling mirrors :meth:`UAVEnv.reset` exactly - uniform users, UAVs
    jittered around the centre - so the benchmark is drawn from the same
    distribution the agent trains on.

    Parameters
    ----------
    cfg : UAVEnvConfig
        Scenario dimensions to generate for.
    n_scenarios : int
        How many layouts to create.
    seed : int
        Seed for the benchmark itself.  Deliberately distinct from the
        training seed so that evaluation layouts are not a subset of the ones
        the agent trained on.

    Returns
    -------
    dict
        ``user_xy`` of shape ``(n_scenarios, n_users, 2)`` and ``uav_xy`` of
        shape ``(n_scenarios, n_uav, 2)``.
    """
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")

    rng = np.random.default_rng(seed)
    size = cfg.area_size_m

    user_xy = rng.uniform(0.0, size, size=(n_scenarios, cfg.n_users, 2))
    centre = np.full((n_scenarios, cfg.n_uav, 2), size / 2.0)
    jitter = rng.uniform(-0.05 * size, 0.05 * size, size=(n_scenarios, cfg.n_uav, 2))
    uav_xy = np.clip(centre + jitter, 0.0, size)

    return {"user_xy": user_xy, "uav_xy": uav_xy}


def save_scenarios(path: str, scenarios: dict[str, np.ndarray], cfg: UAVEnvConfig) -> None:
    """Write a benchmark to ``path``, stamped with the config it belongs to."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    np.savez(
        path,
        user_xy=scenarios["user_xy"],
        uav_xy=scenarios["uav_xy"],
        n_uav=cfg.n_uav,
        n_users=cfg.n_users,
        area_size_m=cfg.area_size_m,
    )


def load_scenarios(path: str, cfg: UAVEnvConfig | None = None) -> dict[str, np.ndarray]:
    """Load a benchmark, optionally checking it matches the current config.

    Raises
    ------
    ValueError
        If ``cfg`` is given and the stored scenario dimensions differ from it.
        Failing loudly here prevents the quiet disaster of comparing two
        methods that were evaluated on different benchmarks.
    """
    with np.load(path) as data:
        scenarios = {"user_xy": data["user_xy"], "uav_xy": data["uav_xy"]}
        stored = {
            "n_uav": int(data["n_uav"]),
            "n_users": int(data["n_users"]),
            "area_size_m": float(data["area_size_m"]),
        }

    if cfg is not None:
        current = {
            "n_uav": cfg.n_uav,
            "n_users": cfg.n_users,
            "area_size_m": float(cfg.area_size_m),
        }
        if stored != current:
            raise ValueError(
                f"scenario file {path!r} was built for {stored}, but the current "
                f"config is {current}. Delete the file to regenerate it, or point "
                f"--eval-scenarios somewhere else."
            )
    return scenarios


def ensure_scenarios(
    path: str, cfg: UAVEnvConfig, n_scenarios: int = 50, seed: int = 12345
) -> dict[str, np.ndarray]:
    """Load the benchmark at ``path``, generating and saving it if absent.

    Calling this from every entry point means the benchmark is created once
    and then reused verbatim by training, evaluation and the baselines.
    """
    if os.path.exists(path):
        return load_scenarios(path, cfg)

    scenarios = generate_scenarios(cfg, n_scenarios, seed)
    save_scenarios(path, scenarios, cfg)
    return scenarios


def n_scenarios(scenarios: dict[str, np.ndarray]) -> int:
    """Number of scenarios held in a benchmark dict."""
    return int(scenarios["user_xy"].shape[0])


def scenario_options(scenarios: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    """Build the ``options`` dict for one scenario, ready for ``env.reset``."""
    return {
        "user_xy": scenarios["user_xy"][index],
        "uav_xy": scenarios["uav_xy"][index],
    }


def iter_options(
    scenarios: dict[str, np.ndarray], limit: int | None = None
) -> Iterator[dict[str, np.ndarray]]:
    """Iterate over ``reset`` options for each scenario.

    Parameters
    ----------
    scenarios : dict
        A benchmark as returned by :func:`ensure_scenarios`.
    limit : int, optional
        Use only the first ``limit`` scenarios - handy for the quick periodic
        evaluations during training, where the full benchmark is too slow.
    """
    total = n_scenarios(scenarios)
    count = total if limit is None else min(limit, total)
    for i in range(count):
        yield scenario_options(scenarios, i)
