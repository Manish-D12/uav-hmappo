"""Training entry point for hierarchical MAPPO on the UAV network task.

Reads ``configs/default.yaml``, trains an agent, and writes everything needed
for the results section into ``results/``:

===========================  ==============================================
file                         contents
===========================  ==============================================
``train_log.csv``            one row per training episode
``eval_log.csv``             one row per periodic evaluation
``convergence.png``          learning curve with the K-means reference line
``eval_scenarios.npz``       the fixed benchmark every method is scored on
``model.pt`` (``.high``/``.low``)  final network weights
``model_best.pt`` (...)      weights at the best periodic evaluation
===========================  ==============================================

Evaluation always runs on the fixed benchmark from :mod:`utils.scenarios`,
never on freshly sampled layouts, so the learned policy and the K-means
baseline are compared on identical user placements.

Examples
--------
Full run with the defaults::

    python scripts/train.py

A quick end-to-end check::

    python scripts/train.py --episodes 6 --max-steps 20 --update-every 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Allow `python scripts/train.py` to find the project packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.hierarchical import HierarchicalMAPPO  # noqa: E402
from agents.mappo import MAPPO  # noqa: E402
from baselines.kmeans_baseline import run_episode as kmeans_episode  # noqa: E402
from envs.uav_env import UAVEnv, UAVEnvConfig  # noqa: E402
from utils import metrics, scenarios as scen  # noqa: E402
from utils.config import build_agent, build_dataclass, load_config  # noqa: E402
from utils.plotting import plot_convergence_curve  # noqa: E402


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------
def rollout(
    env: UAVEnv,
    agent: MAPPO | HierarchicalMAPPO,
    seed: int | None = None,
    options: dict | None = None,
    deterministic: bool = False,
    learn: bool = True,
) -> dict[str, float]:
    """Run one episode and return its aggregate metrics.

    Parameters
    ----------
    env, agent
    seed : int, optional
        Episode seed.
    options : dict, optional
        Fixed-scenario options for :meth:`UAVEnv.reset`.
    deterministic : bool
        Use distribution modes rather than sampling (evaluation).
    learn : bool
        When False, nothing is written to the agent's rollout buffer, so
        evaluation episodes never contaminate the on-policy data.

    Returns
    -------
    dict
        Episode return, length, and the metrics from :mod:`utils.metrics`.
    """
    obs, info = env.reset(seed=seed, options=options)
    state = info["state"]
    agent.begin_episode()

    episode_return = 0.0
    rates, bandwidth, energy = [], [], []
    terminated = truncated = False

    while not (terminated or truncated):
        env_action = agent.act(obs, state, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(env_action)
        next_state = info["state"]

        if learn:
            agent.record(reward, terminated, truncated, next_state)

        state = next_state
        episode_return += reward
        rates.append(info["rates_bps"])
        bandwidth.append(info["bandwidth_hz"])
        energy.append(info["energy_used_j"])

    summary = metrics.summarize(
        np.asarray(rates),
        env.cfg.min_rate_bps,
        energy_j=energy,
        bandwidth_hz=np.asarray(bandwidth),
        available_per_step_hz=env.n_uav * env.cfg.bandwidth_hz,
    )
    summary["return"] = float(episode_return)
    summary["steps"] = len(rates)
    summary["_last_obs"] = obs
    summary["_last_state"] = state
    return summary


def evaluate_agent(
    env: UAVEnv,
    agent: MAPPO | HierarchicalMAPPO,
    scenarios: dict[str, np.ndarray],
    limit: int | None = None,
) -> dict[str, float]:
    """Score the agent greedily on the fixed benchmark."""
    rows = [
        rollout(env, agent, options=options, deterministic=True, learn=False)
        for options in scen.iter_options(scenarios, limit)
    ]
    return _mean_metrics(rows)


def evaluate_kmeans(
    env: UAVEnv, scenarios: dict[str, np.ndarray], limit: int | None = None
) -> dict[str, float]:
    """Score the K-means baseline on the same fixed benchmark."""
    rows = []
    for i, options in enumerate(scen.iter_options(scenarios, limit)):
        result = kmeans_episode(env, seed=i, options=options)
        summary = metrics.summarize(
            result["rates_bps"],
            env.cfg.min_rate_bps,
            energy_j=result["energy_used_j"],
            bandwidth_hz=result["bandwidth_hz"],
            available_per_step_hz=env.n_uav * env.cfg.bandwidth_hz,
        )
        summary["return"] = float(result["return"])
        summary["steps"] = int(result["steps"])
        rows.append(summary)
    return _mean_metrics(rows)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-episode metric dicts, dropping private keys."""
    keys = [k for k in rows[0] if not k.startswith("_")]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train(config: dict, args: argparse.Namespace) -> dict[str, object]:
    """Run the full training loop and write all artefacts to the log directory."""
    train_cfg = config.get("train", {}) or {}
    eval_cfg = config.get("eval", {}) or {}

    env_cfg = build_dataclass(UAVEnvConfig, config.get("env"))
    if args.max_steps is not None:
        env_cfg.max_steps = args.max_steps

    episodes = args.episodes or int(train_cfg.get("episodes", 500))
    update_every = args.update_every or int(train_cfg.get("update_every", 4))
    eval_every = args.eval_every or int(train_cfg.get("eval_every", 50))
    eval_episodes = int(train_cfg.get("eval_episodes", 5))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 0))
    log_dir = args.log_dir or str(train_cfg.get("log_dir", "results"))
    os.makedirs(log_dir, exist_ok=True)

    if args.flat:
        config = {**config, "hierarchical": {**(config.get("hierarchical") or {}), "enabled": False}}

    # Reproducibility: seed torch and numpy, not just the environment.
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = UAVEnv(env_cfg)
    agent = build_agent(env, config)
    hierarchical = isinstance(agent, HierarchicalMAPPO)

    # The fixed benchmark: generated once, reused by every method.
    scenario_path = args.eval_scenarios or str(eval_cfg.get("path", "results/eval_scenarios.npz"))
    scenarios = scen.ensure_scenarios(
        scenario_path,
        env_cfg,
        n_scenarios=args.n_eval_scenarios or int(eval_cfg.get("n_scenarios", 50)),
        seed=int(eval_cfg.get("seed", 12345)),
    )

    print(f"agent          : {'hierarchical MAPPO' if hierarchical else 'flat MAPPO'}")
    print(f"scenario       : {env_cfg.n_uav} UAVs, {env_cfg.n_users} users, "
          f"{env_cfg.max_steps} steps/episode")
    print(f"training       : {episodes} episodes, update every {update_every}")
    print(f"eval benchmark : {scen.n_scenarios(scenarios)} fixed scenarios at {scenario_path}")

    # Reference line: the baseline on the same benchmark, computed once.
    baseline = evaluate_kmeans(env, scenarios, limit=eval_episodes)
    print(f"K-means on the first {eval_episodes} scenarios: "
          f"return {baseline['return']:.2f}, coverage {baseline['coverage']:.1%}, "
          f"rate {baseline['avg_rate_mbps']:.2f} Mbps\n")

    # Distinct names per variant, so training both into one directory does not
    # overwrite either, and scripts/evaluate.py can find both by default.
    model_path = os.path.join(log_dir, "model.pt" if hierarchical else "model_flat.pt")
    best_path = model_path.replace(".pt", "_best.pt")
    best_return = -np.inf

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    started = time.time()

    try:
        for episode in range(1, episodes + 1):
            result = rollout(env, agent, seed=seed + episode, learn=True)

            row = {k: v for k, v in result.items() if not k.startswith("_")}
            row["episode"] = episode
            train_rows.append(row)

            # PPO is on-policy: update on a whole number of episodes, then the
            # buffer is cleared and the next batch is collected under the new
            # policy.
            if episode % update_every == 0:
                agent.set_lr_fraction(1.0 - (episode - update_every) / episodes)
                stats = agent.finish(result["_last_obs"], result["_last_state"])
                train_rows[-1].update(stats)

            if episode % eval_every == 0 or episode == episodes:
                ev = evaluate_agent(env, agent, scenarios, limit=eval_episodes)
                ev["episode"] = episode
                eval_rows.append(ev)
                print(
                    f"[eval @ ep {episode:>4d}] return {ev['return']:>8.2f} "
                    f"(baseline {baseline['return']:>8.2f}) | "
                    f"coverage {ev['coverage']:>6.1%} | "
                    f"rate {ev['avg_rate_mbps']:>5.2f} Mbps | "
                    f"fairness {ev['jains_fairness']:.3f}"
                )
                # PPO does not improve monotonically, so the last policy is
                # often not the best one.  Keep the best-scoring weights too.
                if train_cfg.get("save_model", True) and ev["return"] > best_return:
                    best_return = ev["return"]
                    agent.save(best_path)
                    print(f"  new best eval return {best_return:.2f} -> {best_path}")
            elif episode % max(1, eval_every // 5) == 0:
                recent = np.mean([r["return"] for r in train_rows[-10:]])
                print(f"  ep {episode:>4d} | train return (last 10) {recent:>8.2f}")

    except KeyboardInterrupt:
        print("\ninterrupted - writing partial results")

    elapsed = time.time() - started
    env.close()

    # ---------------------------------------------------------------- output
    train_log = pd.DataFrame(train_rows)
    eval_log = pd.DataFrame(eval_rows)

    train_path = os.path.join(log_dir, "train_log.csv")
    train_log.to_csv(train_path, index=False)

    eval_path = os.path.join(log_dir, "eval_log.csv")
    if not eval_log.empty:
        eval_log.to_csv(eval_path, index=False)

    curve_path = os.path.join(log_dir, "convergence.png")
    if not train_log.empty:
        plot_convergence_curve(
            train_log, eval_log, baseline["return"], curve_path,
            agent_name="Hierarchical MAPPO" if hierarchical else "Flat MAPPO",
        )

    if train_cfg.get("save_model", True):
        agent.save(model_path)

    print(f"\ndone in {elapsed:.1f}s")
    print(f"  {train_path}")
    if not eval_log.empty:
        print(f"  {eval_path}")
    print(f"  {curve_path}")
    print(f"  {scenario_path}")
    if train_cfg.get("save_model", True):
        suffix = "{.high,.low}" if hierarchical else ""
        print(f"  {model_path}{suffix}")
        if np.isfinite(best_return):
            print(f"  {best_path}{suffix}  (best eval return {best_return:.2f})")

    return {"train_log": train_log, "eval_log": eval_log, "baseline": baseline}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line overrides for the YAML config."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(here, "configs", "default.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the environment episode length")
    parser.add_argument("--update-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--flat", action="store_true",
                        help="use flat MAPPO instead of the hierarchical agent")
    parser.add_argument("--eval-scenarios", default=None,
                        help="path to the fixed evaluation benchmark")
    parser.add_argument("--n-eval-scenarios", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the config and start training."""
    args = parse_args(argv)
    train(load_config(args.config), args)


if __name__ == "__main__":
    main()
