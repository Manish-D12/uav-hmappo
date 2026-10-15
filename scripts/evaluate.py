"""Final comparison of every method on the fixed evaluation benchmark.

Runs three methods over the *same* saved user layouts and reports the full
metric suite for each:

* **K-means** - the non-learning geometric heuristic.
* **Flat MAPPO** - one policy controlling all three action types at once.
* **Hierarchical MAPPO** - slow strategic level (deployment + clustering)
  over a fast tactical level (resource allocation).

Using the shared benchmark from :mod:`utils.scenarios` is the whole point: a
method must not be allowed to look better simply because it drew easier user
layouts.  Every method sees scenario 0, scenario 1, ... in the same order,
with identical user positions and identical UAV start positions.

Outputs (into ``results/`` by default)::

    comparison.csv            mean +/- std per method and metric
    comparison_per_scenario.csv   one row per (method, scenario)
    comparison_bars.png       bar chart per metric
    comparison_boxes.png      per-scenario distribution per metric

Examples
--------
Evaluate whatever ``train.py`` last saved::

    python scripts/evaluate.py

Point at a specific run and use fewer scenarios::

    python scripts/evaluate.py --model-dir results --n-scenarios 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Allow `python scripts/evaluate.py` to find the project packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.hierarchical import HierarchicalMAPPO  # noqa: E402
from agents.mappo import MAPPO  # noqa: E402
from baselines.kmeans_baseline import run_episode as kmeans_episode  # noqa: E402
from envs.uav_env import UAVEnv, UAVEnvConfig  # noqa: E402
from utils import metrics, scenarios as scen  # noqa: E402
from utils.config import build_agent, build_dataclass, load_config  # noqa: E402
from utils.plotting import plot_metric_comparison, plot_metric_distributions  # noqa: E402

# Reported in this order, which is also the column order of the table.
METRIC_COLUMNS = [
    "return",
    "total_throughput_mbps",
    "avg_rate_mbps",
    "jains_fairness",
    "coverage",
    "energy_efficiency_kbit_per_j",
    "resource_utilization",
    "spectral_efficiency_bps_hz",
]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_agent(
    env: UAVEnv, config: dict, checkpoint: str, hierarchical: bool
) -> MAPPO | HierarchicalMAPPO | None:
    """Rebuild an agent and restore its weights, or return None if unavailable.

    A missing checkpoint is reported and skipped rather than raised: it is
    normal to have trained only one of the two variants so far, and that
    should not stop the other two methods from being compared.

    Parameters
    ----------
    env : UAVEnv
        Supplies the network dimensions.
    config : dict
        Parsed YAML, used to rebuild the identical architecture.
    checkpoint : str
        Path passed to the agent's ``load``.  The hierarchical agent appends
        ``.high`` and ``.low`` to it.
    hierarchical : bool
        Which variant to build.

    Returns
    -------
    MAPPO, HierarchicalMAPPO or None
    """
    required = (
        [f"{checkpoint}.high", f"{checkpoint}.low"] if hierarchical else [checkpoint]
    )
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"  ! skipping: checkpoint not found ({', '.join(missing)})")
        return None

    agent = build_agent(env, config, hierarchical=hierarchical)
    try:
        agent.load(checkpoint)
    except (RuntimeError, KeyError) as exc:
        # Almost always an architecture mismatch between the config used to
        # train and the one being used to evaluate.
        print(f"  ! skipping: could not load {checkpoint} ({type(exc).__name__}: {exc})")
        return None
    return agent


# --------------------------------------------------------------------------
# Per-method evaluation
# --------------------------------------------------------------------------
def _summarize_episode(
    env: UAVEnv,
    rates: np.ndarray,
    bandwidth: np.ndarray,
    energy: np.ndarray | list,
    episode_return: float,
) -> dict[str, float]:
    """Apply the full metric suite to one finished episode."""
    summary = metrics.summarize(
        np.asarray(rates),
        env.cfg.min_rate_bps,
        energy_j=energy,
        bandwidth_hz=np.asarray(bandwidth),
        available_per_step_hz=env.n_uav * env.cfg.bandwidth_hz,
    )
    summary["return"] = float(episode_return)
    return summary


def evaluate_policy(
    env: UAVEnv,
    agent: MAPPO | HierarchicalMAPPO,
    scenarios: dict[str, np.ndarray],
    limit: int | None = None,
) -> list[dict[str, float]]:
    """Run a trained agent greedily over the benchmark, one row per scenario.

    Acting deterministically (distribution modes rather than samples) is what
    makes the comparison reproducible; sampling would add policy noise on top
    of the scenario variation the benchmark is designed to remove.
    """
    rows = []
    for index, options in enumerate(scen.iter_options(scenarios, limit)):
        obs, info = env.reset(options=options)
        state = info["state"]
        agent.begin_episode()

        episode_return = 0.0
        rates, bandwidth, energy = [], [], []
        terminated = truncated = False

        while not (terminated or truncated):
            # learn=False equivalent: nothing is recorded into the buffer.
            env_action = agent.act(obs, state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(env_action)
            state = info["state"]
            episode_return += reward
            rates.append(info["rates_bps"])
            bandwidth.append(info["bandwidth_hz"])
            energy.append(info["energy_used_j"])

        row = _summarize_episode(env, rates, bandwidth, energy, episode_return)
        row["scenario"] = index
        rows.append(row)
    return rows


def evaluate_kmeans(
    env: UAVEnv,
    scenarios: dict[str, np.ndarray],
    limit: int | None = None,
    power_level: float = 1.0,
) -> list[dict[str, float]]:
    """Run the K-means heuristic over the benchmark, one row per scenario."""
    rows = []
    for index, options in enumerate(scen.iter_options(scenarios, limit)):
        result = kmeans_episode(env, seed=index, options=options, power_level=power_level)
        row = _summarize_episode(
            env,
            result["rates_bps"],
            result["bandwidth_hz"],
            result["energy_used_j"],
            result["return"],
        )
        row["scenario"] = index
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------
def build_summary(per_scenario: pd.DataFrame, method_column: str = "method") -> pd.DataFrame:
    """Aggregate per-scenario rows into mean and standard deviation per method.

    Returns
    -------
    DataFrame
        Indexed by method, with a ``<metric>`` and a ``<metric>_std`` column
        for every metric, in :data:`METRIC_COLUMNS` order.
    """
    available = [c for c in METRIC_COLUMNS if c in per_scenario.columns]
    grouped = per_scenario.groupby(method_column, sort=False)[available]

    means = grouped.mean()
    stds = grouped.std(ddof=0).add_suffix("_std")

    summary = means.join(stds)
    # Interleave each metric with its own error column so the CSV reads well.
    ordered = [c for metric in available for c in (metric, f"{metric}_std")]
    return summary[ordered]


def print_table(summary: pd.DataFrame) -> None:
    """Print the comparison as a fixed-width table of ``mean +/- std``."""
    methods = list(summary.index)
    metrics_present = [c for c in summary.columns if not c.endswith("_std")]

    name_width = max(len("metric"), max(len(m) for m in metrics_present))
    col_width = max(18, max(len(m) for m in methods) + 2)

    header = "metric".ljust(name_width) + "".join(m.rjust(col_width) for m in methods)
    print("\n" + header)
    print("-" * len(header))

    for metric in metrics_present:
        line = metric.ljust(name_width)
        for method in methods:
            mean = summary.loc[method, metric]
            std = summary.loc[method, f"{metric}_std"]
            line += f"{mean:>9.3f} +/-{std:<6.2f}".rjust(col_width)
        print(line)

    if len(methods) < 2:
        return

    # Name the winner per metric.  Higher is better for every metric in this
    # suite, so idxmax is the winner throughout - that would need revisiting
    # if a cost-style metric (latency, energy *consumed*) were ever added.
    print("-" * len(header))
    print("best per metric:")
    for metric in metrics_present:
        winner = summary[metric].idxmax()
        print(f"  {metric.ljust(name_width)}  {winner}")


def run_evaluation(config: dict, args: argparse.Namespace) -> pd.DataFrame:
    """Evaluate every available method and write the table and figures."""
    env_cfg = build_dataclass(UAVEnvConfig, config.get("env"))
    if args.max_steps is not None:
        env_cfg.max_steps = args.max_steps

    eval_cfg = config.get("eval", {}) or {}
    train_cfg = config.get("train", {}) or {}

    model_dir = args.model_dir or str(train_cfg.get("log_dir", "results"))
    out_dir = args.out_dir or model_dir
    os.makedirs(out_dir, exist_ok=True)

    scenario_path = args.scenarios or str(eval_cfg.get("path", "results/eval_scenarios.npz"))
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(
            f"no evaluation benchmark at {scenario_path!r}. Run scripts/train.py first, "
            f"or pass --scenarios."
        )
    # Passing env_cfg makes load_scenarios verify the benchmark matches this
    # scenario size, rather than silently comparing incomparable runs.
    scenarios = scen.load_scenarios(scenario_path, env_cfg)
    limit = args.n_scenarios or scen.n_scenarios(scenarios)
    limit = min(limit, scen.n_scenarios(scenarios))

    env = UAVEnv(env_cfg)

    print(f"scenario  : {env_cfg.n_uav} UAVs, {env_cfg.n_users} users, "
          f"{env_cfg.max_steps} steps/episode")
    print(f"benchmark : {limit} of {scen.n_scenarios(scenarios)} fixed scenarios "
          f"from {scenario_path}")
    print(f"models    : {model_dir}\n")

    rows: list[dict] = []

    # --- 1. K-means baseline -------------------------------------------
    print("K-means baseline")
    for row in evaluate_kmeans(env, scenarios, limit, power_level=args.kmeans_power):
        rows.append({**row, "method": "K-means"})

    # --- 2. flat MAPPO ---------------------------------------------------
    print("Flat MAPPO")
    flat_ckpt = args.flat_model or os.path.join(model_dir, "model_flat.pt")
    flat_agent = load_agent(env, config, flat_ckpt, hierarchical=False)
    if flat_agent is not None:
        for row in evaluate_policy(env, flat_agent, scenarios, limit):
            rows.append({**row, "method": "Flat MAPPO"})

    # --- 3. hierarchical MAPPO ------------------------------------------
    print("Hierarchical MAPPO")
    hier_ckpt = args.hier_model or os.path.join(model_dir, "model.pt")
    hier_agent = load_agent(env, config, hier_ckpt, hierarchical=True)
    if hier_agent is not None:
        for row in evaluate_policy(env, hier_agent, scenarios, limit):
            rows.append({**row, "method": "Hierarchical MAPPO"})

    env.close()

    if not rows:
        raise RuntimeError("no method produced any results - are the checkpoints missing?")

    per_scenario = pd.DataFrame(rows)
    ordered_columns = ["method", "scenario"] + [
        c for c in METRIC_COLUMNS if c in per_scenario.columns
    ]
    per_scenario = per_scenario[ordered_columns]
    summary = build_summary(per_scenario)

    print_table(summary)

    # ---------------------------------------------------------------- output
    summary_path = os.path.join(out_dir, "comparison.csv")
    summary.to_csv(summary_path)

    detail_path = os.path.join(out_dir, "comparison_per_scenario.csv")
    per_scenario.to_csv(detail_path, index=False)

    bars_path = plot_metric_comparison(summary, os.path.join(out_dir, "comparison_bars.png"))
    boxes_path = plot_metric_distributions(
        per_scenario, os.path.join(out_dir, "comparison_boxes.png")
    )

    print("\nwritten:")
    for path in (summary_path, detail_path, bars_path, boxes_path):
        print(f"  {path}")

    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options for the evaluation run."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(here, "configs", "default.yaml"))
    parser.add_argument("--model-dir", default=None,
                        help="directory holding the checkpoints (default: train.log_dir)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the table and figures (default: --model-dir)")
    parser.add_argument("--scenarios", default=None,
                        help="path to the fixed evaluation benchmark")
    parser.add_argument("--n-scenarios", type=int, default=None,
                        help="evaluate on only the first N scenarios")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the environment episode length")
    parser.add_argument("--flat-model", default=None,
                        help="checkpoint for the flat agent (default: <model-dir>/model_flat.pt)")
    parser.add_argument("--hier-model", default=None,
                        help="checkpoint stem for the hierarchical agent "
                             "(default: <model-dir>/model.pt)")
    parser.add_argument("--kmeans-power", type=float, default=1.0,
                        help="transmit power level for the K-means baseline, in [0, 1]")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the config and run the comparison."""
    args = parse_args(argv)
    run_evaluation(load_config(args.config), args)


if __name__ == "__main__":
    main()
