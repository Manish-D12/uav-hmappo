"""Plotting helpers for training curves and method comparisons.

All figures are written to disk with the non-interactive Agg backend, so the
scripts run fine over SSH or in a CI job with no display attached.  Importing
this module sets that backend, which is why it should be imported before
anything else touches pyplot.

Two families of figure live here:

* :func:`plot_convergence_curve` - how a single run learned over time.
* :func:`plot_metric_comparison` and :func:`plot_metric_distributions` - how
  several methods stack up on the fixed evaluation benchmark.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without needing a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Human-readable axis labels, and whether bigger is better.  Keeping this in
# one place stops the training curve and the comparison chart from labelling
# the same quantity two different ways.
METRIC_LABELS: dict[str, str] = {
    "return": "Episode return",
    "total_throughput_mbps": "Total throughput (Mbps)",
    "avg_rate_mbps": "Average user rate (Mbps)",
    "jains_fairness": "Jain's fairness index",
    "coverage": "Coverage (fraction of users)",
    "energy_efficiency_kbit_per_j": "Energy efficiency (kbit/J)",
    "resource_utilization": "Resource utilisation",
    "spectral_efficiency_bps_hz": "Spectral efficiency (bps/Hz)",
}

# A stable colour per method, so a method keeps its colour across every figure.
METHOD_COLORS: dict[str, str] = {
    "K-means": "tab:red",
    "Flat MAPPO": "tab:orange",
    "Hierarchical MAPPO": "tab:blue",
}

_FALLBACK_COLORS = ["tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]


def _label(metric: str) -> str:
    """Human-readable label for a metric column, falling back to the raw name."""
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _color(method: str, index: int) -> str:
    """Stable colour for a method name."""
    return METHOD_COLORS.get(method, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def _grid_shape(n_panels: int, max_cols: int = 4) -> tuple[int, int]:
    """Rows and columns for a roughly square panel grid."""
    cols = min(max_cols, max(1, n_panels))
    rows = int(np.ceil(n_panels / cols))
    return rows, cols


def _ensure_parent(path: str) -> None:
    """Create the directory a figure is about to be written into."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def plot_convergence_curve(
    train_log: pd.DataFrame,
    eval_log: pd.DataFrame | None,
    baseline_return: float | None,
    path: str,
    agent_name: str = "MAPPO",
    window: int = 10,
    metrics: tuple[str, ...] = (
        "return",
        "coverage",
        "avg_rate_mbps",
        "jains_fairness",
    ),
) -> str:
    """Plot the learning curve alongside the K-means reference.

    Several panels rather than one, because a rising return on its own does
    not tell you *which* objective improved - a policy can buy return by
    trading fairness for throughput, and that should be visible.

    Parameters
    ----------
    train_log : DataFrame
        One row per training episode; must contain an ``episode`` column.
    eval_log : DataFrame or None
        Periodic evaluation rows on the fixed benchmark.
    baseline_return : float or None
        Drawn as a horizontal reference line on the return panel.
    path : str
        Output PNG path.
    agent_name : str
        Used in the figure title.
    window : int
        Rolling-mean window; PPO returns are noisy enough that the raw trace
        alone is hard to read.
    metrics : tuple of str
        Which columns to panel.

    Returns
    -------
    str
        The path written.
    """
    _ensure_parent(path)
    available = [m for m in metrics if m in train_log.columns]
    if not available:
        raise ValueError(f"none of {metrics} are columns of train_log")

    rows, cols = _grid_shape(len(available), max_cols=2)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows),
                             constrained_layout=True, squeeze=False)
    flat_axes = axes.flatten()

    for ax, metric in zip(flat_axes, available):
        ax.plot(train_log["episode"], train_log[metric],
                alpha=0.25, color="tab:blue", label="per episode")

        if len(train_log) >= window:
            smoothed = train_log[metric].rolling(window, min_periods=1).mean()
            ax.plot(train_log["episode"], smoothed, color="tab:blue",
                    linewidth=2, label=f"rolling mean ({window})")

        if eval_log is not None and not eval_log.empty and metric in eval_log.columns:
            ax.plot(eval_log["episode"], eval_log[metric], "o-", color="tab:green",
                    markersize=4, label="eval (fixed scenarios)")

        if metric == "return" and baseline_return is not None:
            ax.axhline(baseline_return, color="tab:red", linestyle="--",
                       linewidth=1.5, label="K-means baseline")

        ax.set_xlabel("episode")
        ax.set_ylabel(_label(metric))
        ax.set_title(_label(metric))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in flat_axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle(f"{agent_name} - UAV deployment, clustering and resource allocation")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def plot_metric_comparison(
    summary: pd.DataFrame,
    path: str,
    metrics: list[str] | None = None,
    error_column_suffix: str = "_std",
    title: str = "Method comparison on the fixed evaluation benchmark",
) -> str:
    """One bar chart per metric, with a bar per method.

    Error bars show the standard deviation across evaluation scenarios.  They
    matter more than they might seem: with a spread this wide, two methods
    whose means differ can still be statistically indistinguishable, and a
    bare bar chart would hide that.

    Parameters
    ----------
    summary : DataFrame
        Indexed by method name.  For a metric column ``m``, an optional
        companion column ``m + error_column_suffix`` supplies the error bar.
    path : str
        Output PNG path.
    metrics : list of str, optional
        Columns to plot; defaults to every non-error numeric column.
    error_column_suffix : str
        Suffix identifying the error columns.
    title : str
        Figure title.

    Returns
    -------
    str
        The path written.
    """
    _ensure_parent(path)

    if metrics is None:
        metrics = [
            c for c in summary.columns
            if not c.endswith(error_column_suffix)
            and pd.api.types.is_numeric_dtype(summary[c])
        ]
    if not metrics:
        raise ValueError("no numeric metric columns to plot")

    methods = list(summary.index)
    rows, cols = _grid_shape(len(metrics), max_cols=4)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                             constrained_layout=True, squeeze=False)
    flat_axes = axes.flatten()

    for ax, metric in zip(flat_axes, metrics):
        values = summary[metric].to_numpy(dtype=float)
        error_column = f"{metric}{error_column_suffix}"
        errors = (
            summary[error_column].to_numpy(dtype=float)
            if error_column in summary.columns else None
        )
        colors = [_color(m, i) for i, m in enumerate(methods)]

        bars = ax.bar(range(len(methods)), values, yerr=errors, capsize=4,
                      color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)

        # Print the value on each bar; reading exact numbers off an axis is
        # unreliable, and these figures go straight into a report.
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3g}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_title(_label(metric), fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        # Keep zero in frame so bar lengths stay honest.
        if np.all(values >= 0):
            ax.set_ylim(bottom=0)

    for ax in flat_axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_distributions(
    per_scenario: pd.DataFrame,
    path: str,
    metrics: list[str] | None = None,
    method_column: str = "method",
    title: str = "Per-scenario distributions on the fixed evaluation benchmark",
) -> str:
    """Box plots of each metric across the evaluation scenarios.

    The companion to :func:`plot_metric_comparison`: means say which method is
    ahead, distributions say whether it is ahead *consistently* or just on
    average.

    Parameters
    ----------
    per_scenario : DataFrame
        One row per (method, scenario), with a column naming the method.
    path : str
        Output PNG path.
    metrics : list of str, optional
        Columns to plot; defaults to every numeric column except the scenario
        index.
    method_column : str
        Column holding the method name.
    title : str
        Figure title.

    Returns
    -------
    str
        The path written.
    """
    _ensure_parent(path)

    if metrics is None:
        metrics = [
            c for c in per_scenario.columns
            if c not in (method_column, "scenario")
            and pd.api.types.is_numeric_dtype(per_scenario[c])
        ]
    if not metrics:
        raise ValueError("no numeric metric columns to plot")

    methods = list(dict.fromkeys(per_scenario[method_column]))
    rows, cols = _grid_shape(len(metrics), max_cols=4)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                             constrained_layout=True, squeeze=False)
    flat_axes = axes.flatten()

    for ax, metric in zip(flat_axes, metrics):
        data = [
            per_scenario.loc[per_scenario[method_column] == m, metric].to_numpy(dtype=float)
            for m in methods
        ]
        boxes = ax.boxplot(data, patch_artist=True, widths=0.6,
                           medianprops={"color": "black"})
        for patch, (i, method) in zip(boxes["boxes"], enumerate(methods)):
            patch.set_facecolor(_color(method, i))
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_title(_label(metric), fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    for ax in flat_axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
