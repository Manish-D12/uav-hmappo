"""Performance metrics for UAV network evaluation.

These are the numbers reported in the results section: raw capacity
(throughput, average rate), how evenly that capacity is shared (Jain's
fairness), how many users are actually served (coverage), and what it cost
to deliver (energy efficiency).

Every function accepts either a single snapshot of per-user rates with shape
``(M,)`` or a whole episode with shape ``(T, M)``, and always returns a plain
Python float so results drop straight into a pandas DataFrame.
"""

from __future__ import annotations

import numpy as np


def _as_rates(rates_bps: np.ndarray) -> np.ndarray:
    """Validate and flatten a rate array to 1-D, rejecting non-finite values."""
    rates = np.asarray(rates_bps, dtype=float).reshape(-1)
    if rates.size == 0:
        raise ValueError("rates_bps is empty")
    if not np.all(np.isfinite(rates)):
        raise ValueError("rates_bps contains non-finite values")
    return rates


def _n_steps(rates_bps: np.ndarray) -> int:
    """Number of timesteps: rows of a ``(T, M)`` episode, 1 for a snapshot."""
    rates = np.asarray(rates_bps)
    return rates.shape[0] if rates.ndim == 2 else 1


def total_throughput(rates_bps: np.ndarray) -> float:
    """Network throughput per step, in bits per second.

    The sum of the per-user rates.  For an episode of shape ``(T, M)`` the
    per-step sums are averaged over the ``T`` steps, so the result is the
    typical instantaneous network throughput rather than a cumulative total.
    """
    return float(np.sum(_as_rates(rates_bps))) / _n_steps(rates_bps)


def average_data_rate(rates_bps: np.ndarray) -> float:
    """Mean data rate per user, in bits per second."""
    return float(np.mean(_as_rates(rates_bps)))


def jains_fairness(rates_bps: np.ndarray) -> float:
    """Jain's fairness index of the rate allocation.

    ``J = (sum x_i)^2 / (n * sum x_i^2)``, which lies in ``[1/n, 1]``: it is
    1.0 when every user gets exactly the same rate and approaches ``1/n``
    when a single user takes everything.  An all-zero allocation is defined
    as perfectly fair (1.0), since nobody is disadvantaged relative to
    anyone else.

    Note that for an ``(T, M)`` input every (step, user) pair is treated as
    one share, so the index also penalises rates that swing over time.
    """
    rates = _as_rates(rates_bps)
    numerator = float(np.sum(rates)) ** 2
    denominator = rates.size * float(np.sum(rates**2))
    if denominator == 0.0:
        return 1.0
    return float(numerator / denominator)


def coverage_ratio(rates_bps: np.ndarray, min_rate_bps: float) -> float:
    """Fraction of users meeting the minimum rate requirement.

    Parameters
    ----------
    rates_bps : array_like
        Per-user rates, shape ``(M,)`` or ``(T, M)``.
    min_rate_bps : float
        Quality-of-service threshold in bits per second.
    """
    rates = _as_rates(rates_bps)
    return float(np.mean(rates >= min_rate_bps))


def energy_efficiency(rates_bps: np.ndarray, energy_j: float | np.ndarray) -> float:
    """Delivered bits per joule of energy consumed.

    Parameters
    ----------
    rates_bps : array_like
        Per-user rates, shape ``(M,)`` or ``(T, M)``.
    energy_j : float or array_like
        Energy spent over the same window, in joules.  Arrays are summed.

    Returns
    -------
    float
        Per-step throughput divided by mean per-step energy, i.e. delivered
        bits per joule assuming one-second steps.  Because both terms are
        averaged over the same ``T`` steps this equals total bits over total
        energy.  Returns 0.0 when no energy was drawn, which avoids a division
        by zero in degenerate episodes.
    """
    energy = float(np.sum(np.asarray(energy_j, dtype=float)))
    if energy <= 0.0:
        return 0.0
    energy_per_step = energy / _n_steps(rates_bps)
    return total_throughput(rates_bps) / energy_per_step


def spectral_efficiency(rates_bps: np.ndarray, bandwidth_hz: np.ndarray) -> float:
    """Aggregate spectral efficiency in bits/s/Hz.

    Measures how much rate the network extracts per hertz it occupies, which
    exposes resource waste that raw throughput hides.
    """
    bandwidth = float(np.sum(np.asarray(bandwidth_hz, dtype=float)))
    if bandwidth <= 0.0:
        return 0.0
    # Both sums run over the same steps, so the ratio needs no per-step scaling.
    return float(np.sum(_as_rates(rates_bps))) / bandwidth


def resource_utilization(
    bandwidth_hz: np.ndarray,
    rates_bps: np.ndarray,
    min_rate_bps: float,
    available_per_step_hz: float,
) -> float:
    """Fraction of the available spectrum that delivers usable service.

    Bandwidth counts towards utilisation only if the user receiving it
    actually clears the QoS threshold.  That makes the metric penalise two
    distinct failures at once:

    * **idle spectrum** - a UAV whose cluster is empty hands out nothing, so
      its whole band goes unused;
    * **wasted spectrum** - bandwidth given to a user who still ends up below
      ``min_rate_bps`` bought nothing and does not count.

    Parameters
    ----------
    bandwidth_hz : array_like
        Allocated bandwidth per user, shape ``(M,)`` or ``(T, M)``.
    rates_bps : array_like
        Achieved rates, same shape as ``bandwidth_hz``.
    min_rate_bps : float
        QoS threshold.
    available_per_step_hz : float
        Total spectrum the network could hand out in *one* step.  Under the
        full-reuse model of this project that is ``n_uav * bandwidth_hz``,
        since every UAV independently reuses the whole band.

    Returns
    -------
    float
        Useful allocated bandwidth divided by the total available over the
        same window, in ``[0, 1]``.
    """
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    rates = np.asarray(rates_bps, dtype=float)
    if bandwidth.shape != rates.shape:
        raise ValueError(
            f"bandwidth_hz shape {bandwidth.shape} does not match "
            f"rates_bps shape {rates.shape}"
        )
    if available_per_step_hz <= 0.0:
        return 0.0

    # One row per step for a (T, M) episode; a bare (M,) snapshot is one step.
    n_steps = bandwidth.shape[0] if bandwidth.ndim == 2 else 1
    useful = float(np.sum(bandwidth[rates >= min_rate_bps]))
    return useful / (available_per_step_hz * n_steps)


def summarize(
    rates_bps: np.ndarray,
    min_rate_bps: float,
    energy_j: float | np.ndarray | None = None,
    bandwidth_hz: np.ndarray | None = None,
    available_per_step_hz: float | None = None,
) -> dict[str, float]:
    """Compute every metric at once for one episode.

    Parameters
    ----------
    rates_bps : array_like
        Per-user rates, shape ``(M,)`` or ``(T, M)``.
    min_rate_bps : float
        Coverage threshold in bits per second.
    energy_j : float or array_like, optional
        Energy consumed; required for the energy-efficiency entry.
    bandwidth_hz : array_like, optional
        Allocated bandwidth; required for the spectral-efficiency entry.
    available_per_step_hz : float, optional
        Total spectrum available per step; required, together with
        ``bandwidth_hz``, for the resource-utilisation entry.

    Returns
    -------
    dict
        Metric name to value.  Rates are reported in Mbps for readability,
        energy efficiency in kilobits per joule.
    """
    rates = _as_rates(rates_bps)
    summary = {
        "total_throughput_mbps": total_throughput(rates_bps) / 1e6,
        "avg_rate_mbps": average_data_rate(rates) / 1e6,
        "jains_fairness": jains_fairness(rates),
        "coverage": coverage_ratio(rates, min_rate_bps),
    }
    if energy_j is not None:
        summary["energy_efficiency_kbit_per_j"] = energy_efficiency(rates_bps, energy_j) / 1e3
    if bandwidth_hz is not None:
        summary["spectral_efficiency_bps_hz"] = spectral_efficiency(rates, bandwidth_hz)
        if available_per_step_hz is not None:
            summary["resource_utilization"] = resource_utilization(
                bandwidth_hz, rates_bps, min_rate_bps, available_per_step_hz
            )
    return summary
