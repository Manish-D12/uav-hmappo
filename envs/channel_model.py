"""Air-to-ground (A2G) channel model for UAV communication.

This module implements the widely used probabilistic LoS channel model of
Al-Hourani et al. for low-altitude aerial platforms.  The chain of
computations is:

    geometry -> elevation angle -> LoS probability
             -> path loss (LoS / NLoS, probabilistically averaged)
             -> received power -> SINR -> Shannon capacity

Every function is a pure NumPy function: no state is kept, inputs may be
scalars or arrays of any broadcastable shape, and outputs follow NumPy's
broadcasting rules.  This makes it easy to evaluate all UAV-user pairs at
once, e.g. by passing UAV positions of shape ``(N, 1, 3)`` against user
positions of shape ``(1, M, 2)`` to obtain an ``(N, M)`` result.

Units convention
----------------
Distances are metres, frequencies hertz, bandwidths hertz, powers dBm
unless a name says ``_w`` (watts), and losses/gains dB.  Rates are bits per
second.

References
----------
A. Al-Hourani, S. Kandeepan and S. Lardner, "Optimal LAP Altitude for
Maximum Coverage," IEEE Wireless Communications Letters, 2014.
"""

from __future__ import annotations

import numpy as np

# Speed of light in vacuum [m/s].
SPEED_OF_LIGHT = 299_792_458.0

# Thermal noise power spectral density at room temperature [dBm/Hz].
NOISE_PSD_DBM_PER_HZ = -174.0

# Environment-dependent parameters from Al-Hourani et al.
#   a, b          : sigmoid shape parameters of the LoS probability
#   eta_los       : excess path loss for a LoS link [dB]
#   eta_nlos      : excess path loss for an NLoS link [dB]
ENVIRONMENTS: dict[str, dict[str, float]] = {
    "suburban": {"a": 4.88, "b": 0.43, "eta_los": 0.1, "eta_nlos": 21.0},
    "urban": {"a": 9.61, "b": 0.16, "eta_los": 1.0, "eta_nlos": 20.0},
    "dense_urban": {"a": 12.08, "b": 0.11, "eta_los": 1.6, "eta_nlos": 23.0},
    "high_rise": {"a": 27.23, "b": 0.08, "eta_los": 2.3, "eta_nlos": 34.0},
}


# --------------------------------------------------------------------------
# Unit helpers
# --------------------------------------------------------------------------
def dbm_to_watt(power_dbm: np.ndarray | float) -> np.ndarray:
    """Convert a power in dBm to watts."""
    return 10.0 ** ((np.asarray(power_dbm, dtype=float) - 30.0) / 10.0)


def watt_to_dbm(power_w: np.ndarray | float) -> np.ndarray:
    """Convert a power in watts to dBm."""
    return 10.0 * np.log10(np.asarray(power_w, dtype=float)) + 30.0


def db_to_linear(value_db: np.ndarray | float) -> np.ndarray:
    """Convert a dB ratio to its linear equivalent."""
    return 10.0 ** (np.asarray(value_db, dtype=float) / 10.0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def horizontal_distance(uav_xy: np.ndarray, user_xy: np.ndarray) -> np.ndarray:
    """Ground-projected distance between UAVs and users.

    Parameters
    ----------
    uav_xy : array_like, shape (..., 2)
        Horizontal UAV coordinates ``(x, y)`` in metres.
    user_xy : array_like, shape (..., 2)
        Horizontal user coordinates ``(x, y)`` in metres.

    Returns
    -------
    ndarray
        Euclidean distance in the ground plane, broadcast over the leading
        dimensions of the two inputs.
    """
    delta = np.asarray(uav_xy, dtype=float) - np.asarray(user_xy, dtype=float)
    return np.linalg.norm(delta, axis=-1)


def link_distance(
    uav_xy: np.ndarray, altitude: np.ndarray | float, user_xy: np.ndarray
) -> np.ndarray:
    """Straight-line (3-D) distance of the air-to-ground link in metres."""
    r = horizontal_distance(uav_xy, user_xy)
    return np.hypot(r, np.asarray(altitude, dtype=float))


def elevation_angle_deg(
    uav_xy: np.ndarray, altitude: np.ndarray | float, user_xy: np.ndarray
) -> np.ndarray:
    """Elevation angle of the UAV as seen from the user, in degrees.

    A user directly below the UAV sees 90 degrees; a user infinitely far
    away sees 0 degrees.  A small epsilon guards the degenerate case where
    the user is exactly under the UAV.
    """
    r = horizontal_distance(uav_xy, user_xy)
    h = np.asarray(altitude, dtype=float)
    return np.degrees(np.arctan2(h, np.maximum(r, 1e-9)))


# --------------------------------------------------------------------------
# LoS probability and path loss
# --------------------------------------------------------------------------
def los_probability(
    theta_deg: np.ndarray | float, a: float, b: float
) -> np.ndarray:
    """Probability that an A2G link is line-of-sight.

    Uses the sigmoid model ``P_LoS = 1 / (1 + a * exp(-b * (theta - a)))``,
    which rises towards 1 as the elevation angle increases (the UAV clears
    surrounding buildings).

    Parameters
    ----------
    theta_deg : array_like
        Elevation angle in degrees.
    a, b : float
        Environment-dependent shape parameters, see :data:`ENVIRONMENTS`.
    """
    theta = np.asarray(theta_deg, dtype=float)
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def free_space_path_loss_db(
    distance_m: np.ndarray | float, carrier_hz: float
) -> np.ndarray:
    """Free-space path loss in dB for a given distance and carrier frequency."""
    d = np.maximum(np.asarray(distance_m, dtype=float), 1.0)  # avoid log(0)
    return 20.0 * np.log10(4.0 * np.pi * d * carrier_hz / SPEED_OF_LIGHT)


def path_loss_db(
    uav_xy: np.ndarray,
    altitude: np.ndarray | float,
    user_xy: np.ndarray,
    carrier_hz: float,
    environment: str = "urban",
) -> np.ndarray:
    """Expected A2G path loss in dB, averaged over LoS and NLoS conditions.

    The model computes free-space loss and adds an excess loss that depends
    on whether the link is blocked.  Since blockage is random, the two cases
    are averaged with the LoS probability as weight:

        PL = P_LoS * (FSPL + eta_LoS) + (1 - P_LoS) * (FSPL + eta_NLoS)

    Parameters
    ----------
    uav_xy : array_like, shape (..., 2)
        Horizontal UAV positions in metres.
    altitude : array_like or float
        UAV altitude in metres.
    user_xy : array_like, shape (..., 2)
        Horizontal user positions in metres.
    carrier_hz : float
        Carrier frequency in hertz.
    environment : str
        Key into :data:`ENVIRONMENTS`.

    Returns
    -------
    ndarray
        Average path loss in dB, broadcast over the input shapes.
    """
    if environment not in ENVIRONMENTS:
        raise KeyError(
            f"unknown environment {environment!r}; "
            f"choose from {sorted(ENVIRONMENTS)}"
        )
    params = ENVIRONMENTS[environment]

    d = link_distance(uav_xy, altitude, user_xy)
    theta = elevation_angle_deg(uav_xy, altitude, user_xy)
    p_los = los_probability(theta, params["a"], params["b"])

    fspl = free_space_path_loss_db(d, carrier_hz)
    return fspl + p_los * params["eta_los"] + (1.0 - p_los) * params["eta_nlos"]


# --------------------------------------------------------------------------
# Link budget
# --------------------------------------------------------------------------
def noise_power_dbm(bandwidth_hz: np.ndarray | float, noise_figure_db: float = 7.0) -> np.ndarray:
    """Thermal noise power in dBm over a given bandwidth.

    ``N = -174 dBm/Hz + 10 log10(B) + NF``.  Bandwidth is floored at 1 Hz so
    that a UAV allocating zero bandwidth does not produce ``-inf``.
    """
    b = np.maximum(np.asarray(bandwidth_hz, dtype=float), 1.0)
    return NOISE_PSD_DBM_PER_HZ + 10.0 * np.log10(b) + noise_figure_db


def received_power_dbm(
    tx_power_dbm: np.ndarray | float,
    path_loss: np.ndarray | float,
    antenna_gain_db: float = 0.0,
) -> np.ndarray:
    """Received signal power in dBm: ``P_rx = P_tx + G - PL``."""
    return (
        np.asarray(tx_power_dbm, dtype=float)
        + antenna_gain_db
        - np.asarray(path_loss, dtype=float)
    )


def sinr_linear(
    rx_power_dbm: np.ndarray,
    interference_dbm: np.ndarray | float,
    noise_dbm: np.ndarray | float,
) -> np.ndarray:
    """Linear SINR from dBm-valued signal, interference and noise powers.

    All three arguments are converted to watts before the ratio is formed,
    because powers only add linearly.
    """
    signal_w = dbm_to_watt(rx_power_dbm)
    interference_w = dbm_to_watt(interference_dbm)
    noise_w = dbm_to_watt(noise_dbm)
    return signal_w / (interference_w + noise_w)


def shannon_capacity_bps(
    bandwidth_hz: np.ndarray | float, sinr: np.ndarray | float
) -> np.ndarray:
    """Shannon capacity ``C = B * log2(1 + SINR)`` in bits per second."""
    b = np.asarray(bandwidth_hz, dtype=float)
    s = np.maximum(np.asarray(sinr, dtype=float), 0.0)
    return b * np.log2(1.0 + s)


# --------------------------------------------------------------------------
# Convenience: full downlink evaluation for all UAV-user pairs
# --------------------------------------------------------------------------
def downlink_rates(
    uav_xy: np.ndarray,
    altitudes: np.ndarray,
    user_xy: np.ndarray,
    assignment: np.ndarray,
    bandwidth_hz: np.ndarray,
    tx_power_dbm: np.ndarray,
    carrier_hz: float,
    environment: str = "urban",
    noise_figure_db: float = 7.0,
    antenna_gain_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate per-user data rates for a full UAV deployment.

    Each user is served by exactly one UAV (given by ``assignment``).  UAVs
    are assumed to reuse the same spectrum, so every *other* UAV contributes
    co-channel interference at the user's location.

    Parameters
    ----------
    uav_xy : ndarray, shape (N, 2)
        Horizontal UAV positions in metres.
    altitudes : ndarray, shape (N,)
        UAV altitudes in metres.
    user_xy : ndarray, shape (M, 2)
        User positions in metres.
    assignment : ndarray of int, shape (M,)
        ``assignment[j]`` is the index of the UAV serving user ``j``.
    bandwidth_hz : ndarray, shape (M,)
        Bandwidth allocated to each user in hertz.
    tx_power_dbm : ndarray, shape (N, M) or (M,)
        Transmit power used for each user's stream, in dBm.
    carrier_hz : float
        Carrier frequency in hertz.
    environment : str
        Key into :data:`ENVIRONMENTS`.
    noise_figure_db : float
        Receiver noise figure in dB.
    antenna_gain_db : float
        Combined transmit/receive antenna gain in dB.

    Returns
    -------
    rates : ndarray, shape (M,)
        Achievable rate per user in bits per second.
    sinr : ndarray, shape (M,)
        Linear SINR per user.
    """
    uav_xy = np.atleast_2d(np.asarray(uav_xy, dtype=float))
    user_xy = np.atleast_2d(np.asarray(user_xy, dtype=float))
    altitudes = np.asarray(altitudes, dtype=float).reshape(-1)
    assignment = np.asarray(assignment, dtype=int).reshape(-1)
    bandwidth_hz = np.asarray(bandwidth_hz, dtype=float).reshape(-1)

    n_uav, n_user = uav_xy.shape[0], user_xy.shape[0]

    # (N, M) path loss for every UAV-user pair.
    pl = path_loss_db(
        uav_xy[:, None, :], altitudes[:, None], user_xy[None, :, :],
        carrier_hz, environment,
    )

    tx = np.broadcast_to(np.asarray(tx_power_dbm, dtype=float), (n_uav, n_user))
    rx_all_w = dbm_to_watt(received_power_dbm(tx, pl, antenna_gain_db))  # (N, M)

    # Serving link: pick the row of the assigned UAV for each user.
    users = np.arange(n_user)
    serving_w = rx_all_w[assignment, users]

    # Interference: total received power minus the serving contribution.
    interference_w = rx_all_w.sum(axis=0) - serving_w

    noise_w = dbm_to_watt(noise_power_dbm(bandwidth_hz, noise_figure_db))
    sinr = serving_w / (interference_w + noise_w)
    rates = shannon_capacity_bps(bandwidth_hz, sinr)
    return rates, sinr
