"""Interactive Streamlit demo of the trained hierarchical MAPPO agent.

Replays one scenario from the fixed evaluation benchmark with the trained
hierarchical agent (``results/model_best.pt.high`` / ``.low``), animating the
UAVs step by step next to live coverage / throughput / fairness readouts, and
shows the final three-way comparison from ``results/comparison.csv`` together
with the training convergence curves.

Launch from the project root::

    streamlit run scripts/demo.py
"""

from __future__ import annotations

import os
import sys
import time

import altair as alt
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Allow `streamlit run scripts/demo.py` to find the project packages.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.uav_env import UAVEnv, UAVEnvConfig  # noqa: E402
from utils import metrics, scenarios as scen  # noqa: E402
from utils.config import build_agent, build_dataclass, load_config  # noqa: E402
from utils.plotting import METHOD_COLORS, METRIC_LABELS  # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
RESULTS = os.path.join(ROOT, "results")
CHECKPOINT = os.path.join(RESULTS, "model_best.pt")  # .high / .low appended by load()
SCENARIO_PATH = os.path.join(RESULTS, "eval_scenarios.npz")
COMPARISON_PATH = os.path.join(RESULTS, "comparison.csv")

# One colour per UAV, in fixed categorical order (blue, orange, aqua, ...).
UAV_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
UNCOVERED = "#9a9a94"
METHOD_ORDER = ["K-means", "Flat MAPPO", "Hierarchical MAPPO"]


# --------------------------------------------------------------------------
# Loading (cached across reruns)
# --------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Config, environment, benchmark and trained agent, built once per session."""
    config = load_config(CONFIG_PATH)
    env_cfg = build_dataclass(UAVEnvConfig, config.get("env"))
    env = UAVEnv(env_cfg)
    scenarios = scen.load_scenarios(SCENARIO_PATH, env_cfg)
    agent = build_agent(env, config, hierarchical=True)
    agent.load(CHECKPOINT)
    return config, env, scenarios, agent


@st.cache_data(show_spinner="Running the trained agent...")
def rollout(scenario: int) -> dict[str, np.ndarray]:
    """Run the hierarchical agent greedily over one benchmark scenario.

    Frame 0 is the initial layout; frame ``k >= 1`` is the state after
    environment step ``k``, with that step's clustering and rates.
    """
    _, env, scenarios, agent = load_assets()
    obs, info = env.reset(options=scen.scenario_options(scenarios, scenario))
    state = info["state"]
    agent.begin_episode()

    uav_xy = [env.uav_xy.copy()]
    assignment, rates, bandwidth, energy, rewards = [], [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs, state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        state = info["state"]
        uav_xy.append(info["uav_xy"])
        assignment.append(info["assignment"])
        rates.append(info["rates_bps"])
        bandwidth.append(info["bandwidth_hz"])
        energy.append(info["energy_used_j"])
        rewards.append(reward)

    return {
        "user_xy": env.user_xy.copy(),
        "uav_xy": np.asarray(uav_xy),
        "assignment": np.asarray(assignment),
        "rates": np.asarray(rates),
        "bandwidth": np.asarray(bandwidth),
        "energy": np.asarray(energy),
        "rewards": np.asarray(rewards),
    }


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH, index_col=0)


# --------------------------------------------------------------------------
# Per-step metrics
# --------------------------------------------------------------------------
def step_metrics(rates: np.ndarray, min_rate: float) -> pd.DataFrame:
    """Coverage, network throughput and Jain's fairness for every step."""
    rows = [
        {
            "step": t + 1,
            "coverage": metrics.coverage_ratio(r, min_rate),
            "throughput_mbps": metrics.total_throughput(r) / 1e6,
            "fairness": metrics.jains_fairness(r),
        }
        for t, r in enumerate(rates)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_map(
    env: UAVEnv,
    user_xy: np.ndarray,
    uav_hist: np.ndarray,
    frame: int,
    assignment: np.ndarray | None,
    rates: np.ndarray | None,
) -> plt.Figure:
    """Service area with users, UAVs, their trails and coverage footprints.

    A user is drawn in its serving UAV's colour only if its rate clears the
    QoS threshold; uncovered users are hollow grey.  Each UAV's circle spans
    out to the farthest user it is actually covering.
    """
    size = env.cfg.area_size_m
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="#555")
    ax.set_ylabel("y (m)", color="#555")
    ax.tick_params(colors="#777", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#ccc")
    ax.grid(color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)

    uav_now = uav_hist[frame]
    covered = np.zeros(len(user_xy), dtype=bool)
    if frame > 0 and rates is not None:
        covered = rates[frame - 1] >= env.cfg.min_rate_bps
        assign = assignment[frame - 1]

    for i in range(env.n_uav):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        # Trail up to the current frame.
        trail = uav_hist[: frame + 1, i]
        ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.5, alpha=0.45)
        ax.plot(*uav_hist[0, i], marker="x", color=color, markersize=7, alpha=0.6)

        if frame > 0:
            served = np.flatnonzero((assign == i) & covered)
            if served.size:
                d = np.linalg.norm(user_xy[served] - uav_now[i], axis=1)
                ax.add_patch(
                    plt.Circle(uav_now[i], d.max() + 15, color=color, alpha=0.10, linewidth=0)
                )
                ax.add_patch(
                    plt.Circle(uav_now[i], d.max() + 15, fill=False, edgecolor=color,
                               linewidth=1.2, linestyle="--")
                )
                for j in served:
                    ax.plot([uav_now[i, 0], user_xy[j, 0]], [uav_now[i, 1], user_xy[j, 1]],
                            color=color, linewidth=0.6, alpha=0.5)
            user_colors = [color] * served.size
            ax.scatter(user_xy[served, 0], user_xy[served, 1], s=55, c=user_colors,
                       edgecolors="white", linewidths=1.5, zorder=3)

    uncovered = ~covered
    ax.scatter(user_xy[uncovered, 0], user_xy[uncovered, 1], s=50, facecolors="white",
               edgecolors=UNCOVERED, linewidths=1.5, zorder=3)

    for i in range(env.n_uav):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.scatter(*uav_now[i], marker="^", s=260, c=color, edgecolors="white",
                   linewidths=2, zorder=4)
        ax.annotate(f"UAV {i + 1}", uav_now[i], xytext=(9, 9), textcoords="offset points",
                    fontsize=9, color="#333", zorder=5)

    legend = [
        Line2D([], [], marker="^", linestyle="", color="#555", markersize=10, label="UAV"),
        Line2D([], [], marker="o", linestyle="", color="#555", markeredgecolor="white",
               markersize=8, label="user, covered (UAV colour)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=UNCOVERED, markersize=8, label="user, not covered"),
        Line2D([], [], marker="x", linestyle="", color="#555", markersize=7, label="start"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
              fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def trend_chart(df: pd.DataFrame, columns: list[str], title: str, y_domain=None) -> alt.Chart:
    """Small line chart of one or two same-scale per-step metrics."""
    long = df.melt(id_vars="step", value_vars=columns, var_name="metric", value_name="value")
    y = alt.Y("value:Q", title=None,
              scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined)
    return (
        alt.Chart(long, title=title, height=150)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="step", scale=alt.Scale(domain=[0, int(df["step"].max())])),
            y=y,
            color=alt.Color("metric:N", legend=alt.Legend(orient="top", title=None)
                            if len(columns) > 1 else None),
            tooltip=["step", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
    )


def render_metrics(panel, df: pd.DataFrame, frame: int, n_steps: int, interval: int) -> None:
    """Fill the live metrics panel for the given frame."""
    with panel.container():
        if frame == 0:
            st.info("Initial layout. Press **Run** to fly the trained agent.")
            return
        now = df.iloc[frame - 1]
        prev = df.iloc[frame - 2] if frame > 1 else None
        so_far = df.iloc[:frame]

        st.markdown(f"**Step {frame} / {n_steps}**  ·  high-level decision "
                    f"#{(frame - 1) // interval + 1} (re-plans every {interval} steps)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Coverage", f"{now.coverage:.0%}",
                  None if prev is None else f"{now.coverage - prev.coverage:+.0%}")
        c2.metric("Throughput", f"{now.throughput_mbps:.1f} Mbps",
                  None if prev is None else f"{now.throughput_mbps - prev.throughput_mbps:+.1f}")
        c3.metric("Jain's fairness", f"{now.fairness:.2f}",
                  None if prev is None else f"{now.fairness - prev.fairness:+.2f}")
        st.caption(
            f"Episode average so far: coverage {so_far.coverage.mean():.0%} · "
            f"throughput {so_far.throughput_mbps.mean():.1f} Mbps · "
            f"fairness {so_far.fairness.mean():.2f}"
        )
        full = df.copy()
        full.loc[full.step > frame, ["coverage", "fairness", "throughput_mbps"]] = np.nan
        st.altair_chart(trend_chart(full, ["coverage", "fairness"], "Coverage & fairness",
                                    [0, 1]), width="stretch")
        st.altair_chart(trend_chart(full, ["throughput_mbps"], "Network throughput (Mbps)"),
                        width="stretch")


# --------------------------------------------------------------------------
# Results tab
# --------------------------------------------------------------------------
def comparison_chart(summary: pd.DataFrame, metric: str) -> alt.LayerChart:
    """Bars of one metric per method with +/- 1 std whiskers."""
    df = pd.DataFrame({
        "method": summary.index,
        "mean": summary[metric].values,
        "lo": (summary[metric] - summary[f"{metric}_std"]).values,
        "hi": (summary[metric] + summary[f"{metric}_std"]).values,
        "std": summary[f"{metric}_std"].values,
    })
    methods = [m for m in METHOD_ORDER if m in df.method.values]
    color = alt.Color("method:N", scale=alt.Scale(
        domain=methods, range=[METHOD_COLORS.get(m, "#888") for m in methods]), legend=None)
    x = alt.X("method:N", sort=methods, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140))
    base = alt.Chart(df, title=METRIC_LABELS.get(metric, metric), height=260)
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=46).encode(
        x=x, y=alt.Y("mean:Q", title=None), color=color,
        tooltip=["method", alt.Tooltip("mean:Q", format=".3f"),
                 alt.Tooltip("std:Q", format=".3f", title="std")],
    )
    whiskers = base.mark_rule(color="#666", strokeWidth=1.5).encode(x=x, y="lo:Q", y2="hi:Q")
    labels = base.mark_text(dy=-8, fontSize=11, color="#555").encode(
        x=x, y="hi:Q", text=alt.Text("mean:Q", format=".2f"))
    return bars + whiskers + labels


def results_tab() -> None:
    if not os.path.exists(COMPARISON_PATH):
        st.warning(f"{COMPARISON_PATH} not found - run scripts/evaluate.py first.")
        return
    summary = load_comparison()
    n_scen = scen.n_scenarios(load_assets()[2])
    st.subheader("Final comparison on the fixed benchmark")
    st.caption(f"Mean ± 1 std over the {n_scen} evaluation scenarios, all methods on "
               f"identical user layouts (results/comparison.csv).")

    headline = ["coverage", "total_throughput_mbps", "jains_fairness"]
    for col, metric in zip(st.columns(3), headline):
        col.altair_chart(comparison_chart(summary, metric), width="stretch")

    others = [c for c in summary.columns if not c.endswith("_std") and c not in headline]
    with st.expander("Other metrics"):
        cols = st.columns(3)
        for k, metric in enumerate(others):
            cols[k % 3].altair_chart(comparison_chart(summary, metric), width="stretch")
    with st.expander("Table"):
        st.dataframe(summary.style.format("{:.3f}"), width="stretch")

    st.subheader("Training convergence")
    curves = [("Hierarchical MAPPO", "convergence_hier.png"), ("Flat MAPPO", "convergence.png")]
    curves = [(name, os.path.join(RESULTS, f)) for name, f in curves
              if os.path.exists(os.path.join(RESULTS, f))]
    if not curves:
        st.warning("No convergence image found in results/.")
    for tab, (_, path) in zip(st.tabs([name for name, _ in curves]), curves):
        tab.image(path, width="stretch")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="UAV MAPPO Demo", page_icon="🛩️", layout="wide")
    config, env, scenarios, _ = load_assets()
    interval = int((config.get("hierarchical") or {}).get("interval", 5))
    n_steps = env.cfg.max_steps

    with st.sidebar:
        st.header("Scenario")
        scenario = st.selectbox("Evaluation scenario", range(scen.n_scenarios(scenarios)),
                                format_func=lambda i: f"#{i}")
        run = st.button("▶ Run", type="primary", width="stretch")
        speed = st.select_slider("Animation speed", ["slow", "normal", "fast", "fastest"],
                                 value="fast")
        st.divider()
        st.caption(f"{env.n_uav} UAVs · {env.n_users} users · "
                   f"{env.cfg.area_size_m:.0f} m area · {n_steps} steps\n\n"
                   f"Covered = rate ≥ {env.cfg.min_rate_bps / 1e6:.0f} Mbps\n\n"
                   f"Model: results/model_best.pt.high / .low")

    st.title("Hierarchical MAPPO for UAV networks")
    live, results = st.tabs(["Live demo", "Results"])

    with results:
        results_tab()

    with live:
        # Switching scenario drops the old run, so the map shows the new layout.
        if st.session_state.get("scenario") != scenario:
            st.session_state.scenario = scenario
            st.session_state.run = None
            st.session_state.frame = 0
        if run:
            st.session_state.run = rollout(scenario)

        data = st.session_state.run
        user_xy = scenarios["user_xy"][scenario]
        uav_hist = data["uav_xy"] if data else scenarios["uav_xy"][scenario][None]
        df = step_metrics(data["rates"], env.cfg.min_rate_bps) if data else None
        last = len(uav_hist) - 1

        map_col, metric_col = st.columns([1.1, 1])
        with map_col:
            map_ph = st.empty()
        with metric_col:
            metric_ph = st.empty()

        def show(frame: int) -> None:
            fig = draw_map(env, user_xy, uav_hist, frame,
                           data["assignment"] if data else None,
                           data["rates"] if data else None)
            map_ph.pyplot(fig, clear_figure=True)
            plt.close(fig)
            if df is not None:
                render_metrics(metric_ph, df, frame, last, interval)
            else:
                metric_ph.info("Initial layout. Press **Run** to fly the trained agent.")

        if data:
            if run:
                st.session_state.frame = last
            with map_col:
                frame = st.slider("Step (scrub after the run)", 0, last, key="frame")
        else:
            frame = 0

        if run and data:
            stride, delay = {"slow": (1, 0.15), "normal": (1, 0.0), "fast": (2, 0.0),
                             "fastest": (5, 0.0)}[speed]
            frames = list(range(0, last, stride)) + [last]
            for k in frames:
                show(k)
                if delay:
                    time.sleep(delay)
            ep = metrics.summarize(data["rates"], env.cfg.min_rate_bps,
                                   energy_j=data["energy"], bandwidth_hz=data["bandwidth"])
            st.success(
                f"Episode done · return {data['rewards'].sum():.1f} · coverage "
                f"{ep['coverage']:.0%} · throughput {ep['total_throughput_mbps']:.1f} Mbps · "
                f"fairness {ep['jains_fairness']:.2f}"
            )
        else:
            show(frame)


main()
