"""Plotting: matplotlib for 2D, plotly (optional) for interactive 3D."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .constants import seconds_per_day


def _require_time(sim_data: pd.DataFrame) -> None:
    if "time" not in sim_data.columns:
        raise ValueError("`sim_data` must have a `time` column. "
                         "Did you pass the output of simulate_system()?")


def snapshot_at(sim_data: pd.DataFrame, time: float | None = None) -> pd.DataFrame:
    """Rows of the recorded time step closest to ``time`` (default: the last)."""
    _require_time(sim_data)
    available = np.sort(sim_data["time"].unique())
    target = available[-1] if time is None else available[np.argmin(np.abs(available - time))]
    return sim_data.loc[sim_data["time"] == target]


def downsample_frames(sim_data: pd.DataFrame, fps: float, duration: float) -> pd.DataFrame:
    """Thin a simulation to about ``fps * duration`` evenly spaced time steps."""
    _require_time(sim_data)
    available = np.sort(sim_data["time"].unique())
    n_frames = max(2, round(fps * duration))
    if len(available) <= n_frames:
        return sim_data
    keep = available[np.round(np.linspace(0, len(available) - 1, n_frames)).astype(int)]
    return sim_data.loc[sim_data["time"].isin(keep)]


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    return go


def _use_3d(sim_data: pd.DataFrame, three_d: bool | None, what: str):
    """The plotly module if this plot should be 3D, else None."""
    is_3d = bool((sim_data["z"] != 0).any()) if three_d is None else three_d
    if not is_3d:
        return None
    go = _plotly()
    if go is None:
        warnings.warn(f"3D motion detected, but 'plotly' is not installed. Falling back to a 2D {what}. "
                      "Install it with: pip install plotly", stacklevel=3)
    return go


def _new_axes(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(7, 7))
    return ax


def _finish_axes(ax, title: str):
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(title="Body", loc="best", frameon=False)
    return ax


def _scene_layout(title: str) -> dict:
    axis = dict(showgrid=True)
    return dict(
        title=title,
        scene=dict(xaxis=dict(title="X (m)", **axis), yaxis=dict(title="Y (m)", **axis),
                   zaxis=dict(title="Z (m)", **axis),
                   aspectmode="data"),  # keeps circular orbits looking circular
        plot_bgcolor="white", paper_bgcolor="white",
    )


def _trail_traces(go, sim_data: pd.DataFrame) -> list:
    return [go.Scatter3d(x=g["x"], y=g["y"], z=g["z"], mode="lines", name=str(body),
                         line=dict(width=3), opacity=0.35, hoverinfo="skip", showlegend=False)
            for body, g in sim_data.groupby("id", sort=False)]


# ---- Trajectories ---------------------------------------------------------

def plot_orbits(sim_data: pd.DataFrame, three_d: bool | None = None, ax=None):
    """Plot the full trajectory of every body.

    Draws a 2D matplotlib plot (returning its ``Axes``) unless there is motion
    in ``z``, in which case it returns an interactive plotly figure. Force one
    or the other with ``three_d``.
    """
    if _use_3d(sim_data, three_d, "plot") is not None:
        return plot_orbits_3d(sim_data)

    ax = _new_axes(ax)
    for body, g in sim_data.groupby("id", sort=False):
        ax.plot(g["x"], g["y"], linewidth=1.5, label=str(body))
    return _finish_axes(ax, "2D Orbital Trajectories")


def plot_orbits_3d(sim_data: pd.DataFrame):
    """Interactive 3D trajectories (requires plotly)."""
    go = _plotly()
    if go is None:
        raise ImportError("The 'plotly' package is required for 3D plotting. Install it with: pip install plotly")

    fig = go.Figure()
    for body, g in sim_data.groupby("id", sort=False):
        days = (g["time"] / seconds_per_day).round(1)
        fig.add_trace(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"], mode="lines", name=str(body), line=dict(width=4),
            hoverinfo="text", text=[f"Body: {body}<br>Time: {d} days" for d in days]))
    fig.update_layout(**_scene_layout("3D Orbital Trajectories"))
    return fig


# ---- Snapshots ------------------------------------------------------------

def plot_system(sim_data: pd.DataFrame, time: float | None = None, trails: bool = False,
                three_d: bool | None = None, ax=None):
    """Plot where every body is at one time (default: the last time step).

    ``time`` is in seconds; the closest recorded step is used. ``trails=True``
    draws the full trajectories faintly behind the points.
    """
    if _use_3d(sim_data, three_d, "plot") is not None:
        return plot_system_3d(sim_data, time=time, trails=trails)

    snap = snapshot_at(sim_data, time)
    ax = _new_axes(ax)
    for body, g in sim_data.groupby("id", sort=False):
        line = ax.plot(g["x"], g["y"], linewidth=1, alpha=0.35 if trails else 0)[0]
        now = snap.loc[snap["id"] == body]
        ax.plot(now["x"], now["y"], "o", markersize=9, color=line.get_color(), label=str(body))
    return _finish_axes(ax, f"System Snapshot (t = {snap['time'].iloc[0]:.3g} s)")


def plot_system_3d(sim_data: pd.DataFrame, time: float | None = None, trails: bool = False):
    """Interactive 3D snapshot (requires plotly)."""
    go = _plotly()
    if go is None:
        raise ImportError("The 'plotly' package is required for 3D plotting. Install it with: pip install plotly")

    snap = snapshot_at(sim_data, time)
    fig = go.Figure(_trail_traces(go, sim_data) if trails else [])
    for row in snap.itertuples(index=False):
        fig.add_trace(go.Scatter3d(
            x=[row.x], y=[row.y], z=[row.z], mode="markers", name=str(row.id), marker=dict(size=6),
            hoverinfo="text",
            text=(f"Body: {row.id}<br>Time: {row.time / seconds_per_day:.2f} days"
                  f"<br>x: {row.x:.4g}<br>y: {row.y:.4g}<br>z: {row.z:.4g}")))
    fig.update_layout(**_scene_layout(f"System Snapshot (t = {snap['time'].iloc[0]:.3g} s)"))
    return fig


# ---- Animation ------------------------------------------------------------

def animate_system(sim_data: pd.DataFrame, fps: float = 20, duration: float = 10,
                   trails: bool = False, three_d: bool | None = None, path=None):
    """Animate the system.

    Returns a matplotlib ``FuncAnimation`` in 2D (shown inline in Jupyter with
    ``HTML(anim.to_jshtml())``), or an interactive plotly figure if there is
    motion in ``z``. ``fps`` and ``duration`` (seconds of playback) set how
    many frames are drawn. Pass ``path="orbit.gif"`` to save a 2D animation
    (requires pillow).
    """
    if _use_3d(sim_data, three_d, "animation") is not None:
        return animate_system_3d(sim_data, fps=fps, duration=duration, trails=trails)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = downsample_frames(sim_data, fps=fps, duration=duration)
    times = np.sort(frames["time"].unique())
    bodies = list(dict.fromkeys(frames["id"]))
    by_body = {b: frames.loc[frames["id"] == b].sort_values("time") for b in bodies}
    wake = max(1, round(0.1 * len(times)))

    fig, ax = plt.subplots(figsize=(7, 7))
    pad = 0.05
    for lo_hi, setter in ((sim_data["x"], ax.set_xlim), (sim_data["y"], ax.set_ylim)):
        lo, hi = lo_hi.min(), lo_hi.max()
        span = (hi - lo) or 1.0
        setter(lo - pad * span, hi + pad * span)
    points, tails = {}, {}
    for b in bodies:
        tails[b] = ax.plot([], [], linewidth=1.5, alpha=0.35)[0]
        points[b] = ax.plot([], [], "o", markersize=9, color=tails[b].get_color(), label=str(b))[0]
    _finish_axes(ax, "")

    def draw(k):
        for b in bodies:
            g = by_body[b]
            points[b].set_data(g["x"].iloc[k:k + 1], g["y"].iloc[k:k + 1])
            if trails:
                start = max(0, k - wake)
                tails[b].set_data(g["x"].iloc[start:k + 1], g["y"].iloc[start:k + 1])
        ax.set_title(f"Time: {times[k] / seconds_per_day:.1f} days")
        return [*points.values(), *tails.values()]

    anim = FuncAnimation(fig, draw, frames=len(times), interval=1000 / fps, blit=False)
    if path is not None:
        anim.save(path, writer="pillow", fps=fps)
    return anim


def animate_system_3d(sim_data: pd.DataFrame, fps: float = 20, duration: float = 10, trails: bool = False):
    """Interactive 3D animation with a play button and time slider (requires plotly)."""
    go = _plotly()
    if go is None:
        raise ImportError("The 'plotly' package is required for 3D animation. Install it with: pip install plotly")

    frames = downsample_frames(sim_data, fps=fps, duration=duration)
    times = np.sort(frames["time"].unique())
    bodies = list(dict.fromkeys(frames["id"]))
    trail = _trail_traces(go, sim_data) if trails else []
    marker_idx = list(range(len(trail), len(trail) + len(bodies)))

    def markers(t):
        snap = frames.loc[frames["time"] == t].set_index("id")
        return [go.Scatter3d(
            x=[snap.at[b, "x"]], y=[snap.at[b, "y"]], z=[snap.at[b, "z"]], mode="markers",
            name=str(b), marker=dict(size=6), hoverinfo="text",
            text=f"Body: {b}<br>Time: {t / seconds_per_day:.2f} days") for b in bodies]

    fig = go.Figure(data=trail + markers(times[0]))
    fig.frames = [go.Frame(data=markers(t), traces=marker_idx, name=f"{t:g}") for t in times]

    step = dict(frame=dict(duration=1000 / fps, redraw=True), transition=dict(duration=0))
    lo, hi = sim_data[["x", "y", "z"]].min(), sim_data[["x", "y", "z"]].max()
    layout = _scene_layout("System Animation")
    for axis in "xyz":
        layout["scene"][f"{axis}axis"]["range"] = [lo[axis], hi[axis]]
    fig.update_layout(
        **layout,
        updatemenus=[dict(type="buttons", showactive=False, buttons=[
            dict(label="Play", method="animate", args=[None, dict(fromcurrent=True, **step)]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", **step)]),
        ])],
        sliders=[dict(currentvalue=dict(prefix="Time: ", suffix=" days"), steps=[
            dict(label=f"{t / seconds_per_day:.1f}", method="animate",
                 args=[[f"{t:g}"], dict(mode="immediate", **step)]) for t in times])],
    )
    return fig
