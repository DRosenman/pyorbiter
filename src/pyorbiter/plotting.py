"""Plotting: matplotlib for 2D, plotly (optional) for interactive 3D."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._frames import accepts_frames
from .constants import seconds_per_day


def _require_time(sim_data: pd.DataFrame) -> None:
    """Raise ValueError unless ``sim_data`` has a ``time`` column."""
    if "time" not in sim_data.columns:
        raise ValueError("`sim_data` must have a `time` column. "
                         "Did you pass the output of simulate_system()?")


@accepts_frames
def snapshot_at(sim_data: pd.DataFrame, time: float | None = None) -> pd.DataFrame:
    """Rows of the recorded time step closest to ``time`` (default: the last)."""
    _require_time(sim_data)
    available = np.sort(sim_data["time"].unique())
    target = available[-1] if time is None else available[np.argmin(np.abs(available - time))]
    return sim_data.loc[sim_data["time"] == target]


@accepts_frames
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
    """The ``plotly.graph_objects`` module, or None if plotly is not installed."""
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
    """``ax`` if given, else the Axes of a new 7 by 7 inch figure."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(7, 7))
    return ax


def _finish_axes(ax, title: str):
    """Apply the shared 2D styling: equal aspect, axis labels, title, grid and legend."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(title="Body", loc="best", frameon=False)
    return ax


def _scene_layout(title: str) -> dict:
    """The plotly layout shared by the 3D figures."""
    axis = dict(showgrid=True)
    return dict(
        title=title,
        scene=dict(xaxis=dict(title="X (m)", **axis), yaxis=dict(title="Y (m)", **axis),
                   zaxis=dict(title="Z (m)", **axis),
                   aspectmode="data"),  # keeps circular orbits looking circular
        plot_bgcolor="white", paper_bgcolor="white",
    )


def _trail_traces(go, sim_data: pd.DataFrame) -> list:
    """One faint ``Scatter3d`` line per body through its whole trajectory."""
    return [go.Scatter3d(x=g["x"], y=g["y"], z=g["z"], mode="lines", name=str(body),
                         line=dict(width=3), opacity=0.35, hoverinfo="skip", showlegend=False)
            for body, g in sim_data.groupby("id", sort=False)]


# ---- Trajectories ---------------------------------------------------------

@accepts_frames
def plot_orbits(sim_data: pd.DataFrame, three_d: bool | None = None, ax=None):
    """Plot the full trajectory of every body.

    Draws one line per body through every recorded position. This is the
    quickest way to look at a simulation.

    The plot is 2D or 3D depending on the data. If every ``z`` value in
    ``sim_data`` is zero, the motion is planar and a matplotlib plot of ``y``
    against ``x`` is drawn. If any ``z`` value is non-zero, the work is handed
    to ``plot_orbits_3d``, which returns an interactive plotly figure. Use
    ``three_d`` to force one or the other.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` (pandas or polars). It must have the
        columns ``id``, ``x``, ``y`` and ``z``, with positions in meters. The
        3D plot also uses ``time``, in seconds.
    three_d : bool, optional
        ``None`` (the default) picks 3D only when there is motion in ``z``.
        ``True`` forces a 3D plotly figure even for planar data. ``False``
        forces a 2D plot, which shows the projection onto the x-y plane.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the 2D plot on. By default a new 7 by 7 inch figure is
        created. Ignored when the plot is 3D.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The matplotlib ``Axes`` for a 2D plot, with an equal aspect ratio, axes
        labelled in meters and a legend of body names. A plotly ``Figure`` for
        a 3D plot.

    See Also
    --------
    plot_orbits_3d : The 3D plot this function switches to.
    plot_system : Positions at a single time instead of whole trajectories.
    animate_system : The same motion played as an animation.
    shift_reference_frame : Re-center the data on one body before plotting.

    Notes
    -----
    plotly is an optional dependency. If a 3D plot is called for and plotly is
    not installed, a warning is issued and the 2D plot is drawn instead.

    The 2D plot returns ordinary matplotlib ``Axes``, so it can be restyled
    with the usual methods (``ax.set_title``, ``ax.set_xlim`` and so on).

    Examples
    --------
    Earth and Mars around the Sun. Mars is placed in the same plane as Earth
    (``i=0``), so the motion is planar and the plot is 2D.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=0)
        .simulate(time_step=orb.seconds_per_day, duration=2 * orb.seconds_per_year)
    )
    orb.plot_orbits(sim)
    ```

    Pass ``ax`` to place the plot in a figure of your own, here next to the
    same data seen from Earth.

    ```{python}
    import matplotlib.pyplot as plt
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=0)
        .simulate(time_step=orb.seconds_per_day, duration=2 * orb.seconds_per_year)
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.5))
    orb.plot_orbits(sim, ax=left).set_title("Sun at rest")
    orb.plot_orbits(orb.shift_reference_frame(sim, "Earth"), ax=right).set_title("Seen from Earth")
    ```
    """
    if _use_3d(sim_data, three_d, "plot") is not None:
        return plot_orbits_3d(sim_data)

    ax = _new_axes(ax)
    for body, g in sim_data.groupby("id", sort=False):
        ax.plot(g["x"], g["y"], linewidth=1.5, label=str(body))
    return _finish_axes(ax, "2D Orbital Trajectories")


@accepts_frames
def plot_orbits_3d(sim_data: pd.DataFrame):
    """Plot the full trajectory of every body in interactive 3D.

    Draws one line per body in a plotly 3D scene. Drag to rotate, scroll to
    zoom, and hover over a line to see the body name and the time in days.
    The three axes share one scale, so a circular orbit looks circular from
    any angle.

    ``plot_orbits`` calls this function for you when the data has motion in
    ``z``. Call it directly when you always want the 3D figure.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` (pandas or polars). It must have the
        columns ``id``, ``x``, ``y``, ``z`` (meters) and ``time`` (seconds).

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with one ``Scatter3d`` line per body and axes labelled in
        meters. It displays itself in a notebook. Use ``fig.show()`` in a
        script or ``fig.write_html(path)`` to save it.

    Raises
    ------
    ImportError
        If plotly is not installed.

    See Also
    --------
    plot_orbits : Picks 2D or 3D from the data.
    plot_system_3d : Positions at a single time in 3D.
    animate_system_3d : The same scene with a play button and time slider.

    Examples
    --------
    Mars on an orbit tilted 25 degrees out of Earth's orbital plane.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=25)
        .simulate(time_step=orb.seconds_per_day, duration=2 * orb.seconds_per_year)
    )
    orb.plot_orbits_3d(sim)
    ```
    """
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

@accepts_frames
def plot_system(sim_data: pd.DataFrame, time: float | None = None, trails: bool = False,
                three_d: bool | None = None, ax=None):
    """Plot where every body is at a single time.

    Draws one marker per body at its position at the chosen time, optionally
    with the full trajectories drawn faintly behind. This is the snapshot
    counterpart to ``plot_orbits``, which draws the whole trajectories.

    If every ``z`` value in ``sim_data`` is zero, a 2D matplotlib plot is
    drawn. If any ``z`` value is non-zero, the work is handed to
    ``plot_system_3d``, which returns an interactive plotly figure. Use
    ``three_d`` to force one or the other.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` (pandas or polars). It must have the
        columns ``id``, ``x``, ``y``, ``z`` (meters) and ``time`` (seconds).
    time : float, optional
        The time to show, in seconds from the start of the simulation. The
        closest recorded time step is used. The default, ``None``, shows the
        last time step.
    trails : bool
        If ``True``, the full trajectory of each body is drawn faintly behind
        its marker. The default, ``False``, shows only the markers.
    three_d : bool, optional
        ``None`` (the default) picks 3D only when there is motion in ``z``.
        ``True`` forces a 3D plotly figure even for planar data. ``False``
        forces a 2D plot, which shows the projection onto the x-y plane.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the 2D plot on. By default a new 7 by 7 inch figure is
        created. Ignored when the plot is 3D.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The matplotlib ``Axes`` for a 2D plot, with an equal aspect ratio, axes
        labelled in meters and a legend of body names. A plotly ``Figure`` for
        a 3D plot. Either way the title gives the time shown, in seconds.

    Raises
    ------
    ValueError
        If ``sim_data`` has no ``time`` column.

    See Also
    --------
    plot_system_3d : The 3D plot this function switches to.
    plot_orbits : Whole trajectories instead of a single time.
    animate_system : A moving snapshot.
    system_from_simulation : Get the bodies at one time as a system, not a plot.

    Notes
    -----
    The 2D axis limits always cover the whole trajectories, whether or not
    ``trails`` is set. Snapshots of the same simulation at different times
    therefore share one frame and can be compared directly.

    plotly is an optional dependency. If a 3D plot is called for and plotly is
    not installed, a warning is issued and the 2D plot is drawn instead.

    Examples
    --------
    The final state of a one-year simulation, with the orbits drawn behind.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=0)
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.plot_system(sim, trails=True)
    ```

    The same system at day 100, markers only.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=0)
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.plot_system(sim, time=100 * orb.seconds_per_day)
    ```
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


@accepts_frames
def plot_system_3d(sim_data: pd.DataFrame, time: float | None = None, trails: bool = False):
    """Plot where every body is at a single time in interactive 3D.

    The 3D counterpart to ``plot_system``. Draws one marker per body at its
    position at the chosen time in a plotly 3D scene, optionally with the full
    trajectories drawn faintly behind. Drag to rotate, scroll to zoom, and
    hover over a marker to see the body name, the time in days and the
    coordinates in meters.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` (pandas or polars). It must have the
        columns ``id``, ``x``, ``y``, ``z`` (meters) and ``time`` (seconds).
    time : float, optional
        The time to show, in seconds from the start of the simulation. The
        closest recorded time step is used. The default, ``None``, shows the
        last time step.
    trails : bool
        If ``True``, the full trajectory of each body is drawn faintly behind
        its marker. The default, ``False``, shows only the markers.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with one ``Scatter3d`` marker per body, axes labelled in
        meters and a title giving the time shown, in seconds. It displays
        itself in a notebook. Use ``fig.show()`` in a script or
        ``fig.write_html(path)`` to save it.

    Raises
    ------
    ImportError
        If plotly is not installed.
    ValueError
        If ``sim_data`` has no ``time`` column.

    See Also
    --------
    plot_system : Picks 2D or 3D from the data.
    plot_orbits_3d : Whole trajectories in 3D.
    animate_system_3d : The same scene with a play button and time slider.

    Notes
    -----
    The three axes share one scale and are fitted to what is drawn. Without
    trails that is only the markers, so the view is tight around the bodies.
    With ``trails=True`` it covers the whole trajectories.

    Examples
    --------
    Mars on an orbit tilted 25 degrees, 200 days into the simulation.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=25)
        .simulate(time_step=orb.seconds_per_day, duration=2 * orb.seconds_per_year)
    )
    orb.plot_system_3d(sim, time=200 * orb.seconds_per_day, trails=True)
    ```
    """
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

@accepts_frames
def animate_system(sim_data: pd.DataFrame, fps: float = 20, duration: float = 10,
                   trails: bool = False, three_d: bool | None = None, path=None):
    """Animate the system over time.

    Plays the simulation forward. Each body is a marker that moves along its
    orbit frame by frame, optionally leaving a short wake behind it. This is
    the animated counterpart to ``plot_system``: a moving snapshot instead of
    a single frozen one.

    If every ``z`` value in ``sim_data`` is zero, a 2D matplotlib animation is
    built. If any ``z`` value is non-zero, the work is handed to
    ``animate_system_3d``, which returns an interactive plotly figure. Use
    ``three_d`` to force one or the other.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` (pandas or polars). It must have the
        columns ``id``, ``x``, ``y``, ``z`` (meters) and ``time`` (seconds).
    fps : float
        Frames per second of playback. Default 20.
    duration : float
        Length of the animation in seconds of playback, not of simulated time.
        Default 10. Together with ``fps`` this sets the number of frames: about
        ``fps * duration`` evenly spaced time steps are drawn, and never fewer
        than 2. If the simulation has no more time steps than that, every time
        step becomes a frame.
    trails : bool
        If ``True``, each body leaves a wake behind it. In 2D the wake is a
        faint line through its most recent positions, about a tenth of the
        frames long. In 3D the full trajectories are drawn faintly instead.
        The default, ``False``, shows only the moving markers.
    three_d : bool, optional
        ``None`` (the default) picks 3D only when there is motion in ``z``.
        ``True`` forces a 3D plotly animation even for planar data. ``False``
        forces a 2D animation of the projection onto the x-y plane.
    path : str or path-like, optional
        Where to save a 2D animation, for example ``"orbit.gif"``. The file is
        written as a GIF at ``fps`` frames per second, which needs the pillow
        package. The default, ``None``, saves nothing. Ignored when the
        animation is 3D.

    Returns
    -------
    matplotlib.animation.FuncAnimation or plotly.graph_objects.Figure
        A matplotlib ``FuncAnimation`` in 2D, on a new 7 by 7 inch figure. Each
        frame is titled with the time in days. A plotly ``Figure`` with Play
        and Pause buttons and a time slider in 3D.

    Raises
    ------
    ValueError
        If ``sim_data`` has no ``time`` column.

    See Also
    --------
    animate_system_3d : The 3D animation this function switches to.
    plot_system : A single frozen snapshot.
    plot_orbits : Whole trajectories in one static plot.

    Notes
    -----
    To show a 2D animation in a Jupyter notebook or a Quarto document, convert
    it with ``IPython.display.HTML(anim.to_jshtml())``. In a script, keep a
    reference to the returned object and call ``matplotlib.pyplot.show()``.

    The 2D view is set once, from the full extent of ``sim_data`` with a
    5 percent margin, and both axes share one scale. It does not rescale as
    the bodies move.

    Saving a GIF draws and encodes every frame, so it takes far longer than a
    static plot. Keep ``fps * duration`` modest. The 3D animation involves no
    encoding and is built quickly.

    plotly is an optional dependency. If a 3D animation is called for and
    plotly is not installed, a warning is issued and the 2D animation is built
    instead.

    Examples
    --------
    A short 2D animation of Earth around the Sun, shown inline: 10 frames,
    from 5 frames per second for 2 seconds. ``three_d=False`` states the
    choice of 2D outright. The figure is closed so that only the animation is
    displayed.

    ```{python}
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    anim = orb.animate_system(sim, fps=5, duration=2, trails=True, three_d=False)
    plt.close()
    HTML(anim.to_jshtml())
    ```

    Save the animation as a GIF. A temporary directory is used here so that
    the example leaves nothing behind.

    ```{python}
    import os
    import tempfile

    import matplotlib.pyplot as plt
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    with tempfile.TemporaryDirectory() as folder:
        orb.animate_system(sim, fps=5, duration=1, path=os.path.join(folder, "earth.gif"))
        plt.close()
        saved = os.listdir(folder)
    saved
    ```
    """
    if _use_3d(sim_data, three_d, "animation") is not None:
        if path is not None:
            warnings.warn("`path` is ignored for a 3D animation, which is an interactive plotly figure "
                          "and is not saved as a GIF. Pass three_d=False to save a 2D GIF.", stacklevel=3)
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
    ax.set_aspect("equal", adjustable="box")   # the view is fixed, so fit the box to it

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


@accepts_frames
def animate_system_3d(sim_data: pd.DataFrame, fps: float = 20, duration: float = 10, trails: bool = False):
    """Animate the system over time in interactive 3D.

    The 3D counterpart to ``animate_system``. Builds a plotly 3D scene with
    the bodies as moving markers, Play and Pause buttons, and a time slider
    labelled in days. Optionally the full trajectories are drawn faintly
    behind. The scene can be rotated and zoomed while it plays.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` (pandas or polars). It must have the
        columns ``id``, ``x``, ``y``, ``z`` (meters) and ``time`` (seconds).
    fps : float
        Frames per second of playback. Default 20.
    duration : float
        Length of the animation in seconds of playback, not of simulated time.
        Default 10. Together with ``fps`` this sets the number of frames: about
        ``fps * duration`` evenly spaced time steps are drawn, and never fewer
        than 2. If the simulation has no more time steps than that, every time
        step becomes a frame.
    trails : bool
        If ``True``, the full trajectory of each body is drawn faintly behind
        the moving markers. The default, ``False``, shows only the markers.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with one marker per body, one animation frame per time step
        drawn, Play and Pause buttons and a time slider. It displays itself in
        a notebook. Use ``fig.show()`` in a script or ``fig.write_html(path)``
        to save it.

    Raises
    ------
    ImportError
        If plotly is not installed.
    ValueError
        If ``sim_data`` has no ``time`` column.

    See Also
    --------
    animate_system : Picks 2D or 3D from the data, and can save a 2D GIF.
    plot_system_3d : A single frozen snapshot in 3D.
    plot_orbits_3d : Whole trajectories in 3D.

    Notes
    -----
    The axis ranges are fixed to the full extent of ``sim_data`` in ``x``,
    ``y`` and ``z``, so the view does not rescale as the bodies move.

    No image encoding is involved, so the figure is built quickly even for
    many frames. The animation is played by the browser.

    Examples
    --------
    Mars on an orbit tilted 25 degrees, with the orbits drawn behind. Press
    Play, or drag the slider.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Mars", parent="Sun", i=25)
        .simulate(time_step=orb.seconds_per_day, duration=2 * orb.seconds_per_year)
    )
    orb.animate_system_3d(sim, fps=10, duration=4, trails=True)
    ```
    """
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
