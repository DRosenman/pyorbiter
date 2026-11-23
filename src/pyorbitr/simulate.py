"""The N-body engine: accelerations and the three integrators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as c
from .system import SIM_COLUMNS, OrbitSystem, _check_system, create_system

METHODS = ("verlet", "euler_cromer", "euler")


def check_sim_data(sim_data) -> None:
    if not isinstance(sim_data, pd.DataFrame) or not set(SIM_COLUMNS) <= set(sim_data.columns):
        raise TypeError("`sim_data` must be the output of `simulate_system()` "
                        "(a DataFrame with columns id, mass, x, y, z, vx, vy, vz, time).")


def calc_acceleration(pos: np.ndarray, mass: np.ndarray, G: float, softening: float = 0.0) -> np.ndarray:
    """Gravitational acceleration of every body due to every other body.

    ``pos`` is an (n, 3) array of positions and ``mass`` a length-n array.
    Returns an (n, 3) array. ``softening`` adds a length to every separation,
    ``r^2 -> r^2 + softening^2``, to tame the singularity of a close encounter.
    """
    n = len(mass)
    if G == 0 or n <= 1:
        return np.zeros((n, 3))

    # d[j, k] points from body j toward body k
    d = pos[None, :, :] - pos[:, None, :]
    r2 = np.einsum("jkc,jkc->jk", d, d) + softening * softening
    np.fill_diagonal(r2, np.inf)           # a body does not pull on itself
    r2[r2 == 0] = np.inf                   # nor do two exactly coincident bodies
    w = mass[None, :] / (r2 * np.sqrt(r2))  # m_k / r^3
    return G * np.einsum("jk,jkc->jc", w, d)


def simulate_system(system: OrbitSystem, time_step: float = c.seconds_per_hour,
                    duration: float = c.seconds_per_year, method: str = "verlet",
                    softening: float = 0) -> pd.DataFrame:
    """Integrate a system forward in time.

    Parameters
    ----------
    time_step : seconds between recorded states (default one hour).
    duration : total simulated time in seconds (default one year).
    method : ``"verlet"`` (velocity Verlet, the default; energy stays
        bounded), ``"euler_cromer"`` (semi-implicit Euler), or ``"euler"``
        (explicit Euler; energy drifts).
    softening : softening length in meters (default 0).

    Returns
    -------
    A tidy DataFrame with one row per body per time step and columns
    ``id, mass, x, y, z, vx, vy, vz, time``. The run's ``G``, ``softening``,
    ``method`` and ``time_step`` are recorded in ``DataFrame.attrs``.
    """
    _check_system(system)
    if method not in METHODS:
        raise ValueError("Method must be 'verlet', 'euler_cromer', or 'euler'.")
    if not time_step > 0:
        raise ValueError("`time_step` must be positive.")
    if duration < 0:
        raise ValueError("`duration` must be non-negative.")

    bodies = system._bodies
    n = len(bodies)
    G = system.G
    dt = float(time_step)

    mass = bodies["mass"].to_numpy(dtype=float)
    pos = bodies[["x", "y", "z"]].to_numpy(dtype=float).copy()
    vel = bodies[["vx", "vy", "vz"]].to_numpy(dtype=float).copy()

    # Recorded times 0, dt, 2 dt, ... up to the last whole step within `duration`
    n_steps = int(np.floor(duration / dt + 1e-10)) + 1
    times = np.arange(n_steps) * dt

    out = np.empty((n_steps, n, 6))
    acc = calc_acceleration(pos, mass, G, softening)

    for step in range(n_steps):
        out[step, :, :3] = pos
        out[step, :, 3:] = vel
        if step == n_steps - 1:
            break

        if method == "euler":
            pos = pos + vel * dt
            vel = vel + acc * dt
            acc = calc_acceleration(pos, mass, G, softening)
        elif method == "euler_cromer":
            vel = vel + acc * dt
            pos = pos + vel * dt
            acc = calc_acceleration(pos, mass, G, softening)
        else:  # velocity Verlet
            pos = pos + vel * dt + 0.5 * acc * dt**2
            new_acc = calc_acceleration(pos, mass, G, softening)
            vel = vel + 0.5 * (acc + new_acc) * dt
            acc = new_acc

    flat = out.reshape(n_steps * n, 6)
    result = pd.DataFrame({
        "id": np.tile(bodies["id"].to_numpy(dtype=object), n_steps),
        "mass": np.tile(mass, n_steps),
        "x": flat[:, 0], "y": flat[:, 1], "z": flat[:, 2],
        "vx": flat[:, 3], "vy": flat[:, 4], "vz": flat[:, 5],
        "time": np.repeat(times, n),
    })
    result.attrs.update(G=G, softening=softening, method=method, time_step=time_step)
    return result


def _run_setting(sim_data: pd.DataFrame, name: str, value, default):
    """An explicit value, else the one recorded in the run, else a default."""
    if value is not None:
        return value
    return sim_data.attrs.get(name, default)


def system_from_simulation(sim_data: pd.DataFrame, time: float | None = None,
                           G: float | None = None) -> OrbitSystem:
    """Rebuild a system from one time step of a run (default: the last).

    ``time`` picks the recorded step closest to that time in seconds.
    """
    check_sim_data(sim_data)
    G = _run_setting(sim_data, "G", G, c.gravitational_constant)
    times = sim_data["time"].unique()
    t_use = times.max() if time is None else times[np.argmin(np.abs(times - time))]
    snapshot = sim_data.loc[sim_data["time"] == t_use]

    system = create_system(G=G)
    for row in snapshot.itertuples(index=False):
        system = system.add_body(row.id, row.mass, x=row.x, y=row.y, z=row.z,
                                 vx=row.vx, vy=row.vy, vz=row.vz)
    return system


def continue_simulation(sim_data: pd.DataFrame, time_step: float, duration: float,
                        G: float | None = None, method: str | None = None,
                        softening: float | None = None) -> pd.DataFrame:
    """Run on from a simulation's last state and append the new rows.

    ``method``, ``softening`` and ``G`` default to those of the previous
    segment, so ``continue_simulation(sim, time_step, duration)`` simply
    extends the run.
    """
    check_sim_data(sim_data)
    G = _run_setting(sim_data, "G", G, c.gravitational_constant)
    method = _run_setting(sim_data, "method", method, "verlet")
    softening = _run_setting(sim_data, "softening", softening, 0)

    t0 = sim_data["time"].max()
    system = system_from_simulation(sim_data, time=t0, G=G)
    more = simulate_system(system, time_step=time_step, duration=duration,
                           method=method, softening=softening)
    more["time"] = more["time"] + t0
    more = more.loc[more["time"] > t0]

    out = pd.concat([sim_data, more], ignore_index=True)
    out.attrs.update(G=G, softening=softening, method=method, time_step=time_step)
    return out
