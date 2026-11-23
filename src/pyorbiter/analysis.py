"""Analysis of a simulation: reference frames, orbital elements, conserved quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as c
from ._frames import accepts_frames
from .simulate import _run_setting, check_sim_data

_STATE = ["x", "y", "z", "vx", "vy", "vz"]


@accepts_frames
def shift_reference_frame(sim_data: pd.DataFrame, center_id: str, keep_center: bool = True) -> pd.DataFrame:
    """Re-center a simulation on one body, or on the center of mass.

    ``center_id`` is a body id, or ``"barycenter"`` for the mass-weighted
    center of the whole system. Positions *and* velocities are shifted. With
    ``keep_center=False`` the central body's (all-zero) rows are dropped.
    """
    ids = sim_data["id"]
    out = sim_data.copy()

    if center_id == "barycenter" and not (ids == "barycenter").any():
        total = sim_data.groupby("time")["mass"].transform("sum")
        for col in _STATE:
            weighted = (sim_data["mass"] * sim_data[col]).groupby(sim_data["time"]).transform("sum")
            out[col] = sim_data[col] - weighted / total
        out.attrs = dict(sim_data.attrs)
        return out

    if not (ids == center_id).any():
        raise ValueError(f"Body '{center_id}' not found in the simulation data.")

    ref = sim_data.loc[ids == center_id].set_index("time")
    for col in _STATE:
        out[col] = sim_data[col].to_numpy() - sim_data["time"].map(ref[col]).to_numpy()
    if not keep_center:
        out = out.loc[out["id"] != center_id].reset_index(drop=True)
    out.attrs = dict(sim_data.attrs)
    return out


@accepts_frames
def get_orbital_elements(sim_data: pd.DataFrame, body: str, parent: str,
                         mu: float | None = None, G: float | None = None) -> pd.DataFrame:
    """Osculating orbital elements of ``body`` about ``parent`` at every time step.

    Returns a DataFrame with columns ``time, a, e, i, lan, arg_pe, nu, period``
    (meters, degrees, seconds). ``period`` is NaN for an unbound orbit.
    ``mu`` defaults to ``G * M_parent``; pass ``G * (M_parent + m_body)`` for
    the exact two-body value when the body's mass matters.
    """
    check_sim_data(sim_data)
    ids = sim_data["id"]
    for name in (body, parent):
        if not (ids == name).any():
            raise ValueError(f"Body '{name}' not found in the simulation data.")
    if body == parent:
        raise ValueError("`body` and `parent` must be different bodies.")
    G = _run_setting(sim_data, "G", G, c.gravitational_constant)

    b = sim_data.loc[ids == body, ["time"] + _STATE]
    p = sim_data.loc[ids == parent, ["time", "mass"] + _STATE]
    p = p.rename(columns={col: f"p_{col}" for col in ["mass"] + _STATE})
    d = b.merge(p, on="time").sort_values("time", kind="stable")
    if mu is None:
        mu = G * d["p_mass"].iloc[0]

    rx, ry, rz = (d[k].to_numpy() - d[f"p_{k}"].to_numpy() for k in ("x", "y", "z"))
    ux, uy, uz = (d[k].to_numpy() - d[f"p_{k}"].to_numpy() for k in ("vx", "vy", "vz"))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        v2 = ux**2 + uy**2 + uz**2

        # Specific angular momentum h = r x v, and its unit vector
        hx, hy, hz = ry * uz - rz * uy, rz * ux - rx * uz, rx * uy - ry * ux
        h = np.sqrt(hx**2 + hy**2 + hz**2)
        hhx, hhy, hhz = hx / h, hy / h, hz / h

        # Node vector n = z x h; for an orbit in the reference plane use +x
        nx, ny = -hy, hx
        n = np.sqrt(nx**2 + ny**2)
        flat = n < 1e-12 * h
        nx = np.where(flat, 1.0, nx)
        ny = np.where(flat, 0.0, ny)
        n = np.where(flat, 1.0, n)
        nx, ny = nx / n, ny / n

        # Eccentricity vector; for a circular orbit measure angles from the node
        ex = (uy * hz - uz * hy) / mu - rx / r
        ey = (uz * hx - ux * hz) / mu - ry / r
        ez = (ux * hy - uy * hx) / mu - rz / r
        e = np.sqrt(ex**2 + ey**2 + ez**2)
        circular = e < 1e-12
        edx = np.where(circular, nx, ex / e)
        edy = np.where(circular, ny, ey / e)
        edz = np.where(circular, 0.0, ez / e)

        a = 1 / (2 / r - v2 / mu)
        i = np.arccos(np.clip(hz / h, -1, 1))
        lan = np.arctan2(ny, nx)
        arg_pe = np.arctan2(
            (ny * edz) * hhx + (-nx * edz) * hhy + (nx * edy - ny * edx) * hhz,
            nx * edx + ny * edy,
        )
        arg_pe = np.where(circular, 0.0, arg_pe)
        nu = np.arctan2(
            (edy * rz - edz * ry) * hhx + (edz * rx - edx * rz) * hhy + (edx * ry - edy * rx) * hhz,
            edx * rx + edy * ry + edz * rz,
        )
        period = np.where((e < 1) & (a > 0), 2 * np.pi * np.sqrt(np.abs(a) ** 3 / mu), np.nan)

    def to_deg(theta):
        return np.degrees(theta) % 360

    return pd.DataFrame({
        "time": d["time"].to_numpy(), "a": a, "e": e, "i": np.degrees(i),
        "lan": to_deg(lan), "arg_pe": to_deg(arg_pe), "nu": to_deg(nu), "period": period,
    })


def _potential(sim_data: pd.DataFrame, G: float, softening: float) -> pd.Series:
    """Total gravitational potential energy at each time, indexed by time."""
    ordered = sim_data.sort_values("time", kind="stable")
    sizes = ordered.groupby("time", sort=True).size()
    times = sizes.index
    if G == 0:
        return pd.Series(0.0, index=times)

    def pairs(pos, mass):
        # pos (T, n, 3), mass (T, n) -> (T,) sum over pairs i < j
        total = np.zeros(pos.shape[0])
        for i in range(pos.shape[1] - 1):
            sep = pos[:, i + 1:, :] - pos[:, i:i + 1, :]
            dist = np.sqrt((sep**2).sum(axis=2) + softening**2)
            total -= G * (mass[:, i:i + 1] * mass[:, i + 1:] / dist).sum(axis=1)
        return total

    if sizes.nunique() == 1:
        # The usual case: the same number of bodies at every step
        n = int(sizes.iloc[0])
        pos = ordered[["x", "y", "z"]].to_numpy(dtype=float).reshape(-1, n, 3)
        mass = ordered["mass"].to_numpy(dtype=float).reshape(-1, n)
        with np.errstate(divide="ignore"):
            return pd.Series(pairs(pos, mass), index=times)

    values = []
    for _, group in ordered.groupby("time", sort=True):
        pos = group[["x", "y", "z"]].to_numpy(dtype=float)[None]
        with np.errstate(divide="ignore"):
            values.append(pairs(pos, group["mass"].to_numpy(dtype=float)[None])[0])
    return pd.Series(values, index=times, dtype=float)


@accepts_frames
def get_energy(sim_data: pd.DataFrame, G: float | None = None, softening: float | None = None) -> pd.DataFrame:
    """Total kinetic, potential and mechanical energy (joules) at every time step.

    ``G`` and ``softening`` default to the values recorded in the run.
    """
    check_sim_data(sim_data)
    G = _run_setting(sim_data, "G", G, c.gravitational_constant)
    softening = _run_setting(sim_data, "softening", softening, 0)

    ke = 0.5 * sim_data["mass"] * (sim_data["vx"]**2 + sim_data["vy"]**2 + sim_data["vz"]**2)
    kinetic = ke.groupby(sim_data["time"], sort=True).sum()
    potential = _potential(sim_data, G, softening)
    return pd.DataFrame({"time": kinetic.index.to_numpy(), "kinetic": kinetic.to_numpy(),
                         "potential": potential.to_numpy(),
                         "energy": kinetic.to_numpy() + potential.to_numpy()})


@accepts_frames
def get_momentum(sim_data: pd.DataFrame) -> pd.DataFrame:
    """Total linear momentum ``px, py, pz`` (kg m/s) at every time step."""
    check_sim_data(sim_data)
    m = sim_data["mass"]
    parts = pd.DataFrame({"time": sim_data["time"], "px": m * sim_data["vx"],
                          "py": m * sim_data["vy"], "pz": m * sim_data["vz"]})
    return parts.groupby("time", sort=True).sum().reset_index()


@accepts_frames
def get_angular_momentum(sim_data: pd.DataFrame) -> pd.DataFrame:
    """Total angular momentum ``Lx, Ly, Lz`` (kg m^2/s) about the origin at every time step."""
    check_sim_data(sim_data)
    s = sim_data
    parts = pd.DataFrame({
        "time": s["time"],
        "Lx": s["mass"] * (s["y"] * s["vz"] - s["z"] * s["vy"]),
        "Ly": s["mass"] * (s["z"] * s["vx"] - s["x"] * s["vz"]),
        "Lz": s["mass"] * (s["x"] * s["vy"] - s["y"] * s["vx"]),
    })
    return parts.groupby("time", sort=True).sum().reset_index()


@accepts_frames
def conserved_quantities(sim_data: pd.DataFrame, G: float | None = None,
                         softening: float | None = None) -> pd.DataFrame:
    """Energy, momentum and angular momentum at every step, with relative errors.

    Adds ``energy_error`` (signed, relative to the initial energy),
    ``momentum_error`` and ``angular_momentum_error`` (size of the drift,
    relative to the initial scale). An error is NaN when its scale is zero.
    Run this on the raw, inertial simulation, not a shifted frame.
    """
    check_sim_data(sim_data)
    out = (get_energy(sim_data, G=G, softening=softening)
           .merge(get_momentum(sim_data), on="time")
           .merge(get_angular_momentum(sim_data), on="time"))

    first = sim_data.loc[sim_data["time"] == sim_data["time"].min()]
    p_scale0 = (first["mass"] * np.sqrt(first["vx"]**2 + first["vy"]**2 + first["vz"]**2)).sum()

    E0 = out["energy"].iloc[0]
    P0 = out[["px", "py", "pz"]].iloc[0].to_numpy()
    L0 = out[["Lx", "Ly", "Lz"]].iloc[0].to_numpy()

    def safe_divide(numerator, scale):
        if not np.isfinite(scale) or scale == 0:
            return np.full(len(out), np.nan)
        return np.asarray(numerator) / scale

    out["energy_error"] = safe_divide(out["energy"] - E0, abs(E0))
    out["momentum_error"] = safe_divide(
        np.sqrt(((out[["px", "py", "pz"]].to_numpy() - P0) ** 2).sum(axis=1)), p_scale0)
    out["angular_momentum_error"] = safe_divide(
        np.sqrt(((out[["Lx", "Ly", "Lz"]].to_numpy() - L0) ** 2).sum(axis=1)),
        np.sqrt((L0**2).sum()))
    return out
