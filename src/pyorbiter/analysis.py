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
    r"""Re-center a simulation on one body, or on the center of mass.

    Recalculates the positions and velocities of every body relative to a
    chosen body, at every time step. This anchors the camera to that body: it
    sits at the origin with zero velocity for the whole run, and everything
    else is described as seen from it. Use it to look at a moon from its
    planet, or to view a system from its barycenter, where the star's small
    wobble becomes visible.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system()``: a pandas or polars DataFrame with
        columns ``id``, ``mass``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``
        and ``time``. Positions are in meters, velocities in meters per
        second, masses in kilograms, time in seconds.
    center_id : str
        The id of the body to use as the new origin, or ``"barycenter"`` for
        the mass-weighted center of the whole system. If a body is actually
        named ``"barycenter"``, that body is used.
    keep_center : bool
        Whether to keep the central body's rows, which are zero in all six
        state columns. The default ``True`` keeps them. ``False`` drops them.
        It has no effect when shifting to the computed barycenter, which is
        not a body and has no rows.

    Returns
    -------
    DataFrame
        The same kind of DataFrame it was given (pandas in, pandas out; polars
        in, polars out), with the same columns. ``x``, ``y``, ``z`` (meters)
        and ``vx``, ``vy``, ``vz`` (meters per second) are replaced by their
        values relative to the new origin. ``id``, ``mass`` and ``time`` are
        unchanged. The recorded run settings are carried over.

    Raises
    ------
    ValueError
        If ``center_id`` is not ``"barycenter"`` and no body with that id is
        in ``sim_data``.

    See Also
    --------
    simulate_system : produce the simulation to shift.
    plot_orbits : plot the shifted trajectories.
    get_orbital_elements : elements of one body about another, with no shift needed.
    conserved_quantities : should be run on the unshifted simulation.

    Notes
    -----
    The barycenter at each time step is

    $$\mathbf{R} = \frac{\sum_i m_i \mathbf{r}_i}{\sum_i m_i}, \qquad
    \mathbf{V} = \frac{\sum_i m_i \mathbf{v}_i}{\sum_i m_i}$$

    summed over the bodies present at that step. With no outside forces the
    barycenter moves in a straight line at constant speed, so the barycentric
    frame is inertial and total momentum is zero in it. A frame centered on a
    body is not inertial, because the body accelerates. Momentum and angular
    momentum are not conserved in such a frame.

    Examples
    --------
    Shift a Sun, Earth and Jupiter run to its barycenter. The Sun is no longer
    at the origin: Jupiter holds it several hundred thousand kilometers away.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Jupiter", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.shift_reference_frame(sim, "barycenter").head()
    ```

    Center a Sun, Earth and Moon run on Earth, drop Earth's rows, and plot the
    Moon's path around the origin.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Moon", parent="Earth")
        .simulate(time_step=orb.seconds_per_hour, duration=28 * orb.seconds_per_day)
    )
    earth_view = orb.shift_reference_frame(sim, "Earth", keep_center=False)
    orb.plot_orbits(earth_view[earth_view["id"] == "Moon"], three_d=False)
    ```
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
    r"""Osculating orbital elements of one body about another at every time step.

    Converts the position and velocity of ``body`` relative to ``parent`` into
    the six classical Keplerian elements, once per time step. These are the
    osculating elements: the two-body orbit the body would follow from that
    instant if every other body vanished. In a pure two-body run they stay
    nearly constant, apart from the true anomaly. In a many-body run they
    drift, and the drift shows how the other bodies perturb the orbit.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system()``: a pandas or polars DataFrame with
        columns ``id``, ``mass``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``
        and ``time``, in SI units. It does not need to be shifted first: the
        relative position and velocity are computed here.
    body : str
        The id of the orbiting body.
    parent : str
        The id of the body it orbits.
    mu : float, optional
        The gravitational parameter of the orbit in m^3/s^2. The default
        ``None`` uses ``G * M_parent``, with the parent's mass taken from its
        first row. Pass ``G * (M_parent + m_body)`` for the exact two-body
        value when the body's mass is not negligible.
    G : float, optional
        The gravitational constant in m^3 kg^-1 s^-2, used only to build the
        default ``mu``. The default ``None`` uses the value recorded in the
        run (see ``run_settings()``), or ``gravitational_constant`` if none
        is recorded.

    Returns
    -------
    DataFrame
        The same kind of DataFrame it was given (pandas or polars), with one
        row per time step at which both bodies are present, sorted by time,
        and these columns:

        - ``time``: time in seconds.
        - ``a``: semi-major axis in meters, from the vis-viva equation.
          Negative for a hyperbolic orbit.
        - ``e``: eccentricity, dimensionless. 0 is a circle, between 0 and 1
          an ellipse, 1 or more an unbound orbit.
        - ``i``: inclination to the x-y plane in degrees, from 0 to 180.
          Values above 90 are retrograde orbits.
        - ``lan``: longitude of the ascending node in degrees, measured in
          the x-y plane from the +x axis.
        - ``arg_pe``: argument of periapsis in degrees, measured in the orbit
          plane from the ascending node to periapsis, in the direction of
          motion.
        - ``nu``: true anomaly in degrees, measured in the orbit plane from
          periapsis to the body, in the direction of motion.
        - ``period``: orbital period in seconds,
          $2\pi\sqrt{a^3/\mu}$. ``NaN`` for an unbound orbit.

        ``lan``, ``arg_pe`` and ``nu`` are wrapped into the range 0 to 360.
        An angle a rounding error below zero wraps to just under 360, and can
        print as 360. The result does not carry the run settings of ``sim_data``.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``simulate_system()`` produces.
    ValueError
        If ``body`` or ``parent`` is not an id in ``sim_data``, or if they
        are the same body.

    See Also
    --------
    OrbitSystem.add_body_keplerian : the inverse, place a body from its elements.
    OrbitSystem.add_planet : add a solar-system body from its real elements.
    shift_reference_frame : re-center positions and velocities on a body.
    conserved_quantities : check the accuracy of a run.

    Notes
    -----
    Some elements are undefined for special orbits. These conventions keep
    every angle defined:

    - Equatorial orbit (the orbit lies in the x-y plane, so there is no
      ascending node): the node is taken to be the +x axis. ``lan`` is 0 and
      ``arg_pe`` is the angle from +x to periapsis.
    - Circular orbit (``e`` below 1e-12, so there is no periapsis):
      ``arg_pe`` is 0 and ``nu`` is measured from the ascending node. It is
      then the argument of latitude.
    - Circular and equatorial: ``lan`` and ``arg_pe`` are both 0 and ``nu``
      is the angle from the +x axis.

    An orbit that is close to one of these cases, but not exactly in it, has
    a poorly determined ``lan`` or ``arg_pe``. Small perturbations can then
    swing those angles by large amounts while the orbit itself barely
    changes.

    The elements come from the specific angular momentum
    $\mathbf{h} = \mathbf{r} \times \mathbf{v}$, the eccentricity vector
    $\mathbf{e} = (\mathbf{v} \times \mathbf{h})/\mu - \mathbf{r}/r$, and the
    vis-viva equation $1/a = 2/r - v^2/\mu$, where $\mathbf{r}$ and
    $\mathbf{v}$ are the position and velocity of ``body`` relative to
    ``parent``.

    Examples
    --------
    Earth's elements about the Sun over one year. With only two bodies they
    hold nearly steady while the true anomaly ``nu`` advances. This orbit lies
    in the x-y plane, so ``i`` and ``lan`` are 0 by the equatorial convention
    in the Notes.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.get_orbital_elements(sim, "Earth", "Sun").head()
    ```

    Place a probe from its elements, simulate, and recover those elements
    from the positions and velocities. They match to within the integration
    error of a one-day time step.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_body_keplerian("Probe", mass=1000, a=2e11, e=0.3, i=20, lan=40,
                            arg_pe=60, nu=0, parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=100 * orb.seconds_per_day)
    )
    orb.get_orbital_elements(sim, "Probe", "Sun").iloc[::25].round(4)
    ```
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
    r"""Total kinetic, potential and mechanical energy at every time step.

    Sums the kinetic energy of every body and the gravitational potential
    energy of every pair of bodies. In an isolated system the total is
    constant, so any change in the ``energy`` column over a run is numerical
    error from the integrator. Use it to judge whether a time step is small
    enough.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system()``: a pandas or polars DataFrame with
        columns ``id``, ``mass``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``
        and ``time``, in SI units.
    G : float, optional
        The gravitational constant in m^3 kg^-1 s^-2. The default ``None``
        uses the value recorded in the run (see ``run_settings()``), or
        ``gravitational_constant`` if none is recorded.
    softening : float, optional
        The softening length in meters, which should match the one the run
        used. The default ``None`` uses the value recorded in the run, or 0
        if none is recorded.

    Returns
    -------
    DataFrame
        The same kind of DataFrame it was given (pandas or polars), with one
        row per time step, sorted by time, and these columns:

        - ``time``: time in seconds.
        - ``kinetic``: total kinetic energy in joules.
        - ``potential``: total gravitational potential energy in joules.
          Zero or negative.
        - ``energy``: ``kinetic + potential``, in joules.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``simulate_system()`` produces.

    See Also
    --------
    conserved_quantities : energy, momentum and angular momentum with relative errors.
    get_momentum : total linear momentum.
    get_angular_momentum : total angular momentum.
    run_settings : the ``G`` and ``softening`` recorded with a run.

    Notes
    -----
    The energies are

    $$K = \sum_i \tfrac{1}{2} m_i v_i^2, \qquad
    U = -\sum_{i<j} \frac{G m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}$$

    where $r_{ij}$ is the distance between bodies $i$ and $j$ and $\epsilon$
    is the softening length. A bound system has negative total energy.

    Kinetic energy depends on the reference frame. The total is conserved in
    the simulation's own frame and in the barycentric frame, but not in a
    frame centered on an accelerating body.

    A polars DataFrame only remembers its run settings while it is the exact
    object pyorbiter returned. After a ``filter`` or ``with_columns``, pass
    ``G`` and ``softening`` yourself if the run did not use the defaults.

    Examples
    --------
    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.get_energy(sim).head()
    ```
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
    r"""Total linear momentum at every time step.

    Sums mass times velocity over every body. Gravity acts in equal and
    opposite pairs, so the total momentum of an isolated system is constant.
    In a simulation it should hold steady to within floating-point rounding.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system()``: a pandas or polars DataFrame with
        columns ``id``, ``mass``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``
        and ``time``, in SI units.

    Returns
    -------
    DataFrame
        The same kind of DataFrame it was given (pandas or polars), with one
        row per time step, sorted by time, and these columns:

        - ``time``: time in seconds.
        - ``px``, ``py``, ``pz``: the components of the total momentum in
          kg m/s.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``simulate_system()`` produces.

    See Also
    --------
    conserved_quantities : energy, momentum and angular momentum with relative errors.
    get_angular_momentum : total angular momentum.
    get_energy : total energy.
    shift_reference_frame : move to the barycentric frame, where the total is zero.

    Notes
    -----
    The total momentum is $\mathbf{P} = \sum_i m_i \mathbf{v}_i$. Its value
    depends on the reference frame. A system built around a star at rest has a
    nonzero total, because the planets move and the star does not, and the
    whole system drifts slowly as a result. In the barycentric frame the total
    is zero.

    Examples
    --------
    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.get_momentum(sim).head()
    ```
    """
    check_sim_data(sim_data)
    m = sim_data["mass"]
    parts = pd.DataFrame({"time": sim_data["time"], "px": m * sim_data["vx"],
                          "py": m * sim_data["vy"], "pz": m * sim_data["vz"]})
    return parts.groupby("time", sort=True).sum().reset_index()


@accepts_frames
def get_angular_momentum(sim_data: pd.DataFrame) -> pd.DataFrame:
    r"""Total angular momentum about the origin at every time step.

    Sums $m\,\mathbf{r} \times \mathbf{v}$ over every body. Gravity between
    two bodies acts along the line joining them, so it exerts no net torque
    and the total angular momentum of an isolated system is constant. For a
    system that orbits counterclockwise in the x-y plane the vector points
    along +z.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system()``: a pandas or polars DataFrame with
        columns ``id``, ``mass``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``
        and ``time``, in SI units.

    Returns
    -------
    DataFrame
        The same kind of DataFrame it was given (pandas or polars), with one
        row per time step, sorted by time, and these columns:

        - ``time``: time in seconds.
        - ``Lx``, ``Ly``, ``Lz``: the components of the total angular
          momentum in kg m^2/s.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``simulate_system()`` produces.

    See Also
    --------
    conserved_quantities : energy, momentum and angular momentum with relative errors.
    get_momentum : total linear momentum.
    get_energy : total energy.
    shift_reference_frame : change the origin the angular momentum is measured about.

    Notes
    -----
    The total is $\mathbf{L} = \sum_i m_i\,\mathbf{r}_i \times \mathbf{v}_i$,
    measured about the origin of the coordinates in ``sim_data``. Its value
    depends on that origin. It is conserved in the simulation's own frame and
    in the barycentric frame, but not in a frame centered on an accelerating
    body.

    Examples
    --------
    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.get_angular_momentum(sim).head()
    ```
    """
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
    r"""Energy, momentum and angular momentum at every step, with relative errors.

    Combines ``get_energy()``, ``get_momentum()`` and
    ``get_angular_momentum()`` into one table and adds three error columns
    that measure how far each quantity has moved from its starting value. An
    isolated system conserves all three exactly, so the errors measure the
    integrator, not the physics. This is the quickest check that a time step
    is small enough and that a method suits the problem.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system()``: a pandas or polars DataFrame with
        columns ``id``, ``mass``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``
        and ``time``, in SI units. Use the simulation as it came out, not a
        frame shifted onto a body.
    G : float, optional
        The gravitational constant in m^3 kg^-1 s^-2. The default ``None``
        uses the value recorded in the run (see ``run_settings()``), or
        ``gravitational_constant`` if none is recorded.
    softening : float, optional
        The softening length in meters, which should match the one the run
        used. The default ``None`` uses the value recorded in the run, or 0
        if none is recorded.

    Returns
    -------
    DataFrame
        The same kind of DataFrame it was given (pandas or polars), with one
        row per time step, sorted by time, and these columns:

        - ``time``: time in seconds.
        - ``kinetic``, ``potential``, ``energy``: energies in joules, as in
          ``get_energy()``.
        - ``px``, ``py``, ``pz``: total linear momentum in kg m/s.
        - ``Lx``, ``Ly``, ``Lz``: total angular momentum about the origin in
          kg m^2/s.
        - ``energy_error``: ``(energy - E0) / abs(E0)``, where ``E0`` is the
          energy at the first time step. Signed and dimensionless.
        - ``momentum_error``: the length of the change in the momentum
          vector since the first step, divided by the sum of every body's
          ``mass * speed`` at the first step. Never negative, dimensionless.
        - ``angular_momentum_error``: the length of the change in the
          angular momentum vector since the first step, divided by the length
          of the angular momentum vector at the first step. Never negative,
          dimensionless.

        An error column is all ``NaN`` when the quantity it is divided by is
        zero or not finite.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``simulate_system()`` produces.

    See Also
    --------
    get_energy : the energy columns on their own.
    get_momentum : the momentum columns on their own.
    get_angular_momentum : the angular momentum columns on their own.
    simulate_system : choose the integration ``method`` and ``time_step``.
    run_settings : the ``G`` and ``softening`` recorded with a run.

    Notes
    -----
    In symbols, with a subscript 0 for the first time step:

    $$\frac{E - E_0}{|E_0|}, \qquad
    \frac{|\mathbf{P} - \mathbf{P}_0|}{\sum_i m_i |\mathbf{v}_{i,0}|}, \qquad
    \frac{|\mathbf{L} - \mathbf{L}_0|}{|\mathbf{L}_0|}$$

    for the energy, momentum and angular momentum errors in that order.

    The momentum error is scaled by the sum of the individual momenta, not by
    $|\mathbf{P}_0|$, because the total momentum is often zero or close to it
    and would make a poor yardstick.

    Run this on the simulation as it came out of ``simulate_system()``. A
    frame centered on a body is not inertial, and momentum and angular
    momentum are not conserved in it.

    Examples
    --------
    One year of the Earth's orbit with the default velocity Verlet method.
    The error columns are at the right.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.conserved_quantities(sim).tail()
    ```

    Compare the largest energy error of the three integration methods on the
    same system with the same time step.

    ```{python}
    import pandas as pd
    import pyorbiter as orb

    system = orb.create_system().add_sun().add_planet("Earth", parent="Sun")

    rows = []
    for method in ["verlet", "euler_cromer", "euler"]:
        sim = system.simulate(time_step=orb.seconds_per_day,
                              duration=orb.seconds_per_year, method=method)
        error = orb.conserved_quantities(sim)["energy_error"].abs().max()
        rows.append({"method": method, "max_energy_error": error})
    pd.DataFrame(rows)
    ```
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
