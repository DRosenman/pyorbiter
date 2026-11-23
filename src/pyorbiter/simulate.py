"""The N-body engine: accelerations and the three integrators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as c
from .constants import seconds_per_hour, seconds_per_year
from ._frames import accepts_frames, is_frame, resolve_backend, to_backend
from .system import SIM_COLUMNS, OrbitSystem, _check_system, create_system

METHODS = ("verlet", "euler_cromer", "euler")


def check_sim_data(sim_data) -> None:
    """Raise TypeError unless ``sim_data`` looks like simulation output."""
    if not is_frame(sim_data) or not set(SIM_COLUMNS) <= set(sim_data.columns):
        raise TypeError("`sim_data` must be the output of `simulate_system()` "
                        "(a pandas or polars DataFrame with columns id, mass, x, y, z, vx, vy, vz, time).")


def calc_acceleration(pos: np.ndarray, mass: np.ndarray, G: float, softening: float = 0.0) -> np.ndarray:
    r"""Gravitational acceleration of every body due to every other body.

    This is the force calculation at the heart of the engine. ``simulate_system``
    calls it once at the start of a run and once per time step after that. It
    is exported so you can inspect the accelerations in a configuration, or
    write an integrator of your own on top of it.

    Parameters
    ----------
    pos : numpy.ndarray
        Positions in meters, as an array of shape ``(n, 3)`` with one row per
        body and columns x, y, z.
    mass : numpy.ndarray
        Masses in kilograms, as an array of length ``n`` in the same order as
        the rows of ``pos``.
    G : float
        Gravitational constant in m^3 kg^-1 s^-2. Use
        ``orb.gravitational_constant`` for real gravity. ``G = 0`` gives zero
        acceleration everywhere.
    softening : float
        Softening length in meters. Every squared separation ``r^2`` is
        replaced by ``r^2 + softening^2``, which keeps the acceleration finite
        when two bodies pass very close to each other. The default 0 means no
        softening.

    Returns
    -------
    numpy.ndarray
        Accelerations in m/s^2, as an array of shape ``(n, 3)`` with one row
        per body and columns x, y, z. The result is all zeros when ``G`` is 0
        or there are fewer than two bodies.

    See Also
    --------
    simulate_system : Integrate a system forward in time with these accelerations.
    get_energy : Kinetic and potential energy of a simulation.

    Notes
    -----
    The acceleration of body $j$ is the sum of Newtonian pulls from every
    other body $k$:

    $$
    \mathbf{a}_j = G \sum_{k \neq j} m_k
    \frac{\mathbf{r}_k - \mathbf{r}_j}
    {\left(|\mathbf{r}_k - \mathbf{r}_j|^2 + \epsilon^2\right)^{3/2}}
    $$

    where $\epsilon$ is the softening length. A body does not pull on itself,
    and two bodies at exactly the same position (with no softening) exert no
    force on each other, so the result never contains infinities from a zero
    separation. All pairs are computed at once with numpy, so the cost grows
    as $n^2$.

    Examples
    --------
    The Earth and the Moon, one mean lunar distance apart on the x axis. The
    Moon is pulled toward the Earth about 81 times harder than the Earth is
    pulled toward the Moon, the ratio of their masses.

    ```{python}
    import numpy as np
    import pyorbiter as orb

    pos = np.array([[0.0, 0.0, 0.0],
                    [orb.distance_earth_moon, 0.0, 0.0]])
    mass = np.array([orb.mass_earth, orb.mass_moon])

    orb.calc_acceleration(pos, mass, orb.gravitational_constant)
    ```

    Softening weakens the pull at separations comparable to the softening
    length.

    ```{python}
    import numpy as np
    import pyorbiter as orb

    pos = np.array([[0.0, 0.0, 0.0],
                    [orb.distance_earth_moon, 0.0, 0.0]])
    mass = np.array([orb.mass_earth, orb.mass_moon])

    orb.calc_acceleration(pos, mass, orb.gravitational_constant,
                          softening=orb.distance_earth_moon)
    ```
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


def simulate_system(system: OrbitSystem, time_step: float = seconds_per_hour,
                    duration: float = seconds_per_year, method: str = "verlet",
                    softening: float = 0, backend: str | None = None):
    r"""Integrate a system forward in time.

    Propagates every body in an ``OrbitSystem`` under the mutual gravity of
    all the others and records the state at each time step. The result is a
    tidy DataFrame with one row per body per time step, ready for the analysis
    and plotting functions. The default integrator is velocity Verlet, which
    keeps the total energy bounded and gives stable orbits over long runs.

    ``system.simulate(...)`` is the same function as a method, for use at the
    end of a chain.

    Parameters
    ----------
    system : OrbitSystem
        The system to simulate, built with ``create_system`` and the ``add_``
        methods, or loaded with ``load_solar_system``. It is not modified.
    time_step : float
        Time between recorded states, in seconds. This is also the integration
        step: there are no hidden substeps. The default is one hour
        (``orb.seconds_per_hour``, 3600 s). For planets around a star, one day
        (``orb.seconds_per_day``) is usually enough. For the Moon or tighter
        orbits, use an hour or less.
    duration : float
        Total simulated time in seconds. The default is one year
        (``orb.seconds_per_year``, 31557600 s). States are recorded at times
        0, ``time_step``, 2 ``time_step``, and so on, up to the last whole
        step that fits in ``duration``. A duration of 0 records only the
        initial state.
    method : str
        The integrator: ``"verlet"`` (velocity Verlet, the default),
        ``"euler_cromer"`` (semi-implicit Euler), or ``"euler"`` (explicit
        Euler). See Notes.
    softening : float
        Softening length in meters. The squared separation ``r^2`` between
        each pair of bodies is replaced by ``r^2 + softening^2`` in the force,
        which prevents a numerical blow-up when two bodies pass very close to
        each other. The default 0 means no softening. A value like 1e4
        (10 km) is reasonable for planetary systems.
    backend : str or None
        The kind of DataFrame to return: ``"pandas"`` or ``"polars"``. The
        default ``None`` uses the library chosen with ``set_backend``, which
        is pandas unless you changed it.

    Returns
    -------
    DataFrame
        A pandas DataFrame by default, or a polars DataFrame. It has one row
        per body per recorded time, ordered by time and then by the order the
        bodies were added, with columns:

        - ``id``: name of the body.
        - ``mass``: mass in kilograms.
        - ``x``, ``y``, ``z``: position in meters.
        - ``vx``, ``vy``, ``vz``: velocity in meters per second.
        - ``time``: time since the start of the run, in seconds.

        The run's ``G``, ``softening``, ``method`` and ``time_step`` are
        recorded with the result. Read them back with ``run_settings``. For a
        pandas DataFrame they are also in ``sim.attrs``. The analysis
        functions and ``continue_simulation`` pick them up from there.

    Raises
    ------
    TypeError
        If ``system`` is not an ``OrbitSystem``.
    ValueError
        If ``method`` is not one of the three integrators, if ``time_step``
        is not positive, if ``duration`` is negative, or if ``backend`` is
        not ``"pandas"`` or ``"polars"``.
    ImportError
        If the polars backend is requested and polars is not installed.

    See Also
    --------
    create_system : Start an empty system to add bodies to.
    continue_simulation : Extend a finished run from its last state.
    system_from_simulation : Rebuild a system from one time step of a run.
    calc_acceleration : The gravitational accelerations used at every step.
    run_settings : Read back the settings recorded with a run.
    conserved_quantities : Check energy and momentum conservation of a run.
    plot_orbits : Plot the trajectories.

    Notes
    -----
    Each step advances positions $\mathbf{r}$ and velocities $\mathbf{v}$ by
    a time $\Delta t$ using the accelerations $\mathbf{a}$ from Newtonian
    gravity. The three integrators differ in the order of the updates.

    **Velocity Verlet** (``"verlet"``) moves the positions with the current
    acceleration, recomputes the acceleration at the new positions, and then
    updates the velocities with the average of the old and new accelerations:

    $$
    \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t + \tfrac{1}{2} \mathbf{a}_n \Delta t^2,
    \qquad
    \mathbf{v}_{n+1} = \mathbf{v}_n + \tfrac{1}{2} (\mathbf{a}_n + \mathbf{a}_{n+1}) \Delta t
    $$

    It is second order in $\Delta t$ and symplectic, so the energy error
    oscillates within a bound instead of growing. Use it unless you have a
    reason not to.

    **Euler-Cromer** (``"euler_cromer"``) updates the velocities first and
    then moves the positions with the new velocities:

    $$
    \mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t,
    \qquad
    \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_{n+1} \Delta t
    $$

    It is first order but also symplectic, so the energy error stays bounded,
    with a larger error than Verlet at the same step.

    **Euler** (``"euler"``) moves the positions with the old velocities and
    then updates the velocities:

    $$
    \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t,
    \qquad
    \mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t
    $$

    It is first order and not symplectic. The energy grows steadily and
    orbits spiral outward. It is included for teaching, to show what goes
    wrong.

    All three use one force calculation per step, so they cost the same.
    The force calculation covers every pair of bodies, so the run time grows
    with the square of the number of bodies.

    Softening changes the force law, so a softened run is a slightly
    different physical system. ``get_energy`` uses the softening recorded
    with the run, so the energy it reports is the one that run conserves.

    Examples
    --------
    The Earth around the Sun for one year, in daily steps.

    ```{python}
    import pyorbiter as orb

    sim = orb.create_system().add_sun().add_planet("Earth", parent="Sun").simulate(
        time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    sim.head()
    ```

    The same run with each integrator, comparing the relative change in total
    energy between the first and last step. Verlet and Euler-Cromer hold the
    energy. Explicit Euler does not.

    ```{python}
    import pandas as pd
    import pyorbiter as orb

    system = orb.create_system().add_sun().add_planet("Earth", parent="Sun")

    rows = []
    for method in ["verlet", "euler_cromer", "euler"]:
        sim = orb.simulate_system(system, time_step=orb.seconds_per_day,
                                  duration=orb.seconds_per_year, method=method)
        energy = orb.get_energy(sim)["energy"]
        rows.append({"method": method,
                     "relative_energy_change": (energy.iloc[-1] - energy.iloc[0]) / abs(energy.iloc[0])})
    pd.DataFrame(rows)
    ```
    """
    _check_system(system)
    backend = resolve_backend(backend)
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
    return to_backend(result, backend)


def _run_setting(sim_data: pd.DataFrame, name: str, value, default):
    """An explicit value, else the one recorded in the run, else a default."""
    if value is not None:
        return value
    return sim_data.attrs.get(name, default)


@accepts_frames
def system_from_simulation(sim_data: pd.DataFrame, time: float | None = None,
                           G: float | None = None) -> OrbitSystem:
    """Rebuild a system from one time step of a simulation.

    Takes the positions, velocities and masses recorded at one time and turns
    them back into an ``OrbitSystem``. Use it to restart a run from a chosen
    moment, to change something partway through (add a body, remove one) and
    simulate on, or to save the final state with ``save_system``. The default
    is the last recorded step.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` or ``continue_simulation``, as a
        pandas or polars DataFrame.
    time : float or None
        The time to take the state from, in seconds since the start of the
        run. The recorded step closest to this time is used. The default
        ``None`` uses the last recorded step.
    G : float or None
        Gravitational constant of the new system, in m^3 kg^-1 s^-2. The
        default ``None`` uses the value recorded with the run, or the standard
        value ``orb.gravitational_constant`` if none was recorded.

    Returns
    -------
    OrbitSystem
        A system holding every body present at the chosen time, with its mass
        in kilograms, position in meters and velocity in meters per second.
        Bodies are in the order they appear in ``sim_data``.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``id, mass, x, y, z, vx, vy, vz, time``.

    See Also
    --------
    simulate_system : Produce the simulation this reads from.
    continue_simulation : Extend a run from its last state in one call.
    save_system : Save the rebuilt system to a file.

    Notes
    -----
    The new system starts its own clock: simulating it gives times that begin
    again at 0. ``continue_simulation`` keeps the clock running instead.

    A polars DataFrame only carries the recorded ``G`` if it is the exact
    DataFrame pyorbiter returned. If you ran with a non-standard ``G`` and
    have filtered or transformed the polars result since, pass ``G`` here.

    Examples
    --------
    The state of the Sun and the Earth at the end of a one-year run.

    ```{python}
    import pyorbiter as orb

    sim = orb.create_system().add_sun().add_planet("Earth", parent="Sun").simulate(
        time_step=orb.seconds_per_day, duration=orb.seconds_per_year)

    orb.system_from_simulation(sim)
    ```

    The state 100 days in, with Mars added before simulating on.

    ```{python}
    import pyorbiter as orb

    sim = orb.create_system().add_sun().add_planet("Earth", parent="Sun").simulate(
        time_step=orb.seconds_per_day, duration=orb.seconds_per_year)

    system = orb.system_from_simulation(sim, time=100 * orb.seconds_per_day)
    system = system.add_planet("Mars", parent="Sun")
    system.simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year).tail()
    ```
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


@accepts_frames
def continue_simulation(sim_data: pd.DataFrame, time_step: float, duration: float,
                        G: float | None = None, method: str | None = None,
                        softening: float | None = None) -> pd.DataFrame:
    """Run on from a simulation's last state and append the new rows.

    Picks up the positions and velocities at the last recorded time of
    ``sim_data``, integrates for a further ``duration``, and returns the old
    rows followed by the new ones with the clock still running. Use it to
    extend a run that turned out too short, or to change the time step partway
    through, for example a coarse step for a long approach and a fine one
    for a close encounter.

    Parameters
    ----------
    sim_data : DataFrame
        The output of ``simulate_system`` or an earlier ``continue_simulation``,
        as a pandas or polars DataFrame.
    time_step : float
        Time between recorded states in the new segment, in seconds. It does
        not have to match the step of the earlier segment. There is no
        default.
    duration : float
        Length of the new segment in seconds, counted from the last recorded
        time of ``sim_data``. There is no default.
    G : float or None
        Gravitational constant in m^3 kg^-1 s^-2. The default ``None`` uses
        the value recorded with ``sim_data``, or the standard value
        ``orb.gravitational_constant`` if none was recorded.
    method : str or None
        The integrator: ``"verlet"``, ``"euler_cromer"`` or ``"euler"``. The
        default ``None`` uses the method recorded with ``sim_data``, or
        ``"verlet"`` if none was recorded.
    softening : float or None
        Softening length in meters. The default ``None`` uses the value
        recorded with ``sim_data``, or 0 if none was recorded.

    Returns
    -------
    DataFrame
        The same kind of DataFrame as ``sim_data`` (pandas or polars): all the
        rows of ``sim_data``, then one row per body per new time step, with
        columns ``id, mass, x, y, z, vx, vy, vz, time`` in SI units. The new
        times carry on from the last time in ``sim_data``, and that last state
        is not repeated. The recorded run settings are updated to the ``G``,
        ``softening``, ``method`` and ``time_step`` used for the new segment.

    Raises
    ------
    TypeError
        If ``sim_data`` is not a pandas or polars DataFrame with the columns
        ``id, mass, x, y, z, vx, vy, vz, time``.
    ValueError
        If ``method`` is not one of the three integrators, if ``time_step``
        is not positive, or if ``duration`` is negative.

    See Also
    --------
    simulate_system : Start a run from a system.
    system_from_simulation : Rebuild a system from any time step, to modify it before running on.
    run_settings : See the settings recorded with a run.

    Notes
    -----
    A polars DataFrame only carries its run settings if it is the exact
    DataFrame pyorbiter returned. If you have filtered or transformed a polars
    result since, and the run did not use the defaults, pass ``G``, ``method``
    and ``softening`` yourself.

    Examples
    --------
    Thirty days of the Earth around the Sun in daily steps, then thirty more.

    ```{python}
    import pyorbiter as orb

    sim = orb.create_system().add_sun().add_planet("Earth", parent="Sun").simulate(
        time_step=orb.seconds_per_day, duration=30 * orb.seconds_per_day)

    longer = orb.continue_simulation(sim, time_step=orb.seconds_per_day,
                                     duration=30 * orb.seconds_per_day)
    longer.tail()
    ```

    The second segment can use a different time step. Here it switches from
    daily to hourly steps.

    ```{python}
    import pyorbiter as orb

    sim = orb.create_system().add_sun().add_planet("Earth", parent="Sun").simulate(
        time_step=orb.seconds_per_day, duration=30 * orb.seconds_per_day)

    finer = orb.continue_simulation(sim, time_step=orb.seconds_per_hour,
                                    duration=orb.seconds_per_day)
    finer[finer["id"] == "Earth"].tail()
    ```
    """
    check_sim_data(sim_data)
    G = _run_setting(sim_data, "G", G, c.gravitational_constant)
    method = _run_setting(sim_data, "method", method, "verlet")
    softening = _run_setting(sim_data, "softening", softening, 0)

    t0 = sim_data["time"].max()
    system = system_from_simulation(sim_data, time=t0, G=G)
    more = simulate_system(system, time_step=time_step, duration=duration,
                           method=method, softening=softening, backend="pandas")
    more["time"] = more["time"] + t0
    more = more.loc[more["time"] > t0]

    out = pd.concat([sim_data, more], ignore_index=True)
    out.attrs.update(G=G, softening=softening, method=method, time_step=time_step)
    return out
