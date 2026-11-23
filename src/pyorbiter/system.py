"""Building an orbital system: bodies, Keplerian elements, and the solar system."""

from __future__ import annotations

import inspect
import json
import math
import numbers
import os
import shutil
from typing import Iterable

import pandas as pd

from . import constants as c
from .constants import gravitational_constant, mass_sun, seconds_per_hour, seconds_per_year
from ._frames import to_backend, to_pandas

BODY_COLUMNS = ["id", "mass", "x", "y", "z", "vx", "vy", "vz"]
SIM_COLUMNS = BODY_COLUMNS + ["time"]

# JPL DE440 / NASA Fact Sheet mean elements (J2000). Angles in degrees.
PLANET_CATALOG = {
    "Mercury": dict(mass=c.mass_mercury, a=c.distance_mercury_sun, e=0.2056, i=7.00, lan=48.33, arg_pe=29.12),
    "Venus": dict(mass=c.mass_venus, a=c.distance_venus_sun, e=0.0068, i=3.39, lan=76.68, arg_pe=54.88),
    "Earth": dict(mass=c.mass_earth, a=c.distance_earth_sun, e=0.0167, i=0.00, lan=-11.26, arg_pe=114.21),
    "Mars": dict(mass=c.mass_mars, a=c.distance_mars_sun, e=0.0934, i=1.85, lan=49.58, arg_pe=286.50),
    "Jupiter": dict(mass=c.mass_jupiter, a=c.distance_jupiter_sun, e=0.0489, i=1.30, lan=100.46, arg_pe=273.87),
    "Saturn": dict(mass=c.mass_saturn, a=c.distance_saturn_sun, e=0.0565, i=2.49, lan=113.72, arg_pe=339.39),
    "Uranus": dict(mass=c.mass_uranus, a=c.distance_uranus_sun, e=0.0457, i=0.77, lan=74.23, arg_pe=96.99),
    "Neptune": dict(mass=c.mass_neptune, a=c.distance_neptune_sun, e=0.0113, i=1.77, lan=131.72, arg_pe=273.19),
    "Moon": dict(mass=c.mass_moon, a=c.distance_earth_moon, e=0.0549, i=5.15, lan=125.08, arg_pe=318.15),
    "Pluto": dict(mass=c.mass_pluto, a=c.distance_pluto_sun, e=0.2488, i=17.16, lan=110.30, arg_pe=113.83),
}


def _empty_bodies() -> pd.DataFrame:
    """An empty body table with the right columns and dtypes."""
    return pd.DataFrame({col: pd.Series(dtype="object" if col == "id" else "float64") for col in BODY_COLUMNS})


def _is_number(value) -> bool:
    """True for a single real number that is not a bool and not NaN."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isnan(value)


class OrbitSystem:
    """A set of bodies and the gravitational constant that binds them.

    An ``OrbitSystem`` holds the starting state of a simulation: a table with
    one row per body (name, mass, position, velocity) and the value of ``G``.
    You normally start one with ``create_system()`` rather than calling the
    class directly, then build it up with the ``add_*`` methods.

    A system is never changed in place. Every ``add_*`` method and
    ``remove_body`` returns a new system and leaves the original untouched,
    so calls chain with dots, from an empty system through to a finished
    simulation. Printing a system shows ``G`` and the body table.

    Parameters
    ----------
    G : float
        The gravitational constant in m^3 kg^-1 s^-2. The default is the
        measured value, ``orb.gravitational_constant`` (6.6743e-11). Use
        ``G=0`` to turn gravity off.
    bodies : DataFrame, optional
        A body table with columns ``id``, ``mass``, ``x``, ``y``, ``z``,
        ``vx``, ``vy``, ``vz`` (pandas or polars). The default, ``None``,
        starts the system with no bodies. You rarely need this: add bodies
        with the methods instead, which validate their input.

    Attributes
    ----------
    G : float
        The gravitational constant the system was created with.
    bodies : DataFrame
        The body table, one row per body, with columns ``id``, ``mass``,
        ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``. A copy; the same as
        ``get_bodies()`` with no arguments.

    See Also
    --------
    create_system : The usual way to start a system.
    load_solar_system : A ready-made system of the Sun and planets.
    simulate_system : Integrate a system forward in time.

    Examples
    --------
    Build a system by chaining methods, then print it.

    ```{python}
    import pyorbiter as orb

    system = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .add_planet("Moon", parent="Earth")
    )
    system
    ```

    Each step returns a new system, so an earlier one can be reused.

    ```{python}
    import pyorbiter as orb

    sun_only = orb.create_system().add_sun()
    with_mars = sun_only.add_planet("Mars", parent="Sun")
    len(sun_only), len(with_mars)
    ```
    """

    def __init__(self, G: float = gravitational_constant, bodies: pd.DataFrame | None = None):
        self.G = G
        self._bodies = _empty_bodies() if bodies is None else to_pandas(bodies)

    # ---- inspection -----------------------------------------------------

    @property
    def bodies(self):
        """The body table: one row per body.

        A shortcut for ``get_bodies()`` with no arguments. The table is a
        copy, so editing it does not change the system.

        Returns
        -------
        DataFrame
            A pandas DataFrame by default, or a polars DataFrame if
            ``orb.set_backend("polars")`` is in effect. Columns: ``id``
            (name), ``mass`` (kg), ``x``, ``y``, ``z`` (m), ``vx``, ``vy``,
            ``vz`` (m/s).

        See Also
        --------
        OrbitSystem.get_bodies : The same table, with a choice of backend.

        Examples
        --------
        ```{python}
        import pyorbiter as orb

        system = orb.create_system().add_sun().add_planet("Mars", parent="Sun")
        system.bodies
        ```
        """
        return self.get_bodies()

    def get_bodies(self, backend: str | None = None):
        """Get the body table of a system as a DataFrame.

        Returns the current state of every body: its name, mass, position
        and velocity. Use it to inspect a system before simulating, or to
        pull the starting state into your own analysis. The table is a copy,
        so editing it does not change the system. Also available as
        ``orb.get_bodies(system, ...)``.

        Parameters
        ----------
        backend : str, optional
            The kind of DataFrame to return: ``"pandas"`` or ``"polars"``.
            The default, ``None``, uses the backend chosen with
            ``set_backend()``, which is pandas unless you changed it.

        Returns
        -------
        DataFrame
            A pandas DataFrame by default, or a polars DataFrame. One row
            per body, in the order the bodies were added, with columns:

            - ``id``: the body's name.
            - ``mass``: mass in kilograms.
            - ``x``, ``y``, ``z``: position in meters.
            - ``vx``, ``vy``, ``vz``: velocity in meters per second.

            An empty system gives a table with these columns and no rows.

        Raises
        ------
        ValueError
            If ``backend`` is not ``"pandas"`` or ``"polars"``.
        ImportError
            If ``backend`` is ``"polars"`` and polars is not installed.

        See Also
        --------
        OrbitSystem.bodies : The same table as a property.
        set_backend : Choose pandas or polars for the whole session.
        export_bodies : Write the body table to a CSV file.

        Examples
        --------
        ```{python}
        import pyorbiter as orb

        system = (
            orb.create_system()
            .add_sun()
            .add_planet("Earth", parent="Sun")
            .add_planet("Mars", parent="Sun")
        )
        system.get_bodies()
        ```

        Ask for polars for a single call.

        ```{python}
        import pyorbiter as orb

        system = orb.create_system().add_sun().add_planet("Earth", parent="Sun")
        system.get_bodies(backend="polars")
        ```
        """
        return to_backend(self._bodies.copy(), backend)

    def __len__(self) -> int:
        return len(self._bodies)

    def __repr__(self) -> str:
        width = shutil.get_terminal_size((80, 20)).columns
        label = " orbit_system "
        side = max(1, (width - len(label)) // 2)
        header = "─" * side + label + "─" * max(1, width - side - len(label))

        if self.G == gravitational_constant:
            g_label = "6.6743e-11 (standard)"
        elif self.G == 0:
            g_label = "0 (no gravity)"
        else:
            g_label = f"{self.G:g}"

        lines = [header, f"G: {g_label}"]
        if len(self) == 0:
            lines.append("Bodies: (none)")
        else:
            lines += [f"Bodies: {len(self)}", "", self._bodies.to_string(index=False)]
        return "\n".join(lines)

    def _row(self, id: str) -> pd.Series:
        return self._bodies.loc[self._bodies["id"] == id].iloc[0]

    # ---- building -------------------------------------------------------

    def add_body(self, id: str, mass: float, x: float = 0, y: float = 0, z: float = 0,
                 vx: float = 0, vy: float = 0, vz: float = 0) -> "OrbitSystem":
        """Add a body by its position and velocity.

        Adds a point mass with a given starting position and velocity (its
        Cartesian state vector). This is the most general way to place a
        body: you say exactly where it is and how it is moving. To describe
        an orbit by its shape instead, use ``add_body_keplerian()``, and for
        a real planet use ``add_planet()``. Also available as
        ``orb.add_body(system, ...)``.

        Parameters
        ----------
        id : str
            A unique name for the body, such as ``"Earth"``. It labels the
            body in the results and is how other functions refer to it.
        mass : float
            Mass in kilograms. Must be zero or positive. A mass of 0 gives a
            test particle that feels gravity but exerts none.
        x : float
            Initial position along the x axis, in meters. Default 0.
        y : float
            Initial position along the y axis, in meters. Default 0.
        z : float
            Initial position along the z axis, in meters. Default 0.
        vx : float
            Initial velocity along the x axis, in meters per second.
            Default 0.
        vy : float
            Initial velocity along the y axis, in meters per second.
            Default 0.
        vz : float
            Initial velocity along the z axis, in meters per second.
            Default 0.

        Returns
        -------
        OrbitSystem
            A new system with the body added as the last row. The original
            system is not modified.

        Raises
        ------
        ValueError
            If a body named ``id`` already exists, if ``mass`` is not a
            single number, or if ``mass`` is negative.

        See Also
        --------
        OrbitSystem.add_sun : Add the Sun with its real mass.
        OrbitSystem.add_body_keplerian : Add a body by its orbital elements.
        OrbitSystem.add_planet : Add a real solar system body by name.
        OrbitSystem.remove_body : Take a body out again.

        Examples
        --------
        The Earth and the Moon. The Earth sits at the origin and the Moon
        starts on the x axis, moving in the y direction at its mean orbital
        speed.

        ```{python}
        import pyorbiter as orb

        system = (
            orb.create_system()
            .add_body("Earth", mass=orb.mass_earth)
            .add_body("Moon", mass=orb.mass_moon,
                      x=orb.distance_earth_moon, vy=orb.speed_moon)
        )
        system
        ```

        Simulate it for 28 days.

        ```{python}
        import pyorbiter as orb

        sim = (
            orb.create_system()
            .add_body("Earth", mass=orb.mass_earth)
            .add_body("Moon", mass=orb.mass_moon,
                      x=orb.distance_earth_moon, vy=orb.speed_moon)
            .simulate(time_step=orb.seconds_per_hour, duration=28 * orb.seconds_per_day)
        )
        sim.tail()
        ```
        """
        if id in set(self._bodies["id"]):
            raise ValueError(f"Body '{id}' already exists.")
        if not _is_number(mass):
            raise ValueError("`mass` must be a single numeric value.")
        if mass < 0:
            raise ValueError("`mass` must be non-negative.")

        new = pd.DataFrame([[id, float(mass), float(x), float(y), float(z),
                             float(vx), float(vy), float(vz)]], columns=BODY_COLUMNS)
        bodies = new if len(self._bodies) == 0 else pd.concat([self._bodies, new], ignore_index=True)
        return OrbitSystem(G=self.G, bodies=bodies)

    def add_sun(self, mass: float = mass_sun, x: float = 0, y: float = 0, z: float = 0,
                vx: float = 0, vy: float = 0, vz: float = 0) -> "OrbitSystem":
        """Add the Sun to a system.

        A shortcut for ``add_body("Sun", mass=orb.mass_sun)``. The body is
        named ``"Sun"`` and by default sits at rest at the origin, which is
        the usual starting point for a planetary system. Change ``mass`` to
        model a different star. Also available as
        ``orb.add_sun(system, ...)``.

        Parameters
        ----------
        mass : float
            Mass in kilograms. The default is the mass of the Sun,
            ``orb.mass_sun`` (1.989e30 kg).
        x : float
            Initial position along the x axis, in meters. Default 0.
        y : float
            Initial position along the y axis, in meters. Default 0.
        z : float
            Initial position along the z axis, in meters. Default 0.
        vx : float
            Initial velocity along the x axis, in meters per second.
            Default 0.
        vy : float
            Initial velocity along the y axis, in meters per second.
            Default 0.
        vz : float
            Initial velocity along the z axis, in meters per second.
            Default 0.

        Returns
        -------
        OrbitSystem
            A new system with a body named ``"Sun"`` added. The original
            system is not modified.

        Raises
        ------
        ValueError
            If the system already has a body named ``"Sun"``, if ``mass`` is
            not a single number, or if ``mass`` is negative.

        See Also
        --------
        OrbitSystem.add_body : Add any body by position and velocity.
        OrbitSystem.add_planet : Add a planet in orbit around the Sun.
        load_solar_system : The Sun and all the planets in one call.

        Examples
        --------
        ```{python}
        import pyorbiter as orb

        system = orb.create_system().add_sun().add_planet("Earth", parent="Sun")
        system
        ```

        A star with half the mass of the Sun.

        ```{python}
        import pyorbiter as orb

        system = orb.create_system().add_sun(mass=0.5 * orb.mass_sun)
        system
        ```
        """
        return self.add_body("Sun", mass=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    def add_body_keplerian(self, id: str, mass: float, a: float, e: float = 0, i: float = 0,
                           lan: float = 0, arg_pe: float = 0, nu: float = 0, *,
                           parent: str) -> "OrbitSystem":
        r"""Add a body using Keplerian orbital elements.

        Describes an orbit by its size, shape and orientation instead of a
        raw position and velocity. The six elements are converted to a
        position and velocity relative to the parent body, which must
        already be in the system, and the parent's own position and velocity
        are then added on. Use this when you know what the orbit looks like
        (for example from a table of orbital elements) rather than where the
        body is at one instant. Also available as
        ``orb.add_body_keplerian(system, ...)``.

        Six numbers fully describe a Keplerian orbit:

        - ``a``, the semi-major axis, sets the size: half the longest
          diameter of the ellipse.
        - ``e``, the eccentricity, sets the shape: 0 is a circle, values
          between 0 and 1 are ellipses, values above 1 are hyperbolas.
        - ``i``, the inclination, is the tilt of the orbital plane away from
          the reference plane (the xy plane).
        - ``lan``, the longitude of the ascending node (often written
          $\Omega$), is the angle from the x axis to the point where the
          orbit crosses the reference plane going upward.
        - ``arg_pe``, the argument of periapsis (often written $\omega$), is
          the angle within the orbital plane from the ascending node to the
          point of closest approach.
        - ``nu``, the true anomaly, is where the body is along its orbit
          right now, measured as an angle from periapsis.

        Parameters
        ----------
        id : str
            A unique name for the body.
        mass : float
            Mass in kilograms. Must be zero or positive.
        a : float
            Semi-major axis in meters. Positive for an ellipse (``e < 1``)
            and negative for a hyperbola (``e > 1``). It cannot be 0.
        e : float
            Eccentricity, with no units. 0 is a circle, between 0 and 1 is
            an ellipse, above 1 is a hyperbola. Exactly 1 (a parabola) is not
            supported. Default 0.
        i : float
            Inclination in degrees. Default 0, which puts the orbit in the
            xy plane.
        lan : float
            Longitude of the ascending node in degrees. Default 0.
        arg_pe : float
            Argument of periapsis in degrees. Default 0.
        nu : float
            True anomaly in degrees. 0 is periapsis (closest to the parent)
            and 180 is apoapsis (farthest). Default 0, so the body starts at
            periapsis.
        parent : str
            The name of the body being orbited, which must already be in the
            system. The elements are defined relative to this body. Must be
            passed by keyword.

        Returns
        -------
        OrbitSystem
            A new system with the body added. Its stored state is a position
            (m) and velocity (m/s) in the same frame as the rest of the
            system. The original system is not modified.

        Raises
        ------
        ValueError
            If ``parent`` is not in the system; if ``e`` is negative or
            exactly 1; if ``a`` is 0, negative with ``e < 1``, or positive
            with ``e > 1``; if ``G`` or the
            parent's mass is zero (there is then no orbit to describe; use
            ``add_body()``); if ``nu`` lies outside the asymptotes of a
            hyperbola; or if ``id`` or ``mass`` fail the checks in
            ``add_body()``.

        See Also
        --------
        OrbitSystem.add_body : Add a body by position and velocity.
        OrbitSystem.add_planet : Add a real body with its elements filled in.
        get_orbital_elements : Recover the elements from a simulation.

        Notes
        -----
        With $\mu = G M_\text{parent}$, the distance from the parent at true
        anomaly $\nu$ is

        $$r = \frac{a (1 - e^2)}{1 + e \cos\nu}$$

        and the specific angular momentum is $h = \sqrt{\mu a (1 - e^2)}$.
        The position and velocity are worked out in the plane of the orbit,
        with x pointing at periapsis, and then rotated by ``arg_pe``, ``i``
        and ``lan`` into the system's frame. With ``i``, ``lan`` and
        ``arg_pe`` all 0, periapsis lies on the positive x axis and the body
        moves counterclockwise seen from positive z.

        Only the parent's mass enters $\mu$. The new body's own mass is
        ignored when its speed is set, which is accurate when the body is
        much lighter than its parent. For two bodies of similar mass the
        simulated orbit will differ from the elements you asked for.

        A hyperbolic orbit ($e > 1$) is an unbound flyby: the body arrives
        from far away, swings past the parent once, and leaves. By
        convention its semi-major axis is negative, and the closest approach
        is still $a (1 - e)$. A hyperbola only exists between its two
        asymptotes, so the true anomaly must satisfy
        $|\nu| < \arccos(-1/e)$.

        Examples
        --------
        Mars, from its orbital elements.

        ```{python}
        import pyorbiter as orb

        system = (
            orb.create_system()
            .add_sun()
            .add_body_keplerian(
                "Mars", mass=orb.mass_mars,
                a=orb.distance_mars_sun, e=0.0934, i=1.85,
                lan=49.58, arg_pe=286.50, nu=0,
                parent="Sun",
            )
        )
        system
        ```

        A hyperbolic flyby: ``e`` above 1 with a negative ``a``. The comet
        starts at periapsis, 1e11 m from the Sun.

        ```{python}
        import pyorbiter as orb

        system = (
            orb.create_system()
            .add_sun()
            .add_body_keplerian("Comet", mass=1e13, a=-2e11, e=1.5, parent="Sun")
        )
        system
        ```
        """
        if parent not in set(self._bodies["id"]):
            raise ValueError(f"Parent body '{parent}' not found in system.")
        if not _is_number(e) or e < 0:
            raise ValueError("`e` (eccentricity) must be >= 0.")
        if e == 1:
            raise ValueError("Parabolic orbits (eccentricity exactly 1) are not supported; "
                             "use a value slightly above or below 1.")
        if not _is_number(a) or a == 0:
            raise ValueError("`a` (semi-major axis) must be a non-zero number.")
        if e < 1 and a < 0:
            raise ValueError("`a` (semi-major axis) must be positive for an elliptical orbit (e < 1).")
        if e > 1 and a > 0:
            raise ValueError("`a` (semi-major axis) must be negative for a hyperbolic orbit (e > 1).")

        p = self._row(parent)
        mu = self.G * p["mass"]
        if not mu > 0:
            raise ValueError("Orbital elements need a gravitating parent: `G` and the parent's mass "
                             "must both be positive. Use add_body() to give a velocity directly.")

        i_rad, lan_rad, w_rad, nu_rad = (math.radians(v) for v in (i, lan, arg_pe, nu))

        if e > 1 and 1 + e * math.cos(nu_rad) <= 0:
            limit = math.degrees(math.acos(-1 / e))
            raise ValueError(f"`nu` = {nu} lies outside the hyperbola's asymptotes; "
                             f"|nu| must be less than {limit:.2f} degrees.")

        # Position and velocity in the perifocal frame (x toward periapsis)
        r = a * (1 - e**2) / (1 + e * math.cos(nu_rad))
        x_pf, y_pf = r * math.cos(nu_rad), r * math.sin(nu_rad)
        h = math.sqrt(mu * a * (1 - e**2))
        vx_pf = -(mu / h) * math.sin(nu_rad)
        vy_pf = (mu / h) * (e + math.cos(nu_rad))

        # Rotate to the inertial frame: R = Rz(-lan) Rx(-i) Rz(-arg_pe)
        cos_lan, sin_lan = math.cos(lan_rad), math.sin(lan_rad)
        cos_i, sin_i = math.cos(i_rad), math.sin(i_rad)
        cos_w, sin_w = math.cos(w_rad), math.sin(w_rad)

        r11 = cos_lan * cos_w - sin_lan * sin_w * cos_i
        r21 = sin_lan * cos_w + cos_lan * sin_w * cos_i
        r31 = sin_w * sin_i
        r12 = -cos_lan * sin_w - sin_lan * cos_w * cos_i
        r22 = -sin_lan * sin_w + cos_lan * cos_w * cos_i
        r32 = cos_w * sin_i

        return self.add_body(
            id, mass=mass,
            x=r11 * x_pf + r12 * y_pf + p["x"],
            y=r21 * x_pf + r22 * y_pf + p["y"],
            z=r31 * x_pf + r32 * y_pf + p["z"],
            vx=r11 * vx_pf + r12 * vy_pf + p["vx"],
            vy=r21 * vx_pf + r22 * vy_pf + p["vy"],
            vz=r31 * vx_pf + r32 * vy_pf + p["vz"],
        )

    def add_planet(self, name: str, *, parent: str, nu: float = 0, a: float | None = None,
                   e: float | None = None, i: float | None = None, lan: float | None = None,
                   arg_pe: float | None = None, mass: float | None = None) -> "OrbitSystem":
        """Add a known solar system body by name.

        Looks up the real mass and orbital elements of a well-known body and
        passes them to ``add_body_keplerian()``. Instead of typing out the
        mass, semi-major axis, eccentricity and angles by hand, give the name
        and the parent. Any element, or the mass, can be overridden to
        explore a "what if" while keeping the rest of the real values. Also
        available as ``orb.add_planet(system, ...)``.

        The built-in values are mean orbital elements at the J2000 epoch,
        with angles measured from the plane of Earth's orbit. The mass and
        semi-major axis are the package constants (for example
        ``orb.mass_mars`` and ``orb.distance_mars_sun``).

        | Name    | a (m)     | e      | i (deg) | lan (deg) | arg_pe (deg) |
        |---------|-----------|--------|---------|-----------|--------------|
        | Mercury | 5.791e10  | 0.2056 | 7.00    | 48.33     | 29.12        |
        | Venus   | 1.082e11  | 0.0068 | 3.39    | 76.68     | 54.88        |
        | Earth   | 1.496e11  | 0.0167 | 0.00    | -11.26    | 114.21       |
        | Mars    | 2.279e11  | 0.0934 | 1.85    | 49.58     | 286.50       |
        | Jupiter | 7.785e11  | 0.0489 | 1.30    | 100.46    | 273.87       |
        | Saturn  | 1.434e12  | 0.0565 | 2.49    | 113.72    | 339.39       |
        | Uranus  | 2.871e12  | 0.0457 | 0.77    | 74.23     | 96.99        |
        | Neptune | 4.495e12  | 0.0113 | 1.77    | 131.72    | 273.19       |
        | Moon    | 3.844e8   | 0.0549 | 5.15    | 125.08    | 318.15       |
        | Pluto   | 5.906e12  | 0.2488 | 17.16   | 110.30    | 113.83       |

        Parameters
        ----------
        name : str
            The body to add. One of ``"Mercury"``, ``"Venus"``, ``"Earth"``,
            ``"Mars"``, ``"Jupiter"``, ``"Saturn"``, ``"Uranus"``,
            ``"Neptune"``, ``"Moon"`` or ``"Pluto"``. Case does not matter:
            ``"earth"`` is stored as ``"Earth"``.
        parent : str
            The name of the body being orbited, which must already be in the
            system. For the planets and Pluto this is normally ``"Sun"``; for
            the Moon it is ``"Earth"``. Case does not matter. Must be passed
            by keyword.
        nu : float
            True anomaly in degrees: where the body starts along its orbit.
            Default 0, which is periapsis. This is the element most often
            changed. Use it to spread planets around their orbits instead of
            starting them all at closest approach.
        a : float, optional
            Semi-major axis in meters. The default, ``None``, uses the
            built-in value.
        e : float, optional
            Eccentricity. The default, ``None``, uses the built-in value.
        i : float, optional
            Inclination in degrees. The default, ``None``, uses the built-in
            value.
        lan : float, optional
            Longitude of the ascending node in degrees. The default,
            ``None``, uses the built-in value.
        arg_pe : float, optional
            Argument of periapsis in degrees. The default, ``None``, uses
            the built-in value.
        mass : float, optional
            Mass in kilograms. The default, ``None``, uses the built-in
            value.

        Returns
        -------
        OrbitSystem
            A new system with the body added under its standard name. The
            original system is not modified.

        Raises
        ------
        ValueError
            If ``parent`` is not in the system, if ``name`` is not one of
            the ten available bodies, if the body is already in the system,
            or if an override fails the checks in ``add_body_keplerian()``.

        See Also
        --------
        OrbitSystem.add_body_keplerian : Add any body by its orbital elements.
        OrbitSystem.add_sun : Add the Sun, the usual parent.
        load_solar_system : All of these bodies in one call.

        Notes
        -----
        The elements are taken relative to whichever ``parent`` you name.
        The function does not check that the pairing is the real one, so
        ``add_planet("Moon", parent="Sun")`` puts a Moon-mass body in a
        Moon-sized orbit around the Sun.

        Examples
        --------
        The inner solar system, simulated for one year and drawn from above
        (``three_d=False`` gives the flat view of the x-y plane).

        ```{python}
        import pyorbiter as orb

        sim = (
            orb.create_system()
            .add_sun()
            .add_planet("Mercury", parent="Sun")
            .add_planet("Venus", parent="Sun")
            .add_planet("Earth", parent="Sun")
            .add_planet("Mars", parent="Sun")
            .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
        )
        orb.plot_orbits(sim, three_d=False)
        ```

        Override values to ask "what if". Here Mars has a circular orbit and
        Jupiter has twice its real mass and starts a quarter of the way
        round.

        ```{python}
        import pyorbiter as orb

        system = (
            orb.create_system()
            .add_sun()
            .add_planet("Mars", parent="Sun", e=0)
            .add_planet("Jupiter", parent="Sun", mass=2 * orb.mass_jupiter, nu=90)
        )
        system
        ```
        """
        ids = list(self._bodies["id"])
        lowered = [s.lower() for s in ids]
        if parent.lower() not in lowered:
            raise ValueError(f"Parent body '{parent}' not found in system.")
        parent = ids[lowered.index(parent.lower())]

        names = {key.lower(): key for key in PLANET_CATALOG}
        if name.lower() not in names:
            raise ValueError(f"'{name}' is not a recognized body. "
                             f"Available: {', '.join(PLANET_CATALOG)}.")
        canonical = names[name.lower()]
        body = dict(PLANET_CATALOG[canonical])
        overrides = dict(mass=mass, a=a, e=e, i=i, lan=lan, arg_pe=arg_pe)
        body.update({key: value for key, value in overrides.items() if value is not None})

        return self.add_body_keplerian(canonical, parent=parent, nu=nu, **body)

    def remove_body(self, id: str | Iterable[str]) -> "OrbitSystem":
        """Remove one or more bodies from a system.

        Returns a copy of the system without the named bodies. The other
        bodies keep their positions and velocities, and ``G`` is unchanged.
        This is useful for "what if" experiments on a ready-made system,
        such as the solar system without Jupiter. Also available as
        ``orb.remove_body(system, ...)``.

        Parameters
        ----------
        id : str or iterable of str
            The name of the body to remove, or a list of names. Names are
            matched exactly, including case.

        Returns
        -------
        OrbitSystem
            A new system without the named bodies. The original system is
            not modified.

        Raises
        ------
        ValueError
            If ``id`` is an empty list or contains something that is not a
            string, or if any name is not in the system. The message lists
            the bodies that are present.

        See Also
        --------
        OrbitSystem.add_body : Add a body.
        OrbitSystem.get_bodies : See which bodies a system holds.
        load_solar_system : A ready-made system to remove bodies from.

        Notes
        -----
        Removing a body does not adjust the others. A moon whose planet is
        removed keeps the velocity it had, and will no longer be in a closed
        orbit around anything.

        Examples
        --------
        The solar system without Jupiter.

        ```{python}
        import pyorbiter as orb

        system = orb.load_solar_system().remove_body("Jupiter")
        system
        ```

        Remove several bodies at once.

        ```{python}
        import pyorbiter as orb

        inner = orb.load_solar_system(moon=False, pluto=False).remove_body(
            ["Jupiter", "Saturn", "Uranus", "Neptune"]
        )
        inner.bodies["id"].tolist()
        ```
        """
        ids = [id] if isinstance(id, str) else list(id)
        if not ids or not all(isinstance(s, str) for s in ids):
            raise ValueError("`id` must be a body name or a non-empty list of body names.")
        present = list(self._bodies["id"])
        missing = [s for s in ids if s not in present]
        if missing:
            listed = ", ".join(f"'{s}'" for s in missing)
            raise ValueError(f"Body {listed} not found in system. Present: {', '.join(present)}.")
        bodies = self._bodies.loc[~self._bodies["id"].isin(ids)].reset_index(drop=True)
        return OrbitSystem(G=self.G, bodies=bodies)

    # ---- running and saving ---------------------------------------------

    def simulate(self, time_step: float = seconds_per_hour, duration: float = seconds_per_year,
                 method: str = "verlet", softening: float = 0, backend: str | None = None):
        """Integrate the system forward in time.

        The chained form of ``simulate_system()``, which has the full
        description of each argument and of the integration methods.

        Parameters
        ----------
        time_step : float
            Seconds between recorded states. Default one hour
            (``orb.seconds_per_hour``).
        duration : float
            Total simulated time in seconds. Default one year
            (``orb.seconds_per_year``).
        method : str
            The integrator: ``"verlet"`` (the default), ``"euler_cromer"``
            or ``"euler"``.
        softening : float
            Softening length in meters. Default 0, which is plain Newtonian
            gravity.
        backend : str, optional
            ``"pandas"`` or ``"polars"``. The default, ``None``, uses the
            backend chosen with ``set_backend()``.

        Returns
        -------
        DataFrame
            A pandas DataFrame by default, or a polars DataFrame. One row
            per body per recorded time, with columns ``id``, ``mass`` (kg),
            ``x``, ``y``, ``z`` (m), ``vx``, ``vy``, ``vz`` (m/s) and
            ``time`` (s).

        See Also
        --------
        simulate_system : The same thing as a function, documented in full.
        run_settings : Read back the settings a simulation was run with.

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
        sim.head()
        ```
        """
        from .simulate import simulate_system
        return simulate_system(self, time_step=time_step, duration=duration,
                               method=method, softening=softening, backend=backend)

    def save(self, path) -> "OrbitSystem":
        """Write the system to a JSON file.

        The chained form of ``save_system()``. Read the file back with
        ``load_system()``.

        Parameters
        ----------
        path : str or path-like
            The file to write. It is overwritten if it exists.

        Returns
        -------
        OrbitSystem
            The same system, so the chain can continue.

        See Also
        --------
        save_system : The same thing as a function, documented in full.
        load_system : Read a saved system back.

        Examples
        --------
        ```{python}
        import os
        import tempfile

        import pyorbiter as orb

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "earth.json")
            orb.create_system().add_sun().add_planet("Earth", parent="Sun").save(path)
            restored = orb.load_system(path)

        restored
        ```
        """
        return save_system(self, path)

    def export_bodies(self, path) -> "OrbitSystem":
        """Write the body table to a CSV file.

        The chained form of ``export_bodies()``.

        Parameters
        ----------
        path : str or path-like
            The CSV file to write. It is overwritten if it exists.

        Returns
        -------
        OrbitSystem
            The same system, so the chain can continue.

        See Also
        --------
        export_bodies : The same thing as a function, documented in full.
        OrbitSystem.save : Save the whole system, including ``G``.

        Examples
        --------
        ```{python}
        import os
        import tempfile

        import pyorbiter as orb

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bodies.csv")
            orb.create_system().add_sun().add_planet("Earth", parent="Sun").export_bodies(path)
            with open(path) as fh:
                text = fh.read()

        print(text)
        ```
        """
        return export_bodies(self, path)


def _check_system(system) -> None:
    """Raise TypeError unless ``system`` is an ``OrbitSystem``."""
    if not isinstance(system, OrbitSystem):
        raise TypeError("Input must be an `OrbitSystem`.")


# ---- Function forms, mirroring the R API ----------------------------------

def create_system(G: float = gravitational_constant) -> OrbitSystem:
    """Create a new, empty orbital system.

    This is the first step of every simulation. It returns a system with no
    bodies and a chosen gravitational constant. Add bodies to it with
    ``add_sun()``, ``add_planet()``, ``add_body()`` or
    ``add_body_keplerian()``, then call ``simulate()``.

    Parameters
    ----------
    G : float
        The gravitational constant in m^3 kg^-1 s^-2. The default is the
        measured value, ``orb.gravitational_constant`` (6.6743e-11). Set
        ``G=0`` to turn gravity off, so that every body moves in a straight
        line at constant velocity. Other values let you work in different
        units or explore a universe with stronger or weaker gravity.

    Returns
    -------
    OrbitSystem
        An empty system, ready for bodies to be added.

    See Also
    --------
    OrbitSystem : The object this returns, and its methods.
    load_solar_system : Start from the Sun and planets instead.
    load_system : Start from a system saved to a file.

    Examples
    --------
    ```{python}
    import pyorbiter as orb

    system = orb.create_system()
    system
    ```

    With gravity turned off, a body keeps its starting velocity.

    ```{python}
    import pyorbiter as orb

    sim = (
        orb.create_system(G=0)
        .add_sun()
        .add_body("Probe", mass=1000, x=orb.distance_earth_sun, vy=orb.speed_earth)
        .simulate(time_step=orb.seconds_per_day, duration=10 * orb.seconds_per_day)
    )
    sim[sim["id"] == "Probe"].tail(3)
    ```
    """
    return OrbitSystem(G=G)


def add_body(system: OrbitSystem, id: str, mass: float, x: float = 0, y: float = 0, z: float = 0,
             vx: float = 0, vy: float = 0, vz: float = 0) -> OrbitSystem:
    _check_system(system)
    return system.add_body(id, mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)


def add_sun(system: OrbitSystem, mass: float = mass_sun, **state) -> OrbitSystem:
    _check_system(system)
    return system.add_sun(mass=mass, **state)


def add_body_keplerian(system: OrbitSystem, id: str, mass: float, a: float, e: float = 0,
                       i: float = 0, lan: float = 0, arg_pe: float = 0, nu: float = 0, *,
                       parent: str) -> OrbitSystem:
    _check_system(system)
    return system.add_body_keplerian(id, mass, a, e=e, i=i, lan=lan, arg_pe=arg_pe, nu=nu, parent=parent)


def add_planet(system: OrbitSystem, name: str, *, parent: str, **overrides) -> OrbitSystem:
    _check_system(system)
    return system.add_planet(name, parent=parent, **overrides)


def remove_body(system: OrbitSystem, id: str | Iterable[str]) -> OrbitSystem:
    _check_system(system)
    return system.remove_body(id)


def get_bodies(system: OrbitSystem, backend: str | None = None):
    _check_system(system)
    return system.get_bodies(backend=backend)


def _function_doc(method_doc: str | None) -> str | None:
    """A method's docstring, reworded for the function that takes the system first.

    Each function below is documented once, on the :class:`OrbitSystem` method
    it calls. This adds the ``system`` parameter to that docstring's
    ``Parameters`` section, creating the section if the method takes no
    arguments.
    """
    if not method_doc:
        return method_doc
    doc = inspect.cleandoc(method_doc)
    entry = "system : OrbitSystem\n    The system to work on. It is not modified.\n"
    header = "Parameters\n----------\n"
    if header in doc:
        return doc.replace(header, header + entry, 1)
    summary, _, rest = doc.partition("\n\n")
    for marker in ("Returns\n-------", "Examples\n--------", "See Also\n--------", "Notes\n-----"):
        if marker in rest:
            before, _, after = rest.partition(marker)
            return f"{summary}\n\n{before}{header}{entry}\n{marker}{after}"
    return f"{summary}\n\n{rest}\n\n{header}{entry}".replace("\n\n\n\n", "\n\n")


for _fn, _method in [(add_body, OrbitSystem.add_body), (add_sun, OrbitSystem.add_sun),
                     (add_body_keplerian, OrbitSystem.add_body_keplerian),
                     (add_planet, OrbitSystem.add_planet), (remove_body, OrbitSystem.remove_body),
                     (get_bodies, OrbitSystem.get_bodies)]:
    _fn.__doc__ = _function_doc(_method.__doc__)


def load_solar_system(moon: bool = True, pluto: bool = True) -> OrbitSystem:
    """Load the solar system, ready to simulate.

    Builds a system with the Sun at rest at the origin and the eight planets
    on their real orbits, using the same masses and orbital elements as
    ``add_planet()``. The Moon (in orbit around the Earth) and Pluto are
    included unless you switch them off. Every body starts at periapsis, its
    closest point to the body it orbits.

    The result is an ordinary system, so you can add to it, remove bodies
    from it, or simulate it straight away.

    Parameters
    ----------
    moon : bool
        Whether to include the Moon, in orbit around the Earth. Default
        ``True``.
    pluto : bool
        Whether to include Pluto. Default ``True``.

    Returns
    -------
    OrbitSystem
        A system with the bodies in this order: Sun, Mercury, Venus, Earth,
        Mars, Jupiter, Saturn, Uranus, Neptune, then Moon and Pluto if
        requested. That is 11 bodies with the defaults.

    See Also
    --------
    OrbitSystem.add_planet : Add these bodies one at a time, with overrides.
    OrbitSystem.remove_body : Drop bodies from the loaded system.
    create_system : Start from an empty system instead.

    Notes
    -----
    Because every body starts at periapsis, the planets are not where they
    were on any particular date. The orbits have the right size, shape and
    orientation, but this is not an ephemeris.

    A time step that suits the planets is too coarse for the Moon, whose
    orbit takes about 27 days. Use a step of an hour or so if the Moon is
    included, or pass ``moon=False``.

    Examples
    --------
    ```{python}
    import pyorbiter as orb

    system = orb.load_solar_system()
    system
    ```

    The planets only, simulated for one year and drawn from above
    (``three_d=False`` gives the flat view of the x-y plane).

    ```{python}
    import pyorbiter as orb

    sim = orb.load_solar_system(moon=False, pluto=False).simulate(
        time_step=orb.seconds_per_day, duration=orb.seconds_per_year
    )
    orb.plot_orbits(sim, three_d=False)
    ```
    """
    system = create_system().add_body("Sun", mass=c.mass_sun)
    for planet in ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"):
        system = system.add_planet(planet, parent="Sun")
    if moon:
        system = system.add_planet("Moon", parent="Earth")
    if pluto:
        system = system.add_planet("Pluto", parent="Sun")
    return system


def save_system(system: OrbitSystem, path) -> OrbitSystem:
    """Save a system to a JSON file.

    Writes the system's gravitational constant and its body table to a file
    that ``load_system()`` can read back. Use it to keep a set of starting
    conditions, or to share one with someone else. The file is plain JSON,
    so it can be opened in a text editor. Also available as the chained
    method ``system.save(path)``.

    Parameters
    ----------
    system : OrbitSystem
        The system to save.
    path : str or path-like
        The file to write, for example ``"my_system.json"``. It is
        overwritten if it exists.

    Returns
    -------
    OrbitSystem
        The same system, unchanged, so the call can sit in the middle of a
        chain.

    Raises
    ------
    TypeError
        If ``system`` is not an ``OrbitSystem``.

    See Also
    --------
    load_system : Read a saved system back.
    export_bodies : Write only the body table, as a CSV file.

    Notes
    -----
    The file holds four entries: ``format`` (the text
    ``"pyorbiter.orbit_system"``), ``version`` (1), ``G``, and ``bodies``, a
    list with one record per body giving ``id``, ``mass``, ``x``, ``y``,
    ``z``, ``vx``, ``vy`` and ``vz``. Only the starting state is saved, not
    the results of a simulation.

    Examples
    --------
    Save a system to a temporary folder and load it again.

    ```{python}
    import os
    import tempfile

    import pyorbiter as orb

    system = orb.create_system().add_sun().add_planet("Earth", parent="Sun")

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "earth.json")
        orb.save_system(system, path)
        restored = orb.load_system(path)

    restored
    ```
    """
    _check_system(system)
    payload = {"format": "pyorbiter.orbit_system", "version": 1, "G": system.G,
               "bodies": system._bodies.to_dict(orient="records")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return system


def load_system(path) -> OrbitSystem:
    """Load a system from a JSON file.

    Reads a file written by ``save_system()`` and rebuilds the system: the
    same gravitational constant and the same bodies, in the same order, with
    the same masses, positions and velocities.

    Parameters
    ----------
    path : str or path-like
        The file to read.

    Returns
    -------
    OrbitSystem
        The saved system.

    Raises
    ------
    FileNotFoundError
        If there is no file at ``path``.
    ValueError
        If the file is JSON but was not written by ``save_system()``.

    See Also
    --------
    save_system : Write the file this reads.
    create_system : Start from an empty system instead.

    Examples
    --------
    ```{python}
    import os
    import tempfile

    import pyorbiter as orb

    system = orb.create_system().add_sun().add_planet("Mars", parent="Sun")

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "mars.json")
        system.save(path)
        restored = orb.load_system(path)

    restored.bodies.equals(system.bodies)
    ```
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    # "pyorbitr.orbit_system" is the format name from before the package was renamed.
    formats = ("pyorbiter.orbit_system", "pyorbitr.orbit_system")
    if not isinstance(payload, dict) or payload.get("format") not in formats:
        raise ValueError("The file does not contain an orbit system.")
    system = create_system(G=payload["G"])
    for body in payload["bodies"]:
        system = system.add_body(**body)
    return system


def export_bodies(system: OrbitSystem, path) -> OrbitSystem:
    """Export the body table of a system to a CSV file.

    Writes one row per body with its name, mass, position and velocity. A
    CSV file opens in a spreadsheet or any other tool, which makes this the
    way to take a system's starting state out of Python. The gravitational
    constant is not written, so to save a system you intend to load again,
    use ``save_system()``. Also available as the chained method
    ``system.export_bodies(path)``.

    Parameters
    ----------
    system : OrbitSystem
        The system whose bodies to export.
    path : str or path-like
        The CSV file to write. It is overwritten if it exists.

    Returns
    -------
    OrbitSystem
        The same system, unchanged, so the call can sit in the middle of a
        chain.

    Raises
    ------
    TypeError
        If ``system`` is not an ``OrbitSystem``.

    See Also
    --------
    save_system : Save the whole system, including ``G``, to reload later.
    OrbitSystem.get_bodies : Get the same table as a DataFrame.

    Notes
    -----
    The file has a header row, ``id,mass,x,y,z,vx,vy,vz``, and no index
    column. Mass is in kilograms, positions in meters and velocities in
    meters per second.

    Examples
    --------
    ```{python}
    import os
    import tempfile

    import pyorbiter as orb

    system = orb.create_system().add_sun().add_planet("Earth", parent="Sun")

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "bodies.csv")
        orb.export_bodies(system, path)
        with open(path) as fh:
            text = fh.read()

    print(text)
    ```
    """
    _check_system(system)
    system._bodies.to_csv(path, index=False)
    return system
