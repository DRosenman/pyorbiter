"""Building an orbital system: bodies, Keplerian elements, and the solar system."""

from __future__ import annotations

import json
import math
import numbers
import os
import shutil
from typing import Iterable

import pandas as pd

from . import constants as c

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
    return pd.DataFrame({col: pd.Series(dtype="object" if col == "id" else "float64") for col in BODY_COLUMNS})


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isnan(value)


class OrbitSystem:
    """A set of bodies and the gravitational constant that binds them.

    Create one with :func:`create_system`. Every ``add_*`` / ``remove_body``
    method returns a *new* system and leaves the original untouched, so calls
    chain the way the R pipe does::

        sim = (
            create_system()
            .add_sun()
            .add_planet("Earth", parent="Sun")
            .simulate(time_step=seconds_per_day, duration=seconds_per_year)
        )
    """

    def __init__(self, G: float = c.gravitational_constant, bodies: pd.DataFrame | None = None):
        self.G = G
        self._bodies = _empty_bodies() if bodies is None else bodies

    # ---- inspection -----------------------------------------------------

    @property
    def bodies(self) -> pd.DataFrame:
        """The body table: one row per body, columns id, mass, x, y, z, vx, vy, vz."""
        return self._bodies.copy()

    def get_bodies(self) -> pd.DataFrame:
        """The body table as a DataFrame (a copy)."""
        return self.bodies

    def __len__(self) -> int:
        return len(self._bodies)

    def __repr__(self) -> str:
        width = shutil.get_terminal_size((80, 20)).columns
        label = " orbit_system "
        side = max(1, (width - len(label)) // 2)
        header = "─" * side + label + "─" * max(1, width - side - len(label))

        if self.G == c.gravitational_constant:
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
        """Add a body by its Cartesian state (meters, m/s, kg)."""
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

    def add_sun(self, mass: float = c.mass_sun, x: float = 0, y: float = 0, z: float = 0,
                vx: float = 0, vy: float = 0, vz: float = 0) -> "OrbitSystem":
        """Add the Sun, by default at rest at the origin."""
        return self.add_body("Sun", mass=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    def add_body_keplerian(self, id: str, mass: float, a: float, e: float = 0, i: float = 0,
                           lan: float = 0, arg_pe: float = 0, nu: float = 0, *,
                           parent: str) -> "OrbitSystem":
        """Add a body by its orbital elements about ``parent``.

        ``a`` is the semi-major axis in meters, ``e`` the eccentricity, and
        ``i``, ``lan``, ``arg_pe``, ``nu`` (inclination, longitude of ascending
        node, argument of periapsis, true anomaly) are in degrees. For a
        hyperbolic orbit pass ``e > 1`` with a negative ``a``.
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
        """Add a named solar-system body using its real mass and J2000 elements.

        Available: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus,
        Neptune, Moon, Pluto (case-insensitive). Any element or the mass can be
        overridden.
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
        """Drop one or more bodies by name."""
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

    def simulate(self, time_step: float = c.seconds_per_hour, duration: float = c.seconds_per_year,
                 method: str = "verlet", softening: float = 0) -> pd.DataFrame:
        """Integrate the system forward in time. See :func:`simulate_system`."""
        from .simulate import simulate_system
        return simulate_system(self, time_step=time_step, duration=duration,
                               method=method, softening=softening)

    def save(self, path) -> "OrbitSystem":
        """Write the system to a JSON file. See :func:`save_system`."""
        return save_system(self, path)

    def export_bodies(self, path) -> "OrbitSystem":
        """Write the body table to a CSV file."""
        return export_bodies(self, path)


def _check_system(system) -> None:
    if not isinstance(system, OrbitSystem):
        raise TypeError("Input must be an `OrbitSystem`.")


# ---- Function forms, mirroring the R API ----------------------------------

def create_system(G: float = c.gravitational_constant) -> OrbitSystem:
    """Start an empty system. ``G = 0`` turns gravity off."""
    return OrbitSystem(G=G)


def add_body(system: OrbitSystem, id: str, mass: float, x: float = 0, y: float = 0, z: float = 0,
             vx: float = 0, vy: float = 0, vz: float = 0) -> OrbitSystem:
    _check_system(system)
    return system.add_body(id, mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)


def add_sun(system: OrbitSystem, mass: float = c.mass_sun, **state) -> OrbitSystem:
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


def get_bodies(system: OrbitSystem) -> pd.DataFrame:
    _check_system(system)
    return system.bodies


for _fn, _method in [(add_body, OrbitSystem.add_body), (add_sun, OrbitSystem.add_sun),
                     (add_body_keplerian, OrbitSystem.add_body_keplerian),
                     (add_planet, OrbitSystem.add_planet), (remove_body, OrbitSystem.remove_body),
                     (get_bodies, OrbitSystem.get_bodies)]:
    _fn.__doc__ = _method.__doc__


def load_solar_system(moon: bool = True, pluto: bool = True) -> OrbitSystem:
    """The Sun and eight planets, plus (optionally) the Moon and Pluto."""
    system = create_system().add_body("Sun", mass=c.mass_sun)
    for planet in ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"):
        system = system.add_planet(planet, parent="Sun")
    if moon:
        system = system.add_planet("Moon", parent="Earth")
    if pluto:
        system = system.add_planet("Pluto", parent="Sun")
    return system


def save_system(system: OrbitSystem, path) -> OrbitSystem:
    """Write a system to a JSON file. Read it back with :func:`load_system`."""
    _check_system(system)
    payload = {"format": "pyorbitr.orbit_system", "version": 1, "G": system.G,
               "bodies": system._bodies.to_dict(orient="records")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return system


def load_system(path) -> OrbitSystem:
    """Read a system written by :func:`save_system`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or payload.get("format") != "pyorbitr.orbit_system":
        raise ValueError("The file does not contain an orbit system.")
    system = create_system(G=payload["G"])
    for body in payload["bodies"]:
        system = system.add_body(**body)
    return system


def export_bodies(system: OrbitSystem, path) -> OrbitSystem:
    """Write the body table to a CSV file."""
    _check_system(system)
    system._bodies.to_csv(path, index=False)
    return system
