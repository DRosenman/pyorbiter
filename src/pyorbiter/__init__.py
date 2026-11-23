"""pyorbiter: a tidy N-body physics engine for orbital simulations.

The Python port of the orbitr R package (https://orbit-r.com). Build a system,
simulate it, and get back a tidy pandas DataFrame::

    import pyorbiter as orb

    sim = (
        orb.create_system()
        .add_sun()
        .add_planet("Earth", parent="Sun")
        .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
    )
    orb.plot_orbits(sim)

Units are SI throughout: meters, kilograms, seconds. Angles are in degrees.
"""

from .analysis import (
    conserved_quantities,
    get_angular_momentum,
    get_energy,
    get_momentum,
    get_orbital_elements,
    shift_reference_frame,
)
from .constants import *  # noqa: F401,F403
from .constants import __all__ as _constants
from .plotting import (
    animate_system,
    animate_system_3d,
    plot_orbits,
    plot_orbits_3d,
    plot_system,
    plot_system_3d,
)
from .simulate import calc_acceleration, continue_simulation, simulate_system, system_from_simulation
from .system import (
    OrbitSystem,
    add_body,
    add_body_keplerian,
    add_planet,
    add_sun,
    create_system,
    export_bodies,
    get_bodies,
    load_solar_system,
    load_system,
    remove_body,
    save_system,
)

__version__ = "0.1.0"

__all__ = [
    "OrbitSystem", "create_system", "add_body", "add_sun", "add_body_keplerian", "add_planet",
    "remove_body", "get_bodies", "load_solar_system", "save_system", "load_system", "export_bodies",
    "simulate_system", "continue_simulation", "system_from_simulation", "calc_acceleration",
    "shift_reference_frame", "get_orbital_elements", "get_energy", "get_momentum",
    "get_angular_momentum", "conserved_quantities",
    "plot_orbits", "plot_orbits_3d", "plot_system", "plot_system_3d",
    "animate_system", "animate_system_3d",
    *_constants,
]
