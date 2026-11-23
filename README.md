# pyorbitr

A tidy N-body physics engine for building and visualizing orbital simulations in Python.

`pyorbitr` is the Python port of the [orbitr](https://orbit-r.com) R package. Build a system of bodies, integrate it forward in time, and get back a tidy pandas DataFrame with one row per body per time step, ready for pandas, matplotlib, or plotly.

```python
import pyorbitr as orb

sim = (
    orb.create_system()
    .add_sun()
    .add_planet("Earth", parent="Sun")
    .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
)

orb.plot_orbits(sim)
```

## Installation

From a clone of this repository:

```bash
pip install -e .            # the package
pip install -e ".[plotly]"  # plus interactive 3D plots
pip install -e ".[dev]"     # plus pytest, plotly, pillow
```

Requires Python 3.10+, numpy, pandas, and matplotlib.

## A tour

Units are SI throughout (meters, kilograms, seconds); angles are in degrees.

### Build

Every `add_*` method returns a new system, so calls chain the way the R pipe does.

```python
system = (
    orb.create_system()
    .add_body("Sun", mass=orb.mass_sun)
    .add_body("Earth", mass=orb.mass_earth, x=orb.distance_earth_sun, vy=orb.speed_earth)
    # by orbital elements; e > 1 with a < 0 gives a hyperbola
    .add_body_keplerian("Comet", mass=1e14, a=3e11, e=0.7, i=30, parent="Sun")
)

solar = orb.load_solar_system()              # Sun, 8 planets, Moon, Pluto
inner = solar.remove_body(["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
```

### Simulate

```python
sim = solar.simulate(time_step=orb.seconds_per_hour, duration=orb.seconds_per_year)
sim = orb.continue_simulation(sim, time_step=orb.seconds_per_hour, duration=orb.seconds_per_year)
```

`method` is `"verlet"` (velocity Verlet, the default), `"euler_cromer"`, or `"euler"`. `softening` tames close encounters. The result has columns `id, mass, x, y, z, vx, vy, vz, time`, and the run's `G`, `softening`, `method`, and `time_step` are kept in `sim.attrs`.

### Analyze

```python
orb.shift_reference_frame(sim, "Earth")        # or "barycenter"
orb.get_orbital_elements(sim, "Moon", "Earth") # a, e, i, lan, arg_pe, nu, period at every step
orb.conserved_quantities(sim)                  # energy, momentum, angular momentum, and their errors
```

These take and return DataFrames, so they compose with `DataFrame.pipe`:

```python
sim.pipe(orb.shift_reference_frame, "Earth").pipe(orb.plot_orbits)
```

### Plot

```python
orb.plot_orbits(sim)                     # full trajectories
orb.plot_system(sim, time=30 * orb.seconds_per_day, trails=True)   # a snapshot
orb.animate_system(sim, path="orbit.gif")
```

2D plots are matplotlib (and return the `Axes`). If there is any motion in `z` you get an interactive plotly figure instead; force either with `three_d=True/False`.

## Coming from R

| orbitr (R) | pyorbitr (Python) |
|:--|:--|
| `create_system() \|> add_body(...)` | `orb.create_system().add_body(...)` (or `orb.add_body(system, ...)`) |
| `simulate_system(system, ...)` | `system.simulate(...)` or `orb.simulate_system(system, ...)` |
| `sim \|> shift_reference_frame("Earth")` | `orb.shift_reference_frame(sim, "Earth")` or `sim.pipe(...)` |
| a tibble with attributes | a pandas DataFrame with `.attrs` |
| `save_system()` writes `.rds` | `save_system()` writes JSON |
| ggplot2 / gganimate | matplotlib |
| the C++ kernel (`use_cpp`) | a vectorized numpy kernel |

Function names, argument names, constants, and defaults are otherwise the same. The test suite checks pyorbitr against numbers produced by orbitr 1.0.0 (`tests/reference/`); the two agree to about 8 significant figures or better.

## The simulator app

`app/` holds a [Shiny for Python](https://shiny.posit.co/py/) simulator with preset scenarios, a snapshot slider, conservation diagnostics, orbital elements, and the code to reproduce each run.

```bash
pip install -e ".[app]"
shiny run app/app.py
```

## Tests

```bash
pytest
```

## License

MIT
