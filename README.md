# pyorbiter

A tidy N-body physics engine for building and visualizing orbital simulations in Python.

Build a system of bodies, integrate it forward in time, and get back a tidy DataFrame with one row per body per time step, ready for pandas or polars, matplotlib, or plotly. `pyorbiter` is the Python sibling of the [orbitr](https://orbit-r.com) R package.

Full documentation, with a function reference and articles: <https://pyorbiter.com>.

```python
import pyorbiter as orb

sim = (
    orb.create_system()
    .add_sun()
    .add_planet("Earth", parent="Sun")
    .simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year)
)

orb.plot_orbits(sim)
```

## Installation

```bash
pip install pyorbiter              # the package
pip install "pyorbiter[plotly]"    # plus interactive 3D plots
pip install "pyorbiter[polars]"    # plus polars DataFrames
pip install "pyorbiter[gif]"       # plus saving animations as GIFs
```

To work on the package itself, clone the repository and install it in editable mode with the development tools:

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+, numpy, pandas, and matplotlib. polars is optional.

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

`method` is `"verlet"` (velocity Verlet, the default), `"euler_cromer"`, or `"euler"`. `softening` tames close encounters. The result has columns `id, mass, x, y, z, vx, vy, vz, time`, and the run's `G`, `softening`, `method`, and `time_step` are kept in `sim.attrs` (or, for either kind of DataFrame, `orb.run_settings(sim)`).

### pandas or polars

Results are pandas DataFrames by default. To get polars DataFrames instead, install the extra (`pip install "pyorbiter[polars]"`) and switch the backend once:

```python
orb.set_backend("polars")

sim = solar.simulate(time_step=orb.seconds_per_hour, duration=orb.seconds_per_year)   # a polars DataFrame
energy = orb.get_energy(sim)                                                           # also polars
```

Or ask for it in a single call with `backend="polars"` on `simulate()`, `simulate_system()`, or `get_bodies()`. Every function that takes a simulation (analysis, plotting, `continue_simulation()`) accepts either kind and returns the kind it was given, so the two can be mixed freely. A polars `LazyFrame` is accepted too; it is collected, and a polars `DataFrame` comes back. Columns of your own that you add to a polars simulation are carried through untouched.

A polars DataFrame has no `.attrs`, so pyorbiter remembers the run's settings for the exact DataFrame it returned; `orb.run_settings(sim)` shows them. A frame you derive yourself (`sim.filter(...)`, `sort`, `with_columns`, a join, a file you read back) no longer carries them. If the run did not use the defaults, pass `G` and `softening` to the analysis functions, and `G`, `softening`, and `method` to `continue_simulation()`, explicitly. (pandas keeps `.attrs` through most of those operations, so this only matters for polars.)

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

| orbitr (R) | pyorbiter (Python) |
|:--|:--|
| `create_system() \|> add_body(...)` | `orb.create_system().add_body(...)` (or `orb.add_body(system, ...)`) |
| `simulate_system(system, ...)` | `system.simulate(...)` or `orb.simulate_system(system, ...)` |
| `sim \|> shift_reference_frame("Earth")` | `orb.shift_reference_frame(sim, "Earth")` or `sim.pipe(...)` |
| a tibble with attributes | a pandas DataFrame with `.attrs`, or a polars DataFrame |
| `save_system()` writes `.rds` | `save_system()` writes JSON |
| ggplot2 / gganimate | matplotlib |
| the C++ kernel (`use_cpp`) | a vectorized numpy kernel |

Function names, argument names, constants, and defaults are otherwise the same. The test suite checks pyorbiter against numbers produced by orbitr 1.0.0 (`tests/reference/`); the two agree to about 8 significant figures or better.

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
