"""The polars backend must give the same numbers as the pandas one."""

import gc

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyorbiter as orb

pl = pytest.importorskip("polars")

DAY = orb.seconds_per_day


@pytest.fixture(autouse=True)
def pandas_by_default():
    orb.set_backend("pandas")
    yield
    orb.set_backend("pandas")


@pytest.fixture
def system():
    # Inclined, so every state column and orbital element is exercised
    return (orb.create_system()
            .add_sun()
            .add_planet("Earth", parent="Sun")
            .add_body_keplerian("Comet", mass=1e14, a=3e11, e=0.7, i=30, parent="Sun"))


@pytest.fixture
def sim_pd(system):
    return system.simulate(time_step=DAY, duration=30 * DAY)


@pytest.fixture
def sim_pl(system):
    return system.simulate(time_step=DAY, duration=30 * DAY, backend="polars")


def assert_same(polars_frame, pandas_frame):
    """A polars DataFrame holding exactly what a pandas one does."""
    assert isinstance(polars_frame, pl.DataFrame)
    assert isinstance(pandas_frame, pd.DataFrame)
    assert polars_frame.columns == list(pandas_frame.columns)
    assert polars_frame.height == len(pandas_frame)
    for name in polars_frame.columns:
        got, want = polars_frame[name].to_numpy(), pandas_frame[name].to_numpy()
        if name == "id":
            assert list(got) == list(want)
        else:
            np.testing.assert_array_equal(got, want)


# ---- Choosing a backend ----------------------------------------------------

def test_pandas_is_the_default(system):
    assert orb.get_backend() == "pandas"
    assert isinstance(system.simulate(time_step=DAY, duration=DAY), pd.DataFrame)
    assert isinstance(system.bodies, pd.DataFrame)


def test_backend_argument(sim_pl, sim_pd):
    assert_same(sim_pl, sim_pd)
    assert sim_pl.schema["id"] == pl.String
    assert all(sim_pl.schema[name] == pl.Float64 for name in sim_pl.columns if name != "id")


def test_set_backend(system, sim_pd):
    orb.set_backend("polars")
    assert orb.get_backend() == "polars"
    assert_same(system.simulate(time_step=DAY, duration=30 * DAY), sim_pd)
    assert_same(orb.simulate_system(system, time_step=DAY, duration=30 * DAY), sim_pd)
    assert isinstance(system.bodies, pl.DataFrame)
    # a single call can still ask for pandas
    assert isinstance(system.simulate(time_step=DAY, duration=DAY, backend="pandas"), pd.DataFrame)


def test_unknown_backend_is_rejected(system):
    with pytest.raises(ValueError, match="backend"):
        orb.set_backend("dask")
    with pytest.raises(ValueError, match="backend"):
        system.simulate(time_step=DAY, duration=DAY, backend="arrow")
    assert orb.get_backend() == "pandas"


def test_get_bodies(system):
    assert_same(system.get_bodies(backend="polars"), system.bodies)
    assert_same(orb.get_bodies(system, backend="polars"), system.bodies)
    assert system.get_bodies(backend="polars").schema["id"] == pl.String
    assert orb.create_system().get_bodies(backend="polars").schema["id"] == pl.String


# ---- Run settings ----------------------------------------------------------

def test_run_settings(sim_pl, sim_pd):
    expected = dict(G=orb.gravitational_constant, softening=0, method="verlet", time_step=DAY)
    assert orb.run_settings(sim_pl) == expected
    assert orb.run_settings(sim_pd) == expected == sim_pd.attrs


def test_run_settings_do_not_follow_a_derived_frame(sim_pl):
    assert orb.run_settings(sim_pl.filter(pl.col("id") == "Earth")) == {}
    assert orb.run_settings(pl.DataFrame({"x": [1.0]})) == {}


def test_run_settings_are_forgotten_with_the_frame(system):
    from pyorbiter import _frames

    before = len(_frames._run_settings)
    frame = system.simulate(time_step=DAY, duration=DAY, backend="polars")
    assert len(_frames._run_settings) == before + 1
    del frame
    gc.collect()
    assert len(_frames._run_settings) == before


def test_recorded_settings_are_used(system):
    """A run with its own G and softening is analyzed with them, in either backend."""
    odd = orb.create_system(G=2 * orb.gravitational_constant)
    for row in system.bodies.itertuples(index=False):
        odd = odd.add_body(**row._asdict())
    as_pd = odd.simulate(time_step=DAY, duration=10 * DAY, softening=1e9)
    as_pl = odd.simulate(time_step=DAY, duration=10 * DAY, softening=1e9, backend="polars")

    assert_same(orb.get_energy(as_pl), orb.get_energy(as_pd))
    standard = orb.get_energy(as_pd, G=orb.gravitational_constant, softening=0)
    assert not np.allclose(orb.get_energy(as_pl)["potential"].to_numpy(), standard["potential"])
    assert_same(orb.get_orbital_elements(as_pl, "Earth", "Sun"),
                orb.get_orbital_elements(as_pd, "Earth", "Sun"))


# ---- Analysis --------------------------------------------------------------

@pytest.mark.parametrize("analyze", [
    orb.get_energy,
    orb.get_momentum,
    orb.get_angular_momentum,
    orb.conserved_quantities,
    lambda sim: orb.get_orbital_elements(sim, "Earth", "Sun"),
    lambda sim: orb.get_orbital_elements(sim, "Comet", "Sun", mu=1.3e20),
    lambda sim: orb.shift_reference_frame(sim, "Earth"),
    lambda sim: orb.shift_reference_frame(sim, "Earth", keep_center=False),
    lambda sim: orb.shift_reference_frame(sim, "barycenter"),
])
def test_analysis_matches_pandas(analyze, sim_pl, sim_pd):
    assert_same(analyze(sim_pl), analyze(sim_pd))


def test_analysis_returns_what_it_was_given(sim_pl, sim_pd):
    """The input decides the output, whatever the default backend is."""
    orb.set_backend("polars")
    assert isinstance(orb.get_energy(sim_pd), pd.DataFrame)
    assert isinstance(orb.shift_reference_frame(sim_pd, "Earth"), pd.DataFrame)
    orb.set_backend("pandas")
    assert isinstance(orb.get_energy(sim_pl), pl.DataFrame)
    assert isinstance(orb.shift_reference_frame(sim_pl, "Earth"), pl.DataFrame)


def test_shift_reference_frame_keeps_run_settings(sim_pl):
    shifted = orb.shift_reference_frame(sim_pl, "Earth")
    assert orb.run_settings(shifted) == orb.run_settings(sim_pl)


def test_lazy_frames_are_accepted(sim_pl, sim_pd):
    assert_same(orb.get_momentum(sim_pl.lazy()), orb.get_momentum(sim_pd))


def test_errors_are_the_same(sim_pl):
    with pytest.raises(ValueError, match="not found"):
        orb.shift_reference_frame(sim_pl, "Vulcan")
    with pytest.raises(ValueError, match="not found"):
        orb.get_orbital_elements(sim_pl, "Vulcan", "Sun")
    with pytest.raises(TypeError, match="simulate_system"):
        orb.get_energy(sim_pl.select("id", "mass"))


# ---- Continuing ------------------------------------------------------------

def test_continue_simulation(system, sim_pl, sim_pd):
    more_pl = orb.continue_simulation(sim_pl, time_step=DAY, duration=10 * DAY)
    more_pd = orb.continue_simulation(sim_pd, time_step=DAY, duration=10 * DAY)
    assert_same(more_pl, more_pd)
    assert orb.run_settings(more_pl) == more_pd.attrs

    whole = system.simulate(time_step=DAY, duration=40 * DAY)
    np.testing.assert_allclose(more_pl.select("x", "y", "z").to_numpy(),
                               whole[["x", "y", "z"]].to_numpy(), rtol=1e-12, atol=1e-6)


def test_continue_simulation_ignores_the_default_backend(sim_pd, sim_pl):
    orb.set_backend("polars")
    assert isinstance(orb.continue_simulation(sim_pd, time_step=DAY, duration=DAY), pd.DataFrame)
    orb.set_backend("pandas")
    assert isinstance(orb.continue_simulation(sim_pl, time_step=DAY, duration=DAY), pl.DataFrame)


def test_continue_simulation_keeps_the_method(system):
    euler = system.simulate(time_step=DAY, duration=5 * DAY, method="euler", backend="polars")
    longer = orb.continue_simulation(euler, time_step=DAY, duration=5 * DAY)
    assert orb.run_settings(longer)["method"] == "euler"
    whole = system.simulate(time_step=DAY, duration=10 * DAY, method="euler")
    np.testing.assert_allclose(longer["x"].to_numpy(), whole["x"].to_numpy(), rtol=1e-12, atol=1e-6)


def test_system_from_simulation(system, sim_pl):
    rebuilt = orb.system_from_simulation(sim_pl, time=0)
    pd.testing.assert_frame_equal(rebuilt.bodies, system.bodies)


# ---- Plotting --------------------------------------------------------------

def test_plots_accept_polars(sim_pl, tmp_path):
    import matplotlib.pyplot as plt

    flat = orb.plot_orbits(sim_pl, three_d=False)
    assert len(flat.lines) == 3
    snapshot = orb.plot_system(sim_pl, time=10 * DAY, trails=True, three_d=False)
    assert snapshot is not None
    plt.close("all")

    pytest.importorskip("plotly")
    assert len(orb.plot_orbits(sim_pl).data) == 3          # the comet's orbit is inclined
    assert len(orb.plot_orbits_3d(sim_pl).data) == 3
    assert orb.plot_system_3d(sim_pl, time=10 * DAY) is not None
    assert orb.animate_system_3d(sim_pl, fps=2, duration=2) is not None
