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

    anim = orb.animate_system(sim_pl, fps=2, duration=2, three_d=False)
    assert anim is not None
    plt.close("all")
    pytest.importorskip("PIL")
    orb.animate_system(sim_pl, fps=2, duration=1, three_d=False, path=tmp_path / "orbit.gif")
    assert (tmp_path / "orbit.gif").stat().st_size > 0
    plt.close("all")


# ---- Your own columns ------------------------------------------------------

@pytest.fixture
def annotated(sim_pl):
    """A simulation carrying columns pyorbiter knows nothing about."""
    return sim_pl.with_columns(
        label=pl.when(pl.col("id") == "Sun").then(pl.lit("star")).otherwise(None),
        kind=pl.col("id").cast(pl.Categorical),
        day=pl.date(2020, 1, 1),
        count=pl.when(pl.col("id") == "Earth").then(None).otherwise(pl.lit(7, dtype=pl.Int32)),
        pair=pl.struct(a=pl.col("x"), b=pl.col("id")),
    )


def test_extra_columns_survive(annotated, sim_pl):
    shifted = orb.shift_reference_frame(annotated, "Earth")
    assert shifted.columns == annotated.columns
    assert shifted.schema == annotated.schema
    extras = ["label", "kind", "day", "count", "pair"]
    assert shifted.select(extras).equals(annotated.select(extras))
    assert shifted.select(sim_pl.columns).equals(orb.shift_reference_frame(sim_pl, "Earth"))

    dropped = orb.shift_reference_frame(annotated, "Earth", keep_center=False)
    kept = annotated.filter(pl.col("id") != "Earth")
    assert dropped.select(extras).equals(kept.select(extras))

    # functions that build a new table are not bothered by them
    assert orb.get_energy(annotated).equals(orb.get_energy(sim_pl))
    orb.plot_orbits(annotated, three_d=False)


def test_extra_columns_are_null_on_continued_rows(annotated, sim_pl):
    longer = orb.continue_simulation(annotated, time_step=DAY, duration=5 * DAY)
    plain = orb.continue_simulation(sim_pl, time_step=DAY, duration=5 * DAY)
    assert longer.schema == annotated.schema
    assert longer.select(sim_pl.columns).equals(plain)
    old, new = longer.head(annotated.height), longer.tail(longer.height - annotated.height)
    assert old.equals(annotated)
    assert new.height == 3 * 5
    assert new.select(pl.col("label", "kind", "day", "count").is_null().all()).row(0) == (True,) * 4


def test_input_types_are_kept(sim_pl):
    """Rows handed back keep the types they arrived with."""
    odd = sim_pl.with_columns(pl.col("id").cast(pl.Categorical), pl.col("mass").cast(pl.Float32))
    assert orb.shift_reference_frame(odd, "Earth").schema == odd.schema


def test_null_text_does_not_break_conversion():
    from pyorbiter import _frames

    frame = pd.DataFrame({"id": ["Sun", None, "Earth"], "mass": [1.0, 2.0, 3.0]})
    assert _frames.to_polars(frame)["id"].to_list() == ["Sun", None, "Earth"]


# ---- Edge cases ------------------------------------------------------------

def test_system_from_a_polars_body_table(system):
    orb.set_backend("polars")
    copy = orb.OrbitSystem(G=system.G, bodies=system.bodies)
    assert isinstance(system.bodies, pl.DataFrame)
    assert isinstance(orb.get_bodies(system), pl.DataFrame)
    assert isinstance(system.get_bodies(), pl.DataFrame)
    assert "Earth" in repr(copy)
    assert len(copy.add_body("Probe", mass=1.0, x=1e9)) == 4
    assert_same(copy.simulate(time_step=DAY, duration=DAY),
                system.simulate(time_step=DAY, duration=DAY, backend="pandas"))


def test_empty_simulation_keeps_numeric_columns(sim_pl):
    energy = orb.get_energy(sim_pl.clear())
    assert energy.height == 0
    assert all(dtype == pl.Float64 for dtype in energy.schema.values())


def test_missing_polars_is_reported_and_changes_nothing(system, monkeypatch):
    import sys

    for name in [name for name in sys.modules if name == "polars" or name.startswith("polars.")]:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"pyorbiter\[polars\]"):
        orb.set_backend("polars")
    assert orb.get_backend() == "pandas"
    with pytest.raises(ImportError, match=r"pyorbiter\[polars\]"):
        system.simulate(time_step=DAY, duration=DAY, backend="polars")
    assert isinstance(system.simulate(time_step=DAY, duration=DAY), pd.DataFrame)
    assert isinstance(system.bodies, pd.DataFrame)
