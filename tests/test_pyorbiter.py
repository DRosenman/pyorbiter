"""Behavior tests: building systems, validation, physics sanity, plotting."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyorbiter as orb


@pytest.fixture
def earth_sun():
    return (orb.create_system()
            .add_sun()
            .add_body("Earth", mass=orb.mass_earth, x=orb.distance_earth_sun, vy=orb.speed_earth))


@pytest.fixture
def sim(earth_sun):
    return earth_sun.simulate(time_step=orb.seconds_per_day, duration=30 * orb.seconds_per_day)


# ---- Building -------------------------------------------------------------

def test_create_system_is_empty():
    system = orb.create_system()
    assert len(system) == 0
    assert system.G == orb.gravitational_constant
    assert list(system.bodies.columns) == ["id", "mass", "x", "y", "z", "vx", "vy", "vz"]
    assert "Bodies: (none)" in repr(system)


def test_add_body_returns_a_new_system():
    empty = orb.create_system()
    one = empty.add_body("A", mass=1.0, x=2.0)
    assert len(empty) == 0 and len(one) == 1
    assert one.bodies.loc[0, "x"] == 2.0


def test_function_and_method_forms_agree():
    a = orb.add_body(orb.create_system(), "A", 5.0, vx=1.0)
    b = orb.create_system().add_body("A", 5.0, vx=1.0)
    pd.testing.assert_frame_equal(orb.get_bodies(a), b.bodies)


def test_add_body_validation():
    system = orb.create_system().add_body("A", mass=1.0)
    with pytest.raises(ValueError, match="already exists"):
        system.add_body("A", mass=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        system.add_body("B", mass=-1.0)
    with pytest.raises(ValueError, match="single numeric"):
        system.add_body("B", mass="heavy")
    with pytest.raises(TypeError):
        orb.add_body("not a system", "B", 1.0)


def test_keplerian_validation():
    system = orb.create_system().add_sun()
    with pytest.raises(ValueError, match="not found"):
        system.add_body_keplerian("X", 1.0, a=1e11, parent="Nope")
    with pytest.raises(ValueError, match="Parabolic"):
        system.add_body_keplerian("X", 1.0, a=1e11, e=1, parent="Sun")
    with pytest.raises(ValueError, match="positive for an elliptical"):
        system.add_body_keplerian("X", 1.0, a=-1e11, e=0.5, parent="Sun")
    with pytest.raises(ValueError, match="negative for a hyperbolic"):
        system.add_body_keplerian("X", 1.0, a=1e11, e=1.5, parent="Sun")
    with pytest.raises(ValueError, match="asymptotes"):
        system.add_body_keplerian("X", 1.0, a=-1e11, e=1.5, nu=170, parent="Sun")


def test_circular_keplerian_orbit_has_circular_speed():
    system = orb.create_system().add_sun().add_body_keplerian(
        "P", mass=1.0, a=orb.distance_earth_sun, parent="Sun")
    p = system.bodies.iloc[1]
    assert p["x"] == pytest.approx(orb.distance_earth_sun)
    assert p["vy"] == pytest.approx(np.sqrt(orb.gravitational_constant * orb.mass_sun / orb.distance_earth_sun))


def test_keplerian_round_trip():
    elements = dict(a=2.5e11, e=0.3, i=20.0, lan=40.0, arg_pe=60.0, nu=80.0)
    sim = (orb.create_system().add_sun()
           .add_body_keplerian("P", mass=0.0, parent="Sun", **elements)
           .simulate(time_step=orb.seconds_per_hour, duration=0))
    got = orb.get_orbital_elements(sim, "P", "Sun").iloc[0]
    for key, value in elements.items():
        assert got[key] == pytest.approx(value, rel=1e-10)


def test_add_planet_is_case_insensitive_and_overridable():
    system = orb.create_system().add_sun().add_planet("earth", parent="sun", e=0, i=0)
    assert list(system.bodies["id"]) == ["Sun", "Earth"]
    with pytest.raises(ValueError, match="not a recognized body"):
        system.add_planet("Vulcan", parent="Sun")


def test_load_solar_system_options():
    assert len(orb.load_solar_system()) == 11
    ids = list(orb.load_solar_system(moon=False, pluto=False).bodies["id"])
    assert "Moon" not in ids and "Pluto" not in ids and len(ids) == 9


def test_remove_body():
    system = orb.load_solar_system().remove_body(["Moon", "Pluto"])
    assert len(system) == 9
    with pytest.raises(ValueError, match="not found"):
        system.remove_body("Moon")


def test_save_load_export(tmp_path, earth_sun):
    path = tmp_path / "system.json"
    earth_sun.save(path)
    loaded = orb.load_system(path)
    pd.testing.assert_frame_equal(loaded.bodies, earth_sun.bodies)
    assert loaded.G == earth_sun.G

    csv = tmp_path / "bodies.csv"
    orb.export_bodies(earth_sun, csv)
    assert list(pd.read_csv(csv)["id"]) == ["Sun", "Earth"]

    with pytest.raises(FileNotFoundError):
        orb.load_system(tmp_path / "missing.json")


# ---- Simulating -----------------------------------------------------------

def test_simulation_shape_and_metadata(sim):
    assert list(sim.columns) == ["id", "mass", "x", "y", "z", "vx", "vy", "vz", "time"]
    assert len(sim) == 2 * 31
    assert sim["time"].max() == 30 * orb.seconds_per_day
    assert sim.attrs == dict(G=orb.gravitational_constant, softening=0, method="verlet",
                             time_step=orb.seconds_per_day)


def test_duration_stops_at_last_whole_step(earth_sun):
    sim = earth_sun.simulate(time_step=1000, duration=2500)
    assert list(sim["time"].unique()) == [0, 1000, 2000]


def test_simulate_validation(earth_sun):
    with pytest.raises(ValueError, match="Method must be"):
        earth_sun.simulate(method="rk4")
    with pytest.raises(ValueError, match="time_step"):
        earth_sun.simulate(time_step=0)
    with pytest.raises(TypeError):
        orb.simulate_system(earth_sun.bodies)


def test_no_gravity_is_straight_line_motion():
    sim = (orb.create_system(G=0)
           .add_body("A", mass=1.0, vx=2.0)
           .add_body("B", mass=1.0, x=1.0)
           .simulate(time_step=1, duration=10))
    a = sim.loc[sim["id"] == "A"]
    np.testing.assert_allclose(a["x"], 2.0 * a["time"])


def test_acceleration_matches_newton():
    pos = np.array([[0.0, 0, 0], [orb.distance_earth_sun, 0, 0]])
    mass = np.array([orb.mass_sun, orb.mass_earth])
    acc = orb.calc_acceleration(pos, mass, orb.gravitational_constant)
    expected = orb.gravitational_constant * orb.mass_sun / orb.distance_earth_sun**2
    assert acc[1, 0] == pytest.approx(-expected)
    # Newton's third law: total force is zero
    np.testing.assert_allclose((mass[:, None] * acc).sum(axis=0), 0, atol=1e-6 * mass[1] * expected)


def test_earth_returns_after_one_year(earth_sun):
    sim = earth_sun.simulate(time_step=orb.seconds_per_hour, duration=orb.seconds_per_year)
    earth = sim.loc[sim["id"] == "Earth"].iloc[-1]
    assert np.hypot(earth["x"] - orb.distance_earth_sun, earth["y"]) < 0.02 * orb.distance_earth_sun


def test_verlet_conserves_energy_better_than_euler(earth_sun):
    def worst(method):
        run = earth_sun.simulate(time_step=orb.seconds_per_day, duration=orb.seconds_per_year, method=method)
        return orb.conserved_quantities(run)["energy_error"].abs().max()

    assert worst("verlet") < 1e-4
    assert worst("verlet") < worst("euler") / 100


def test_continue_simulation_equals_one_long_run(earth_sun):
    day = orb.seconds_per_day
    whole = earth_sun.simulate(time_step=day, duration=20 * day)
    half = earth_sun.simulate(time_step=day, duration=10 * day)
    joined = orb.continue_simulation(half, time_step=day, duration=10 * day)
    assert len(joined) == len(whole)
    np.testing.assert_allclose(joined[["x", "y", "vx", "vy"]], whole[["x", "y", "vx", "vy"]], rtol=1e-12, atol=1e-6)
    assert joined.attrs["method"] == "verlet"


def test_system_from_simulation(sim):
    system = orb.system_from_simulation(sim, time=0)
    assert list(system.bodies["id"]) == ["Sun", "Earth"]
    assert system.bodies.loc[1, "x"] == orb.distance_earth_sun


# ---- Analysis -------------------------------------------------------------

def test_shift_reference_frame(sim):
    shifted = orb.shift_reference_frame(sim, "Earth")
    earth = shifted.loc[shifted["id"] == "Earth"]
    assert (earth[["x", "y", "z", "vx", "vy", "vz"]] == 0).all().all()
    assert shifted.attrs == sim.attrs
    assert "Earth" not in set(orb.shift_reference_frame(sim, "Earth", keep_center=False)["id"])
    with pytest.raises(ValueError, match="not found"):
        orb.shift_reference_frame(sim, "Pluto")


def test_barycentric_frame_has_zero_momentum(sim):
    bary = orb.shift_reference_frame(sim, "barycenter")
    momentum = orb.get_momentum(bary)
    scale = orb.mass_earth * orb.speed_earth
    assert (momentum[["px", "py", "pz"]].abs() < 1e-9 * scale).all().all()


def test_orbital_elements_of_earth(sim):
    el = orb.get_orbital_elements(sim, "Earth", "Sun")
    assert len(el) == 31
    assert el["a"].iloc[0] == pytest.approx(orb.distance_earth_sun, rel=0.01)
    assert el["e"].iloc[0] < 0.01
    assert el["period"].iloc[0] == pytest.approx(orb.seconds_per_year, rel=0.01)
    with pytest.raises(ValueError, match="different"):
        orb.get_orbital_elements(sim, "Earth", "Earth")


def test_analysis_rejects_non_simulations():
    with pytest.raises(TypeError, match="simulate_system"):
        orb.get_energy(pd.DataFrame({"x": [1]}))


# ---- Plotting -------------------------------------------------------------

def test_2d_plots_return_axes(sim):
    import matplotlib.pyplot as plt

    ax = orb.plot_orbits(sim)
    assert len(ax.lines) == 2 and ax.get_xlabel() == "X (m)"
    ax = orb.plot_system(sim, time=5 * orb.seconds_per_day, trails=True)
    assert "System Snapshot" in ax.get_title()
    plt.close("all")


def test_3d_plots_use_plotly():
    go = pytest.importorskip("plotly.graph_objects")
    sim = orb.load_solar_system().simulate(time_step=orb.seconds_per_day, duration=20 * orb.seconds_per_day)
    assert isinstance(orb.plot_orbits(sim), go.Figure)
    assert isinstance(orb.plot_system(sim, trails=True), go.Figure)
    anim = orb.animate_system(sim, fps=5, duration=2, trails=True)
    assert len(anim.frames) == 10


def test_2d_animation_saves_gif(tmp_path, sim):
    import matplotlib.pyplot as plt

    pytest.importorskip("PIL")
    path = tmp_path / "orbit.gif"
    orb.animate_system(sim, fps=5, duration=1, trails=True, path=path)
    assert path.stat().st_size > 0
    plt.close("all")


def test_keplerian_needs_a_gravitating_parent():
    massless = orb.create_system().add_body("Marker", mass=0)
    with pytest.raises(ValueError, match="gravitating parent"):
        massless.add_body_keplerian("Probe", mass=1.0, a=1e9, parent="Marker")
    no_gravity = orb.create_system(G=0).add_sun()
    with pytest.raises(ValueError, match="gravitating parent"):
        no_gravity.add_planet("Earth", parent="Sun")


def test_saving_a_3d_animation_warns_that_path_is_ignored(tmp_path):
    pytest.importorskip("plotly")
    tilted = (orb.create_system().add_sun()
              .add_body_keplerian("Comet", mass=1e14, a=3e11, e=0.5, i=40, parent="Sun")
              .simulate(time_step=orb.seconds_per_day, duration=20 * orb.seconds_per_day))
    with pytest.warns(UserWarning, match="three_d=False"):
        orb.animate_system(tilted, fps=2, duration=2, path=tmp_path / "comet.gif")
    assert not (tmp_path / "comet.gif").exists()
