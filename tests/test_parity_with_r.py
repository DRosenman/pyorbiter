"""pyorbitr must reproduce the orbitr R package's numbers.

The reference values in reference/r_reference.json come from orbitr 1.0.0
(regenerate with reference/make_reference.R).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyorbitr as orb

REF = json.loads((Path(__file__).parent / "reference" / "r_reference.json").read_text())
STATE = ["x", "y", "z", "vx", "vy", "vz"]


def last_step(sim):
    return sim.loc[sim["time"] == sim["time"].max()].reset_index(drop=True)


def assert_rows_match(actual: pd.DataFrame, expected_rows, rtol=1e-8):
    expected = pd.DataFrame(expected_rows)
    assert list(actual["id"]) == list(expected["id"])
    for col in [c for c in expected.columns if c != "id"]:
        want = expected[col].to_numpy(dtype=float)
        got = actual[col].to_numpy(dtype=float)
        # Small entries (e.g. the Sun's wobble) are compared against the
        # column's overall scale, not their own size.
        scale = np.abs(want).max()
        np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * scale, err_msg=col)


def assert_row_matches(actual: pd.Series, expected: dict, rtol=1e-8, atol=0.0):
    for key, want in expected.items():
        if want is None:
            assert np.isnan(actual[key]), key
        else:
            np.testing.assert_allclose(actual[key], want, rtol=rtol, atol=atol, err_msg=key)


@pytest.fixture(scope="module")
def earth_sun():
    return (orb.create_system()
            .add_body("Sun", mass=orb.mass_sun)
            .add_body("Earth", mass=orb.mass_earth, x=orb.distance_earth_sun, vy=orb.speed_earth))


@pytest.fixture(scope="module")
def earth_sun_sim(earth_sun):
    return earth_sun.simulate(time_step=orb.seconds_per_day, duration=365 * orb.seconds_per_day)


@pytest.fixture(scope="module")
def solar():
    return orb.load_solar_system().simulate(time_step=orb.seconds_per_hour,
                                            duration=30 * orb.seconds_per_day)


@pytest.mark.parametrize("method", ["verlet", "euler_cromer", "euler"])
def test_integrators(earth_sun, method):
    sim = earth_sun.simulate(time_step=orb.seconds_per_day,
                             duration=365 * orb.seconds_per_day, method=method)
    assert_rows_match(last_step(sim), REF[f"earth_sun_{method}"])


def test_row_count(earth_sun_sim):
    assert len(earth_sun_sim) == REF["earth_sun_n_rows"]


def test_conserved_quantities(earth_sun_sim):
    got = orb.conserved_quantities(earth_sun_sim).iloc[-1]
    want = dict(REF["earth_sun_conserved"])
    errors = {k: want.pop(k) for k in list(want) if k.endswith("_error")}
    # The Sun starts at rest, so px begins at 0 and never grows past rounding noise
    px = want.pop("px")
    assert abs(got["px"] - px) < 1e-9 * abs(want["py"])
    assert_row_matches(got, want)
    # Errors are differences of nearly equal numbers, so only their size is stable
    for key, value in errors.items():
        assert got[key] == pytest.approx(value, rel=1e-3, abs=1e-14), key


def test_continue_simulation(earth_sun_sim):
    more = orb.continue_simulation(earth_sun_sim, time_step=orb.seconds_per_hour,
                                   duration=10 * orb.seconds_per_day)
    assert_rows_match(last_step(more), REF["earth_sun_continued"])


def test_solar_system_initial_state():
    assert_rows_match(orb.load_solar_system().bodies, REF["solar_system_bodies"], rtol=1e-12)


def test_solar_system_run(solar):
    assert_rows_match(last_step(solar), REF["solar_last"])


@pytest.mark.parametrize("body,parent", [("Moon", "Earth"), ("Mars", "Sun")])
def test_orbital_elements(solar, body, parent):
    got = orb.get_orbital_elements(solar, body, parent).iloc[-1]
    assert_row_matches(got, REF[f"solar_elements_{body.lower()}"], rtol=1e-7)


def test_solar_conserved(solar):
    got = orb.conserved_quantities(solar).iloc[-1]
    want = {k: v for k, v in REF["solar_conserved"].items() if not k.endswith("_error")}
    assert_row_matches(got, want, rtol=1e-8)


def test_barycentric_frame(solar):
    assert_rows_match(last_step(orb.shift_reference_frame(solar, "barycenter")), REF["solar_barycentric"])


def test_body_frame(solar):
    shifted = orb.shift_reference_frame(solar, "Earth", keep_center=False)
    assert_rows_match(last_step(shifted), REF["solar_earth_frame"])


def test_hyperbolic_orbit():
    system = orb.create_system().add_sun().add_body_keplerian(
        "Oumuamua", mass=1e10, parent="Sun",
        a=-1.2765 * orb.distance_earth_sun, e=1.2, arg_pe=90, nu=-130)
    assert_rows_match(system.bodies, REF["hyperbolic_bodies"], rtol=1e-12)

    sim = system.simulate(time_step=orb.seconds_per_hour, duration=200 * orb.seconds_per_day)
    got = orb.get_orbital_elements(sim, "Oumuamua", "Sun").iloc[-1]
    # i = 0, so lan is fixed at 0 by convention; allow for 0 vs 360
    assert_row_matches(got, REF["hyperbolic_elements"], rtol=1e-7, atol=1e-9)
    assert np.isnan(got["period"])


def test_softening_and_3d():
    sim = (orb.create_system()
           .add_body("A", mass=1e30, vy=5000)
           .add_body("B", mass=1e30, x=1e11, vy=-2500)
           .add_body("C", mass=1e30, x=5e10, y=8.66e10, z=1e9, vy=-2500, vz=100)
           .simulate(time_step=orb.seconds_per_hour, duration=20 * orb.seconds_per_day, softening=1e9))
    assert_rows_match(last_step(sim), REF["softened_last"])
    assert_row_matches(orb.get_energy(sim).iloc[-1], REF["softened_energy"])
