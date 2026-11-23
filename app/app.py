"""pyorbiter simulator: interactive N-body orbital simulations in Python.

Run with:  shiny run app/app.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from shiny import App, reactive, render, req, ui

import pyorbiter as orb

DAY = orb.seconds_per_day
AU = orb.distance_earth_sun
STEP_SECONDS = {"hour": orb.seconds_per_hour, "day": orb.seconds_per_day}
METHOD_LABELS = {"verlet": "Velocity Verlet", "euler_cromer": "Euler-Cromer", "euler": "Euler"}


# =====================================================================
# Scenarios
# =====================================================================
# Bodies are Cartesian (id, mass, x, y, vx, vy) unless they set
# type="keplerian", in which case they take add_body_keplerian() arguments and
# their parent must be listed first. The optional `parents` entry names the
# body each orbit is measured about in the Orbital Elements tab ("barycenter" =
# the center of mass of all the other bodies); otherwise that is the most
# massive body.

def _body(id, mass, x=0.0, y=0.0, vx=0.0, vy=0.0):
    return dict(id=id, mass=mass, x=x, y=y, vx=vx, vy=vy)


def _kepler16_bodies():
    G = orb.gravitational_constant
    m_a, m_b = 0.68 * orb.mass_sun, 0.20 * orb.mass_sun
    a_bin = 0.22 * AU
    r_a = a_bin * m_b / (m_a + m_b)
    r_b = a_bin * m_a / (m_a + m_b)
    v_a = math.sqrt(G * m_b**2 / ((m_a + m_b) * a_bin))
    v_b = math.sqrt(G * m_a**2 / ((m_a + m_b) * a_bin))
    r_planet = 0.7048 * AU
    v_planet = math.sqrt(G * (m_a + m_b) / r_planet)
    return [
        _body("Star A", m_a, x=r_a, vy=v_a),
        _body("Star B", m_b, x=-r_b, vy=-v_b),
        _body("Kepler-16b", 0.333 * orb.mass_jupiter, x=r_planet, vy=v_planet),
    ]


_SUN = _body("Sun", orb.mass_sun)
_EARTH = _body("Earth", orb.mass_earth, x=orb.distance_earth_sun, vy=orb.speed_earth)
_MARS = _body("Mars", orb.mass_mars, x=orb.distance_mars_sun, vy=orb.speed_mars)
_VENUS = _body("Venus", orb.mass_venus, x=orb.distance_venus_sun, vy=orb.speed_venus)

SCENARIOS = {
    "Earth-Sun": dict(
        description="Earth orbiting the Sun for one year.",
        bodies=[_SUN, _EARTH],
        time_step="day", duration_days=365,
    ),
    "Earth-Moon": dict(
        description="The Moon orbiting Earth for 28 days.",
        bodies=[_body("Earth", orb.mass_earth),
                _body("Moon", orb.mass_moon, x=orb.distance_earth_moon, vy=orb.speed_moon)],
        time_step="hour", duration_days=28,
    ),
    "Sun-Earth-Moon (Earth frame)": dict(
        description="Sun, Earth, and Moon — viewed from Earth's reference frame.",
        bodies=[_SUN, _EARTH,
                _body("Moon", orb.mass_moon, x=orb.distance_earth_sun + orb.distance_earth_moon,
                      vy=orb.speed_earth + orb.speed_moon)],
        time_step="hour", duration_days=365, shift_to="Earth", parents={"Moon": "Earth"},
    ),
    "Sun-Earth-Venus": dict(
        description="Inner solar system: Sun, Earth, and Venus for one year.",
        bodies=[_SUN, _EARTH, _VENUS],
        time_step="day", duration_days=365,
    ),
    "Inner Solar System": dict(
        description="Mercury, Venus, Earth, and Mars orbiting the Sun.",
        bodies=[_SUN,
                _body("Mercury", orb.mass_mercury, x=orb.distance_mercury_sun, vy=orb.speed_mercury),
                _VENUS, _EARTH, _MARS],
        time_step="day", duration_days=687,
    ),
    "Kepler-16 (Circumbinary)": dict(
        description="Kepler-16b orbits two stars — a real-life Tatooine.",
        bodies=_kepler16_bodies(),
        time_step="hour", duration_days=round(228.8 * 3), parents={"Kepler-16b": "barycenter"},
    ),
    "Three-Body Chaos": dict(
        description="Three equal-mass bodies in an unstable configuration.",
        bodies=[_body("Body A", 1e30, vy=5000),
                _body("Body B", 1e30, x=1e11, vy=-2500),
                _body("Body C", 1e30, x=5e10, y=8.66e10, vy=-2500)],
        time_step="hour", duration_days=365 * 3,
    ),
    "Interstellar Flyby (Hyperbolic)": dict(
        description=("An 'Oumuamua-like interstellar visitor on a hyperbolic orbit (e = 1.2) "
                     "swings past the Sun, Earth, and Mars, then leaves for good."),
        bodies=[_SUN, _EARTH, _MARS,
                # e > 1 needs a negative semi-major axis. Perihelion q = a (1 - e) = 0.2553 AU.
                # nu = -130 starts it ~2.5 AU out, inbound, about 95 days before perihelion.
                dict(type="keplerian", id="Oumuamua", mass=1e10, parent="Sun",
                     a=-1.2765 * AU, e=1.2, i=0, lan=0, arg_pe=90, nu=-130)],
        time_step="hour", duration_days=200,
    ),
}
DEFAULT_SCENARIO = "Earth-Sun"
KEPLER_ANGLES = ("i", "lan", "arg_pe", "nu")


def is_keplerian(body: dict) -> bool:
    return body.get("type") == "keplerian"


def body_ids(scenario: dict) -> list[str]:
    return [b["id"] for b in scenario["bodies"]]


def natural_parent(scenario: dict, body: str, primary: str) -> str:
    """Body an orbit is measured about in the Orbital Elements tab."""
    return scenario.get("parents", {}).get(body, primary)


def build_system(scenario: dict) -> orb.OrbitSystem:
    system = orb.create_system()
    for b in scenario["bodies"]:
        args = {k: v for k, v in b.items() if k != "type"}
        system = system.add_body_keplerian(**args) if is_keplerian(b) else system.add_body(**args)
    return system


def duration_problem(days) -> str | None:
    """A friendly message if the duration is unusable, otherwise None."""
    if days is None or not isinstance(days, (int, float)) or not math.isfinite(days) or days <= 0:
        return "Duration must be a positive number of days."
    return None


# ---- Reference frames -------------------------------------------------

def frame_choices(scenario: dict) -> dict[str, str]:
    choices = {"default": "Scenario default", "inertial": "Inertial (as simulated)",
               "barycenter": "Barycenter"}
    choices.update({f"body:{b}": f"Centered on {b}" for b in body_ids(scenario)})
    return choices


def frame_center(frame: str | None, scenario: dict) -> str | None:
    """Resolve a frame choice to a shift_reference_frame() center (None = no shift)."""
    default = scenario.get("shift_to")
    if frame in (None, "default"):
        return default
    if frame == "inertial":
        return None
    if frame == "barycenter":
        return "barycenter"
    center = frame.removeprefix("body:")
    return center if center in body_ids(scenario) else default


def frame_subtitle(frame: str | None, center: str | None) -> str | None:
    if frame in (None, "default"):
        return None
    if center is None:
        return "Inertial frame (as simulated)"
    if center == "barycenter":
        return "Barycentric frame (center of mass at the origin)"
    return f"{center}-centered frame"


# ---- Diagnostics ------------------------------------------------------

def snapshot_rows(sim: pd.DataFrame, snap_day: float) -> pd.DataFrame:
    """Rows of the time step closest to a given day."""
    available = np.sort(sim["time"].unique())
    target = available[np.argmin(np.abs(available - snap_day * DAY))]
    return sim.loc[sim["time"] == target]


def compute_conservation(raw: pd.DataFrame, max_steps: int = 3000) -> dict:
    """conserved_quantities() on the raw run, sampling long runs evenly."""
    times = np.sort(raw["time"].unique())
    sampled = raw
    if len(times) > max_steps:
        keep = times[np.unique(np.round(np.linspace(0, len(times) - 1, max_steps)).astype(int))]
        sampled = raw.loc[raw["time"].isin(keep)]
        sampled.attrs = dict(raw.attrs)
    cq = orb.conserved_quantities(sampled)
    return dict(cq=cq, n_total=len(times), n_used=len(cq))


CONSERVATION_FLOOR = 1e-16
ELEMENT_COLUMNS = ["Body", "Relative to", "Orbit", "Semi-major axis (AU)", "Eccentricity",
                   "Inclination (°)", "True anomaly (°)", "Period (days)"]
DASH = "—"


def fmt_fixed(x: float, digits: int) -> str:
    return f"{x:.{digits}f}" if math.isfinite(x) else DASH


def fmt_sig(x: float, digits: int = 5) -> str:
    return f"{x:#.{digits}g}" if math.isfinite(x) else DASH


def snapshot_elements(snap: pd.DataFrame, scenario: dict, G: float) -> pd.DataFrame | None:
    """get_orbital_elements() for every body except the primary, at one snapshot."""
    d = snap.reset_index(drop=True)
    primary = d["id"][d["mass"].idxmax()]
    rows = []
    for b in [s for s in d["id"] if s != primary]:
        parent = natural_parent(scenario, b, primary)
        data_b = d
        if parent == "barycenter":
            # Pseudo-body at the center of mass of everything else (e.g. a binary star)
            others = d.loc[d["id"] != b]
            total = others["mass"].sum()
            cm = {col: (others["mass"] * others[col]).sum() / total
                  for col in ("x", "y", "z", "vx", "vy", "vz")}
            cm.update(id=".barycenter", mass=total, time=d["time"][0])
            data_b = pd.concat([d, pd.DataFrame([cm])], ignore_index=True)
            parent_id = ".barycenter"
            parent_label = "Barycenter of " + " + ".join(others["id"])
        else:
            parent_id = parent if parent in set(d["id"]) else primary
            parent_label = parent_id

        # Exact two-body mu = G (M_parent + m_body). The default, G M_parent,
        # misreports orbits whose companion is heavy (e.g. Kepler-16's Star B).
        m_parent = data_b.loc[data_b["id"] == parent_id, "mass"].iloc[0]
        m_body = d.loc[d["id"] == b, "mass"].iloc[0]
        el = orb.get_orbital_elements(data_b, b, parent_id, mu=G * (m_parent + m_body)).iloc[0]

        bound = bool(el["e"] < 1 and el["a"] > 0)
        nu = el["nu"]
        if math.isfinite(nu) and nu > 180:
            nu -= 360  # signed: negative = before periapsis
        orbit = DASH if not math.isfinite(el["e"]) else "Bound (ellipse)" if bound else "Unbound (hyperbola)"
        rows.append([b, parent_label, orbit, fmt_sig(el["a"] / AU), fmt_fixed(el["e"], 5),
                     fmt_fixed(el["i"], 2), fmt_fixed(nu, 1),
                     fmt_fixed(el["period"] / DAY, 2) if bound else DASH])
    return pd.DataFrame(rows, columns=ELEMENT_COLUMNS) if rows else None


# =====================================================================
# Code tab: the Python that reproduces a run
# =====================================================================
CONSTANT_NAMES = {getattr(orb, name): f"orb.{name}" for name in dir(orb)
                  if name.startswith(("mass_", "distance_", "speed_"))}


def fmt_value(value: float) -> str:
    return CONSTANT_NAMES.get(value) or f"{value:.10g}"


def fmt_sum(value: float, parts: list[float]) -> str:
    """Express a value as a sum of two known constants where possible."""
    for part in parts:
        if part in CONSTANT_NAMES and (value - part) in CONSTANT_NAMES:
            return f"{CONSTANT_NAMES[part]} + {CONSTANT_NAMES[value - part]}"
    return fmt_value(value)


def fmt_au(value: float) -> str:
    """A semi-major axis as a multiple of distance_earth_sun when that is exact."""
    k = value / AU
    if abs(k - round(k, 6)) < 1e-9 * max(1, abs(k)):
        return f"{round(k, 6):.10g} * orb.distance_earth_sun"
    return f"{value:.15g}"


def reproduce_code(params: dict) -> str:
    s = params["scenario"]
    step = f"orb.seconds_per_{params['time_step_choice']}"
    cartesian = [b for b in s["bodies"] if not is_keplerian(b)]
    all_x = [b["x"] for b in cartesian]
    all_vy = [b["vy"] for b in cartesian]

    lines = ["import pyorbiter as orb", "", "sim = (", "    orb.create_system()"]
    for b in s["bodies"]:
        if is_keplerian(b):
            # e > 1 with a < 0 is a hyperbolic orbit
            args = [f"a={fmt_au(b['a'])}", f"e={b.get('e', 0):.15g}"]
            args += [f"{k}={b[k]:.15g}" for k in KEPLER_ANGLES if b.get(k, 0) != 0]
            lines.append(f'    .add_body_keplerian("{b["id"]}", mass={fmt_value(b["mass"])}, parent="{b["parent"]}",')
            lines.append(f"                        {', '.join(args)})")
            continue
        args = [f"mass={fmt_value(b['mass'])}"]
        if b["x"] != 0:
            args.append(f"x={fmt_sum(b['x'], all_x)}")
        if b["y"] != 0:
            args.append(f"y={fmt_value(b['y'])}")
        if b["vx"] != 0:
            args.append(f"vx={fmt_value(b['vx'])}")
        if b["vy"] != 0:
            args.append(f"vy={fmt_sum(b['vy'], all_vy)}")
        lines.append(f'    .add_body("{b["id"]}", {", ".join(args)})')
    lines.append(f"    .simulate(time_step={step}, duration={params['duration_days']} * orb.seconds_per_day, "
                 f'method="{params["method"]}")')
    lines.append(")")

    # One continue_simulation() per Continue click; it reuses the run's integrator
    for days in params["segments"]:
        lines.append(f"sim = orb.continue_simulation(sim, time_step={step}, "
                     f"duration={days} * orb.seconds_per_day)")
    lines.append("")

    # Plot in the chosen reference frame
    center = frame_center(params["frame"], s)
    if center is None:
        lines.append("orb.plot_orbits(sim)")
    else:
        lines.append(f'orb.plot_orbits(orb.shift_reference_frame(sim, "{center}"))')

    # Diagnostics; prefer a Keplerian body for the elements example
    ids = body_ids(s)
    masses = {b["id"]: b["mass"] for b in s["bodies"]}
    primary = max(ids, key=lambda k: masses[k])  # first of equal masses, like which.max()
    candidates = [b for b in s["bodies"]
                  if b["id"] != primary and natural_parent(s, b["id"], primary) != "barycenter"]
    lines += ["", "# Orbital elements and conservation diagnostics"]
    if candidates:
        pick = next((b for b in candidates if is_keplerian(b)), candidates[0])
        parent = natural_parent(s, pick["id"], primary)
        m_body, m_parent = masses[pick["id"]], masses[parent]
        # Exact two-body mu (as in the Elements tab) when the body's mass matters
        mu = (f", mu=orb.gravitational_constant * ({fmt_value(m_parent)} + {fmt_value(m_body)})"
              if m_body > 0.01 * m_parent else "")
        lines.append(f'orb.get_orbital_elements(sim, "{pick["id"]}", "{parent}"{mu})')
    lines.append("orb.conserved_quantities(sim)")
    return "\n".join(lines)


# =====================================================================
# Plot styling
# =====================================================================
BG, PANEL, BORDER, GRID = "#0d1117", "#161b22", "#30363d", "#21262d"
TEXT, MUTED, BRIGHT = "#c9d1d9", "#8b949e", "#f0f6fc"
# Categorical series colors for a dark surface, assigned in this fixed order
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]


def body_colors(scenario: dict) -> dict[str, str]:
    """Color follows the body, whatever frame or tab it is drawn in."""
    return {b: SERIES[k % len(SERIES)] for k, b in enumerate(body_ids(scenario))}


def new_figure(title: str, subtitle: str | None, xlabel: str, ylabel: str):
    # Shiny lays figures out with tight_layout, which only makes room for
    # artists that belong to the axes, so the title and legend are attached there.
    fig = Figure(facecolor=BG)
    ax = fig.add_subplot(facecolor=BG)
    ax.set_title(title, color=BRIGHT, fontsize=16, fontweight="bold", loc="left",
                 pad=44 if subtitle else 22)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 22),
                    textcoords="offset points", color=MUTED, fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=12)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=12)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)
    ax.xaxis.get_offset_text().set_color(MUTED)
    ax.yaxis.get_offset_text().set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def add_legend(ax, title: str) -> None:
    legend = ax.legend(title=title, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                       fontsize=12, title_fontsize=12, labelcolor=TEXT)
    legend.get_title().set_color(BRIGHT)


def html_table(table: pd.DataFrame):
    return ui.HTML(table.to_html(index=False, classes="shiny-table", border=0))


def message_figure(text: str) -> Figure:
    fig = Figure(facecolor=BG)
    fig.text(0.5, 0.5, text, color=MUTED, fontsize=13, ha="center", va="center")
    return fig


# =====================================================================
# UI
# =====================================================================
CSS = """
body { background-color: #0d1117; color: #c9d1d9;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
.header-bar { padding: 20px 0 10px 0; margin-bottom: 20px; border-bottom: 1px solid #21262d; }
.header-bar h2 { margin: 0; color: #f0f6fc; }
.header-bar .tagline { color: #8b949e; font-size: 14px; margin-top: 4px; }
.bslib-sidebar-layout { border: none; --bslib-sidebar-bg: #161b22; --bslib-sidebar-fg: #c9d1d9; }
.bslib-sidebar-layout > .sidebar { border: 1px solid #30363d; border-radius: 6px; }
.bslib-sidebar-layout > .main { background-color: #0d1117; padding-top: 0; }
h4 { color: #f0f6fc; font-size: 1.1rem; }
hr { border-color: #30363d; opacity: 1; }
label, .control-label { color: #c9d1d9; }
.form-control, .form-select, .selectize-input, .selectize-input.full, .selectize-input input {
  background-color: #0d1117 !important; border-color: #30363d !important; color: #c9d1d9 !important; }
.selectize-input { border: 1px solid #30363d !important; box-shadow: none !important; }
.selectize-dropdown { background-color: #161b22 !important; border: 1px solid #30363d !important;
                      color: #c9d1d9 !important; }
.selectize-dropdown-content .option { color: #c9d1d9 !important; }
.selectize-dropdown-content .active { background-color: #1f6feb !important; color: #fff !important; }
.btn-primary { background-color: #238636; border-color: #238636; color: #fff; font-weight: 600; }
.btn-primary:hover, .btn-primary:focus, .btn-primary:active {
  background-color: #2ea043 !important; border-color: #2ea043 !important; }
.btn.btn-continue { background-color: #21262d; border: 1px solid #30363d; color: #c9d1d9; font-weight: 600; }
.btn.btn-continue:hover, .btn.btn-continue:focus, .btn.btn-continue:active {
  background-color: #30363d; border-color: #8b949e; color: #f0f6fc; }
.scenario-desc { color: #8b949e; font-style: italic; font-size: 14px; }
.sidebar-hint { color: #8b949e; font-size: 12px; text-align: center; }
.powered { color: #6e7681; font-size: 12px; text-align: center; }
.powered a { color: #58a6ff; }
.nav-tabs { border-bottom: 1px solid #21262d; }
.nav-tabs .nav-link { color: #8b949e; border: none; border-bottom: 2px solid transparent; }
.nav-tabs .nav-link:hover { color: #c9d1d9; border-color: transparent; }
.nav-tabs .nav-link.active { color: #f0f6fc; background-color: #0d1117;
                             border-bottom: 2px solid #1f6feb; }
.tab-content { padding-top: 15px; }
.tab-summary { color: #c9d1d9; font-size: 14px; padding: 0 0 10px 0; }
.tab-caption { color: #8b949e; font-size: 13px; padding: 10px 0; }
.tab-caption strong { color: #c9d1d9; }
.tab-summary code, .tab-caption code { background-color: #161b22; color: #79c0ff;
                                       border-radius: 3px; padding: 1px 4px; }
.sim-error { color: #f85149; padding: 0 0 10px 0; }
.table-scroll { max-height: 600px; overflow-y: auto; }
table.shiny-table { color: #c9d1d9; width: 100%; font-variant-numeric: tabular-nums; }
table.shiny-table th { color: #f0f6fc; border-bottom: 1px solid #30363d; text-align: right; padding: 6px 10px; }
table.shiny-table td { border-top: 1px solid #21262d; text-align: right; padding: 6px 10px; }
table.shiny-table th:first-child, table.shiny-table td:first-child { text-align: left; }
table.shiny-table tbody tr:hover { background-color: #161b22; }
pre, pre.shiny-text-output { background-color: #161b22 !important; color: #c9d1d9 !important;
                             border: 1px solid #30363d !important; border-radius: 6px; padding: 14px; }
.irs--shiny .irs-line { background: #21262d; border-color: #30363d; }
.irs--shiny .irs-grid-text, .irs--shiny .irs-min, .irs--shiny .irs-max { color: #8b949e; background: none; }
.irs--shiny .irs-bar, .irs--shiny .irs-single { background: #1f6feb; border-color: #1f6feb; }
.slider-animate-button { color: #8b949e; opacity: 1; }
.shiny-notification { background-color: #161b22; color: #c9d1d9; border: 1px solid #30363d; }
.shiny-notification-message { border-left: 4px solid #3fb950; }
.shiny-notification-warning { border-left: 4px solid #d29922; }
.shiny-notification-error { border-left: 4px solid #f85149; color: #f85149; }
"""

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.title("pyorbiter"), ui.tags.style(CSS)),
    ui.div(
        ui.h2("pyorbiter"),
        ui.div("Interactive N-body orbital simulations in Python", class_="tagline"),
        class_="header-bar",
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("scenario", "Scenario", choices=list(SCENARIOS), selected=DEFAULT_SCENARIO),
            ui.output_ui("scenario_description"),
            ui.hr(),
            ui.h4("Parameters"),
            ui.input_numeric("duration_days", "Duration (days)", value=365, min=1, step=1),
            ui.input_select("time_step_choice", "Time Step",
                            choices={"hour": "1 hour", "day": "1 day"}, selected="day"),
            ui.input_select("method", "Integrator", choices=METHOD_LABELS, selected="verlet"),
            ui.input_select("frame", "Reference Frame",
                            choices=frame_choices(SCENARIOS[DEFAULT_SCENARIO]), selected="default"),
            ui.hr(),
            ui.input_action_button("run", "▶  Run Simulation", class_="btn-primary", width="100%"),
            ui.input_action_button("continue_", "⏩  Continue Simulation", class_="btn-continue",
                                   width="100%"),
            ui.div("Extends the last run (same scenario) by Duration (days), with the same time step "
                   "and integrator.", class_="sidebar-hint"),
            ui.div(ui.HTML("Powered by <a href='https://orbit-r.com' target='_blank'>pyorbiter</a> "
                           f"{orb.__version__}"), class_="powered"),
            width=320, open="always",
        ),
        ui.output_ui("sim_error_box"),
        ui.navset_tab(
            ui.nav_panel("Orbit Plot", ui.output_plot("orbit_plot", height="600px")),
            ui.nav_panel(
                "Snapshot",
                ui.input_slider("snap_day", "Day", min=0, max=365, value=0, step=1, width="100%",
                                animate=True),
                ui.output_plot("snapshot_plot", height="550px"),
            ),
            ui.nav_panel(
                "Data",
                ui.div(ui.output_ui("sim_table"), class_="table-scroll"),
                ui.div(ui.HTML(
                    "<strong>Units:</strong> <b>day</b> — simulation day, set with the Snapshot slider "
                    "&nbsp;·&nbsp; <b>mass</b> — kg &nbsp;·&nbsp; <b>x, y</b> — position "
                    "in meters &nbsp;·&nbsp; <b>vx, vy</b> — velocity in m/s"), class_="tab-caption"),
            ),
            ui.nav_panel(
                "Conservation",
                ui.output_ui("conservation_summary"),
                ui.output_plot("conservation_plot", height="500px"),
                ui.div(ui.HTML(
                    "How well the integrator conserved energy, angular momentum, and linear momentum, "
                    "from <code>conserved_quantities()</code>. These are always computed on the raw, "
                    "inertial simulation, whatever frame is selected, because energy and momentum are not "
                    "conserved in a frame that rides along with an accelerating body. "
                    "<b>Velocity Verlet</b> keeps the energy error bounded (it oscillates but does not "
                    "grow), while <b>Euler</b>'s drifts steadily: switch the Integrator and click Run to "
                    "compare. Values on the floor of the axis (1e-16) are at floating-point rounding "
                    "level. Dashed lines mark where a Continue segment began."), class_="tab-caption"),
            ),
            ui.nav_panel(
                "Orbital Elements",
                ui.output_ui("elements_caption"),
                ui.div(ui.output_ui("elements_table"), class_="table-scroll"),
                ui.div(ui.HTML(
                    "<strong>Units:</strong> <b>semi-major axis</b> — AU (negative for an unbound "
                    "orbit) &nbsp;·&nbsp; <b>eccentricity</b> — 0 circle, below 1 ellipse, above 1 "
                    "hyperbola &nbsp;·&nbsp; <b>inclination, true anomaly</b> — degrees (negative "
                    "anomaly = before periapsis) &nbsp;·&nbsp; <b>period</b> — days (none for an "
                    "unbound orbit). Elements use μ = G·(M<sub>parent</sub> + m<sub>body</sub>), "
                    "the exact two-body value (passed as <code>mu</code> to "
                    "<code>get_orbital_elements()</code>)."), class_="tab-caption"),
            ),
            ui.nav_panel(
                "Code",
                ui.h4("Reproduce this simulation in Python", style="padding-top: 10px;"),
                ui.output_code("code_output"),
            ),
            id="tabs",
        ),
    ),
)


# =====================================================================
# Server
# =====================================================================
def server(input, output, session):
    # Inputs captured at Run time, so the tabs only change on a Run/Continue click
    run_params = reactive.value(dict(
        scenario_name=DEFAULT_SCENARIO, scenario=SCENARIOS[DEFAULT_SCENARIO], duration_days=365,
        time_step_choice="day", method="verlet", frame="default",
        segments=[],         # days requested by each Continue click (Code tab)
        segment_starts=[],   # simulated day on which each Continue segment began
    ))
    raw_sim = reactive.value(None)    # raw (inertial) simulation, incl. continued segments
    sim_error = reactive.value(None)  # friendly message if the last Run failed

    @reactive.effect
    @reactive.event(input.scenario)
    def _scenario_defaults():
        s = SCENARIOS[input.scenario()]
        ui.update_numeric("duration_days", value=s["duration_days"])
        ui.update_select("time_step_choice", selected=s["time_step"])
        ui.update_select("frame", choices=frame_choices(s), selected="default")

    @render.ui
    def scenario_description():
        return ui.div(SCENARIOS[input.scenario()]["description"], class_="scenario-desc")

    @reactive.effect
    @reactive.event(input.run, ignore_none=False)
    def _run():
        scenario = SCENARIOS[input.scenario()]
        days = input.duration_days()
        run_params.set(dict(
            scenario_name=input.scenario(), scenario=scenario, duration_days=days,
            time_step_choice=input.time_step_choice(), method=input.method(),
            frame=input.frame() or "default", segments=[], segment_starts=[],
        ))

        problem = duration_problem(days)
        if problem:
            raw_sim.set(None)
            sim_error.set(problem)
            return
        try:
            with ui.Progress() as progress:
                progress.set(0.5, message="Simulating...")
                result = build_system(scenario).simulate(
                    time_step=STEP_SECONDS[input.time_step_choice()],
                    duration=days * DAY, method=input.method())
        except Exception as err:  # show the reason instead of a broken page
            raw_sim.set(None)
            sim_error.set(f"The simulation could not run: {err}")
            return
        sim_error.set(None)
        raw_sim.set(result)
        # The run stops at the last whole time step within the duration
        sim_days = float(result["time"].max() / DAY)
        ui.update_slider("snap_day", max=sim_days, value=sim_days)

    @reactive.effect
    @reactive.event(input.continue_)
    def _continue():
        raw = raw_sim()
        params = run_params()
        days = input.duration_days()
        step = STEP_SECONDS[params["time_step_choice"]]
        problem = duration_problem(days)
        if problem is None and days * DAY < step:
            problem = "Duration is shorter than one time step, so there is nothing to add."

        if raw is None:
            ui.notification_show("Run a simulation first, then continue it.", type="warning")
        elif input.scenario() != params["scenario_name"]:
            ui.notification_show(
                f"Continue extends the last run ({params['scenario_name']}). Switch back to it, or "
                f"click Run to simulate {input.scenario()} first.", type="warning")
        elif problem:
            ui.notification_show(problem, type="warning")
        else:
            try:
                with ui.Progress() as progress:
                    progress.set(0.5, message="Continuing simulation...")
                    extended = orb.continue_simulation(raw, time_step=step, duration=days * DAY,
                                                       method=params["method"])
            except Exception as err:
                ui.notification_show(f"Could not continue the simulation: {err}", type="error")
                return
            # Junctions and totals come from the data: the continuation restarts at
            # max(time), which can fall short of the requested duration
            run_params.set({**params, "segments": params["segments"] + [days],
                            "segment_starts": params["segment_starts"] + [float(raw["time"].max() / DAY)]})
            raw_sim.set(extended)
            total = float(extended["time"].max() / DAY)
            ui.update_slider("snap_day", max=total, value=total)
            ui.notification_show(f"Extended {params['scenario_name']} by {days} days "
                                 f"({total:g} days in total).", type="message")

    @render.ui
    def sim_error_box():
        return ui.div(sim_error(), class_="sim-error") if sim_error() else None

    def require_sim() -> pd.DataFrame:
        raw = raw_sim()
        req(raw is not None)
        return raw

    @reactive.calc
    def center():
        params = run_params()
        return frame_center(params["frame"], params["scenario"])

    @reactive.calc
    def sim_data():
        """The simulation in the chosen reference frame (plots and Data tab)."""
        raw = require_sim()
        return raw if center() is None else orb.shift_reference_frame(raw, center())

    def subtitle():
        return frame_subtitle(run_params()["frame"], center())

    @render.plot
    def orbit_plot():
        sim, params = sim_data(), run_params()
        colors = body_colors(params["scenario"])
        fig, ax = new_figure(params["scenario_name"], subtitle(), "X (m)", "Y (m)")
        for body, g in sim.groupby("id", sort=False):
            ax.plot(g["x"], g["y"], color=colors[body], linewidth=2, alpha=0.95, label=body)
            # Mark where each body ends up; this also shows a body that barely
            # moves (the Sun, or the frame's center), whose path is too small to see
            ax.plot(g["x"].iloc[-1:], g["y"].iloc[-1:], "o", color=colors[body], markersize=8,
                    markeredgecolor=BG, markeredgewidth=2)
        ax.set_aspect("equal", adjustable="datalim")
        add_legend(ax, "Body")
        return fig

    @render.plot
    def snapshot_plot():
        sim, params = sim_data(), run_params()
        colors = body_colors(params["scenario"])
        snap = snapshot_rows(sim, input.snap_day())

        # Fixed, square limits from the full simulation so the scale never jumps
        x_lo, x_hi, y_lo, y_hi = sim["x"].min(), sim["x"].max(), sim["y"].min(), sim["y"].max()
        half = (max(x_hi - x_lo, y_hi - y_lo) or 1.0) / 2 * 1.05
        x_mid, y_mid = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2

        fig, ax = new_figure(f"{params['scenario_name']}  —  Day {input.snap_day():g}",
                             subtitle(), "X (m)", "Y (m)")
        for body, g in sim.groupby("id", sort=False):
            ax.plot(g["x"], g["y"], color=colors[body], linewidth=1, alpha=0.25)
        for row in snap.itertuples(index=False):
            ax.plot([row.x], [row.y], "o", color=colors[row.id], markersize=11,
                    markeredgecolor=BG, markeredgewidth=2, label=row.id)
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_aspect("equal", adjustable="box")
        add_legend(ax, "Body")
        return fig

    @render.ui
    def sim_table():
        snap = snapshot_rows(sim_data(), input.snap_day())
        table = pd.DataFrame({"id": snap["id"], "day": (snap["time"] / DAY).round().astype(int)})
        table["mass"] = snap["mass"].map("{:.4g}".format)
        for col in ("x", "y", "vx", "vy"):
            table[col] = snap[col].map("{:,.2f}".format)
        return html_table(table)

    # --- Conservation tab ---
    # Always the raw, inertial run: a body-centred frame is non-inertial, so
    # energy and momentum are not conserved in it.
    @reactive.calc
    def conservation():
        return compute_conservation(require_sim())

    @render.ui
    def conservation_summary():
        res, params = conservation(), run_params()
        cq = res["cq"]

        def max_abs(values):
            values = values.dropna()
            return "n/a" if values.empty else f"{values.abs().max():.2e}"

        steps = (f"{res['n_used']:,} of {res['n_total']:,} time steps"
                 if res["n_used"] < res["n_total"] else f"{res['n_total']:,} time steps")
        return ui.div(
            ui.tags.b(METHOD_LABELS[params["method"]]), " · ",
            "max |ΔE/E₀| = ", ui.tags.b(max_abs(cq["energy_error"])), " · ",
            "max |ΔL/L₀| = ", ui.tags.b(max_abs(cq["angular_momentum_error"])), " · ",
            f"inertial frame, {steps}", class_="tab-summary")

    @render.plot
    def conservation_plot():
        cq, params = conservation()["cq"], run_params()
        cq = cq.loc[cq["time"] > cq["time"].min()]  # the first step is exactly 0
        if cq.empty:
            return message_figure("Run for at least two time steps to measure conservation.")

        fig, ax = new_figure(f"{params['scenario_name']}  —  {METHOD_LABELS[params['method']]}",
                             None, "Day", "|Relative error|  (log scale)")
        series = [("Energy", "energy_error"), ("Angular momentum", "angular_momentum_error"),
                  ("Linear momentum", "momentum_error")]
        for (label, col), color in zip(series, SERIES):
            errors = cq[col].abs()
            if errors.notna().any():
                ax.plot(cq["time"] / DAY, errors.clip(lower=CONSERVATION_FLOOR), color=color,
                        linewidth=2, label=label)
        ax.set_yscale("log")
        ax.minorticks_off()
        for start in params["segment_starts"]:
            ax.axvline(start, color=MUTED, linewidth=1, linestyle="--")
        add_legend(ax, "Quantity")
        return fig

    # --- Orbital Elements tab ---
    @render.ui
    def elements_caption():
        snap = snapshot_rows(require_sim(), input.snap_day())
        primary = snap["id"].loc[snap["mass"].idxmax()]
        tied = (snap["mass"] == snap["mass"].max()).sum() > 1
        note = (" (the bodies tie for the largest mass, so the first listed is used),"
                if tied else ", the most massive body,")
        return ui.div(
            "Osculating orbital elements at ", ui.tags.b(f"day {snap['time'].iloc[0] / DAY:.4g}"),
            " (set the day with the Snapshot slider), from ", ui.tags.code("get_orbital_elements()"),
            " on the inertial run. Each body is measured relative to ", ui.tags.b(primary), note,
            " unless the ", ui.tags.i("Relative to"), " column names its natural parent.",
            class_="tab-summary")

    @render.ui
    def elements_table():
        raw = require_sim()
        snap = snapshot_rows(raw, input.snap_day())
        elements = snapshot_elements(snap, run_params()["scenario"],
                                     raw.attrs.get("G", orb.gravitational_constant))
        if elements is None:
            return ui.div("This scenario has no orbiting bodies to describe.", class_="tab-caption")
        return html_table(elements)

    @render.code
    def code_output():
        require_sim()
        return reproduce_code(run_params())


app = App(app_ui, server)
