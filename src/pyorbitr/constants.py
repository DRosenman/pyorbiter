"""Physical constants in SI units (meters, kilograms, seconds).

The values match the orbitr R package exactly, so a simulation built from
these constants reproduces the R result.
"""

# --- Gravitational constant (m^3 kg^-1 s^-2) ---
gravitational_constant = 6.6743e-11

# --- Time conversions (seconds) ---
seconds_per_hour = 3600
seconds_per_day = 86400
seconds_per_year = 86400 * 365.25

# --- Masses (kg) ---
mass_sun = 1.989e30
mass_earth = 5.972e24
mass_moon = 7.342e22
mass_mars = 6.417e23
mass_jupiter = 1.898e27
mass_saturn = 5.683e26
mass_venus = 4.867e24
mass_mercury = 3.301e23
mass_uranus = 8.681e25
mass_neptune = 1.024e26
mass_pluto = 1.309e22

# --- Orbital distances: semi-major axes (m) ---
distance_earth_sun = 1.496e11
distance_earth_moon = 3.844e8
distance_mars_sun = 2.279e11
distance_jupiter_sun = 7.785e11
distance_venus_sun = 1.082e11
distance_mercury_sun = 5.791e10
distance_saturn_sun = 1.434e12
distance_uranus_sun = 2.871e12
distance_neptune_sun = 4.495e12
distance_pluto_sun = 5.906e12

# --- Mean orbital speeds (m/s) ---
speed_earth = 29780
speed_moon = 1022
speed_mars = 24070
speed_jupiter = 13060
speed_venus = 35020
speed_mercury = 47360
speed_saturn = 9680
speed_uranus = 6800
speed_neptune = 5430
speed_pluto = 4740

__all__ = [name for name in dir() if not name.startswith("_")]
