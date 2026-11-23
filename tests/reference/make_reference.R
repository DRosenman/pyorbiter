# Regenerates r_reference.json from the orbitr R package (>= 1.0.0).
# pyorbiter's parity tests compare against these numbers.
#   Rscript tests/reference/make_reference.R
library(orbitr)

rows <- function(d) lapply(seq_len(nrow(d)), function(k) as.list(as.data.frame(d)[k, , drop = FALSE]))
last_step <- function(sim) rows(sim[sim$time == max(sim$time), ])
last_row  <- function(d) as.list(as.data.frame(d)[nrow(d), , drop = FALSE])

earth_sun <- create_system() |>
  add_body("Sun", mass = mass_sun) |>
  add_body("Earth", mass = mass_earth, x = distance_earth_sun, vy = speed_earth)

ref <- list(orbitr_version = as.character(packageVersion("orbitr")))

for (m in c("verlet", "euler_cromer", "euler")) {
  sim <- simulate_system(earth_sun, time_step = seconds_per_day,
                         duration = 365 * seconds_per_day, method = m)
  ref[[paste0("earth_sun_", m)]] <- last_step(sim)
}

sim <- simulate_system(earth_sun, time_step = seconds_per_day, duration = 365 * seconds_per_day)
ref$earth_sun_n_rows <- nrow(sim)
ref$earth_sun_conserved <- last_row(conserved_quantities(sim))
ref$earth_sun_continued <- last_step(
  continue_simulation(sim, time_step = seconds_per_hour, duration = 10 * seconds_per_day))

ref$solar_system_bodies <- rows(get_bodies(load_solar_system()))

solar <- simulate_system(load_solar_system(), time_step = seconds_per_hour,
                         duration = 30 * seconds_per_day)
ref$solar_last <- last_step(solar)
ref$solar_elements_moon <- last_row(get_orbital_elements(solar, "Moon", "Earth"))
ref$solar_elements_mars <- last_row(get_orbital_elements(solar, "Mars", "Sun"))
ref$solar_conserved <- last_row(conserved_quantities(solar))
ref$solar_barycentric <- last_step(shift_reference_frame(solar, "barycenter"))
ref$solar_earth_frame <- last_step(shift_reference_frame(solar, "Earth", keep_center = FALSE))

flyby <- create_system() |>
  add_sun() |>
  add_body_keplerian("Oumuamua", mass = 1e10, parent = "Sun",
                     a = -1.2765 * distance_earth_sun, e = 1.2, arg_pe = 90, nu = -130)
ref$hyperbolic_bodies <- rows(get_bodies(flyby))
fly <- simulate_system(flyby, time_step = seconds_per_hour, duration = 200 * seconds_per_day)
ref$hyperbolic_elements <- last_row(get_orbital_elements(fly, "Oumuamua", "Sun"))

soft <- create_system() |>
  add_body("A", mass = 1e30, vy = 5000) |>
  add_body("B", mass = 1e30, x = 1e11, vy = -2500) |>
  add_body("C", mass = 1e30, x = 5e10, y = 8.66e10, z = 1e9, vy = -2500, vz = 100) |>
  simulate_system(time_step = seconds_per_hour, duration = 20 * seconds_per_day, softening = 1e9)
ref$softened_last <- last_step(soft)
ref$softened_energy <- last_row(get_energy(soft))

jsonlite::write_json(ref, "tests/reference/r_reference.json",
                     digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null")
