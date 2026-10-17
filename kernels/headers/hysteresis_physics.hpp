// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once

// SI reference inputs. Keep independent of hysteresis_types.hpp so both
// code generators and the CPU/CUDA numeric types can include this header.
namespace hysteresis_physics {
inline constexpr double pi = 3.141592653589793238462643383279502884;
// Apparatus reference for the existing Python magnet-count scaling convention.
// This is not the active magnet_number from hysteresis_config_field.hpp.
inline constexpr int reference_magnet_count = 8;
inline constexpr double Rref = 0.006;                    // Ball radius [m].
inline constexpr double L = 0.070;                       // Reference magnet ring radius [m].
inline constexpr double omega_ref = 100.0 * 2.0 * pi / 60.0; // Disk rate [rad/s].
inline constexpr double rho_b = 7850.0;                  // Ball density [kg/m^3].
inline constexpr double rho_air = 1.225;                 // Air density [kg/m^3].
inline constexpr double Cd = 0.47;                       // Quadratic drag coefficient.
inline constexpr double g = 9.81;                        // Gravity [m/s^2].
inline constexpr double M_ref = 1.1e6;                   // Magnetization unit [A/m], not fitted Ms.
inline constexpr double mu0 = 4.0 * pi * 1.0e-7;         // Model permeability convention [H/m].
// Active 56c6 reduced-overshoot fit; not a measured ball property.
inline constexpr double sigma = 9331270.9871829;         // Electrical conductivity [S/m].

static_assert(Rref > 0.0 && L > 0.0 && omega_ref > 0.0 && rho_b > 0.0);
static_assert(rho_air >= 0.0 && Cd >= 0.0 && g > 0.0 && M_ref > 0.0 && mu0 > 0.0);
static_assert(sigma > 0.0);
} // namespace hysteresis_physics
