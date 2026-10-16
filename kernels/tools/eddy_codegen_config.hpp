// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Approximation settings, not additional material parameters. Frequencies are
// for the reference radius. For R=sr*Rref, a physical f maps to f*sr^2 here.
namespace eddy_codegen_config {
inline constexpr double frequency_min_hz = 0.1;
inline constexpr double frequency_max_hz = 300.0;
inline constexpr int fit_samples = 192;
inline constexpr int validation_samples = 769;
inline constexpr int retained_slow_modes = 2;
inline constexpr double fastest_tau_ratio = 0.001;
inline constexpr double maximum_relative_error = 0.001;
} // namespace eddy_codegen_config
