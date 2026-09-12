#pragma once

#include "1dmodel-hysteresis.hpp"
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

static_assert(std::is_same_v<real_t, double> && std::numeric_limits<double>::is_iec559,
              "The critical-slowing CPU core requires FP64: build with USE_FP64");

struct critical_slow_result {
    int answer_state; // -1: evolution error, 0: no confirmed phase lock, 1: phase locked
    double transit_time; // midpoint of first in-band 1000-time-unit phase-rate mean; -1 if unconfirmed
};

enum class critical_slow_scan_mode : std::int32_t {
    somega = 0, // fixed sr, scan somega
    sr = 1      // fixed somega, scan sr
};

// At dt=0.001, sample every 50 time units. Preserve the physical
// phase-lock confirmation duration of 2500 time units.

// Detector sampling parameters.
// Assumes integration timestep dt = 0.001.
//
// One detector sample spans:
//   sample_dt = dt * critical_slow_confirm_evolve_steps
//
// Keep the physical phase-lock confirmation duration fixed:
//   locking: 2500 -> lock_confirm_window * sample_dt = 2500
//
// Therefore, when changing evolve_steps, scale the window size inversely.
// Current values: evolve_steps = 50000 -> sample_dt = 50,
//                 lock window = 50.
inline constexpr int critical_slow_confirm_evolve_steps = 50000;
inline constexpr int critical_slow_lock_confirm_window = 50;
inline constexpr double critical_slow_phase_rate_tolerance = 0.05;

// One call is one CUDA-thread-equivalent trajectory.  The complete initial
// state [t, x, xdot, theta, theta_dot, phase, qx, qy] is fixed at zero.
critical_slow_result evolve_critical_slow(double dt, std::int64_t max_confirm,
                                         const solver_params &params);

// Synchronous CPU kernel: all arrays have k elements and must not overlap.
// scan_values contains somega when mode==somega and sr when mode==sr;
// fixed_scale is the other scale.  reference contains the unscaled parameters.
// Input is immutable until return; outputs are written in original index order.
// num_threads > 0; at most min(k, num_threads) workers are started per call.
void calc_slow_cpu(std::int32_t *answer_state, double *transit_time, const double *scan_values,
                   std::size_t k, double fixed_scale, critical_slow_scan_mode mode,
                   double dt, std::int64_t max_confirm, solver_params reference,
                   int num_threads);
