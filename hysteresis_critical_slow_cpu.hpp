#pragma once

#include "1dmodel-hysteresis.hpp"
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

static_assert(std::is_same_v<real_t, double> && std::numeric_limits<double>::is_iec559,
              "The critical-slowing CPU core requires FP64: build with USE_FP64");

struct critical_slow_result {
    int answer_state; // -1: evolution error, 0: undecided, 1: locked, 2: rising
    double transit_time; // locked: midpoint of first in-band 1000-time-unit mean in final run; rising: global minimum endpoint; -1 if unconfirmed
};

inline constexpr int critical_slow_confirm_evolve_steps = 50000;
inline constexpr int critical_slow_confirm_window = 100; // recovery
inline constexpr int critical_slow_lock_confirm_window = 50;

// One call is one CUDA-thread-equivalent trajectory, with an explicit v0.
critical_slow_result evolve_critical_slow(double v0, double dt, std::int64_t max_confirm,
                                         const solver_params &params);

// Synchronous CPU kernel: all arrays have k elements and must not overlap.
// Input is immutable until return; outputs are written in original index order.
// num_threads > 0; at most min(k, num_threads) workers are started per call.
void calc_slow_cpu(std::int32_t *answer_state, double *transit_time, const double *v0,
                   std::size_t k, double dt, std::int64_t max_confirm,
                   solver_params params, int num_threads);
