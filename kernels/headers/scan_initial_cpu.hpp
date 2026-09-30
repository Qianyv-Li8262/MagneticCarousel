// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include "hysteresis_types.hpp"
#include <cstddef>
#include <cstdint>

// Outputs are row-major with S0 as rows and v0 as columns. The caller owns
// non-overlapping buffers of k*k, k*k*20, and k*k*15 elements respectively.
void scan_initial_grid_cpu(const real_t *v0_values, const real_t *s0_values, std::size_t k,
                           real_t dt, int steps, int window_steps, int sample_stride,
                           real_t sr, real_t somega, const SolverParams &reference,
                           int num_threads, std::int32_t *status, real_t *metrics,
                           real_t *states);
