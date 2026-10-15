// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once

#include <cstddef>
#include <cstdint>

#include "1dmodel-hysteresis.hpp"

struct initial_scan_settings {
    double sr;
    double somega;
    double dt;
    std::int64_t steps;
    std::int64_t window_steps;
    std::int64_t sample_stride;
    solver_params reference;
    int num_threads;
};

constexpr std::size_t initial_scan_metric_count = 19;

// Output is row-major [S0 index, v0 index]. Each worker owns whole pixels.
void scan_initial_grid_cpu(const double *v0_values, std::size_t v0_count,
                           const double *s0_values, std::size_t s0_count,
                           const initial_scan_settings &settings,
                           std::int32_t *status, double *metrics);
