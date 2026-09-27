// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once

#include <cstddef>
#include <cstdint>

#include "1dmodel-hysteresis.hpp"

struct phase_boundary_settings {
    double somega_lower;
    double somega_upper;
    double tolerance;
    double phase_rate_threshold;
    double dt;
    std::int64_t total_steps;
    std::int64_t window_steps;
    std::int64_t sample_stride;
    int max_iterations;
};

struct phase_boundary_point {
    double sr;
    double lower;
    double upper;
    double lower_mean_phase_rate;
    double upper_mean_phase_rate;
    int iterations;
    int status;
};

// status: 0=success, 1=endpoints have same classification, negative=evolution/input error.
void bisect_phase_boundaries_cpu(phase_boundary_point *output, std::size_t count,
                                 double sr_lower, double sr_upper,
                                 const phase_boundary_settings &settings,
                                 const solver_params &reference, int num_threads);
