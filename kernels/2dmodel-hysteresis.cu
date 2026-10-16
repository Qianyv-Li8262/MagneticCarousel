// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

// NVRTC entry point for one trajectory. No host CUDA build is required.
// Status 0 means success; positive values map to AdvanceStepResult + 1.
extern "C" GLOBAL void evolve_trajectory_cuda(State<kEddyCurrentModeCount> *states, const SolverParams params,
                                                const real_t dt, const int steps,
                                                bool *sliding_out, Vec2 *sliding_dir_out,
                                                int *status_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || steps < 0 ||
        !real_isfinite(params.alpha) || params.alpha <= REAL_LITERAL(0.0) ||
        !real_isfinite(params.delta) || params.delta < REAL_LITERAL(0.0)) {
        *status_out = -1;
        return;
    }

    State<kEddyCurrentModeCount> state = *states;
    const Vec2 slip = vec_cast<real_t>(state.speed) +
                      params.alpha * make_Vec2(-static_cast<real_t>(state.omega.y),
                                                static_cast<real_t>(state.omega.x));
    const real_t magnitude = length(slip);
    if (!real_isfinite(magnitude)) {
        *status_out = -1;
        return;
    }
    bool sliding = magnitude > REAL_LITERAL(1.0e-10);
    Vec2 sliding_dir = sliding ? slip / magnitude : make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    if (!sliding)
        project_pure_rolling(state, params);

    bool field_cache_valid = false;
    FieldCache cache{};
    FieldResult field_cache{};
    for (int i = 0; i < steps; ++i) {
        const AdvanceStepResult result = fwd1step(dt, params, state, sliding, sliding_dir,
                                                  field_cache_valid, &cache, nullptr, &field_cache);
        if (result != AdvanceStepResult::kStepOk) {
            *status_out = static_cast<int>(result) + 1;
            return;
        }
    }
    *states = state;
    *sliding_out = sliding;
    *sliding_dir_out = sliding_dir;
    *status_out = 0;
}
