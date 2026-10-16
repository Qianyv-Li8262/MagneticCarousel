// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

// Compile this kernel with nvcc to check the complete device-call path.
GLOBAL void test_fwd_cuda(State<kEddyCurrentModeCount> *states, AdvanceStepResult *results, const SolverParams params,
                          const real_t dt)
{
    State<kEddyCurrentModeCount> state = states[0];
    bool sliding = false;
    Vec2 direction{};
    bool field_valid = false;
    results[0] = fwd1step(dt, params, state, sliding, direction, field_valid);
    states[0] = state;
}

// Force device-side value reads, including constants not reached by this step.
GLOBAL void test_config_constants_cuda(real_t *out)
{
    out[0] = Mu_r;
    out[1] = Alpha_JA;
    out[2] = C_JA;
    out[3] = K;
    out[4] = Xi_JA;
    out[5] = Beta;
    out[6] = Eta;
    out[7] = Mu;
    out[8] = height;
    out[9] = magnet_phi0;
    out[10] = kLUTLowerBoundRatio;
    out[11] = kLUTUpperBoundRatio;
    out[12] = static_cast<real_t>(magnet_number + kFieldFourierModeCount + kSplineNodes +
                                  kStepRefinementRecursionDepth);
    out[13] = static_cast<real_t>(kFieldEvalMode == FieldEvalMode::kExact);
    out[14] = static_cast<real_t>(kLUTCacheMode == LUTCacheMode::kNoCache);
    out[15] = static_cast<real_t>(kAttitudeNormalizeMode == AttitudeNormalizeMode::kNewtonOnce);
    out[16] = static_cast<real_t>(kUseBigEndian);
}
