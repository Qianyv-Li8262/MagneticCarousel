// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

#include <cassert>

int main()
{
    static_assert(kFieldEvalMode == FieldEvalMode::kFourierSpline);
    SolverParams params{};
    State<kEddyCurrentModeCount> state{};
    state.pos.x = REAL_LITERAL(7.0);
    state.attitude.real = REAL_LITERAL(1.0);
    const State<kEddyCurrentModeCount> before = state;
    bool sliding = false;
    Vec2 direction{};
    bool valid = false;
    assert(fwd1step(REAL_LITERAL(0.001), params, state, sliding, direction, valid) ==
           AdvanceStepResult::kStepLUTOutOfDomain);
    assert(state.pos.x == before.pos.x && !sliding && !valid);
}
