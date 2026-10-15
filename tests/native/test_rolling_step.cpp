// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

#include <cassert>
#include <cmath>

int main()
{
    // Zero normal/gravitational load isolates uniform motion even with tilt enabled.
    SolverParams params = {REAL_LITERAL(1.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0),
                           REAL_LITERAL(0.0), REAL_LITERAL(0.0)};
    State state{};
    state.pos.x = REAL_LITERAL(4.0);
    state.speed.x = REAL_LITERAL(1.0);
    state.omega.y = REAL_LITERAL(1.0);
    state.attitude.real = REAL_LITERAL(1.0);
    bool sliding = false;
    Vec2 direction = {REAL_LITERAL(0.0), REAL_LITERAL(0.0)};
    bool valid = false;
    FieldResult field{};
    const real_t dt = REAL_LITERAL(0.001);

    // The public full-step wrapper agrees with a directly accepted step.
    State wrapped = state;
    bool wrapped_sliding = sliding;
    Vec2 wrapped_dir = direction;
    bool wrapped_valid = valid;
    FieldResult wrapped_field{};
    assert(fwd1step(dt, params, wrapped, wrapped_sliding, wrapped_dir, wrapped_valid,
                    nullptr, nullptr, &wrapped_field) == AdvanceStepResult::kStepOk);

    // No magnetic force or drag: a rolling ball keeps its velocity.
    assert(try_advance_1_step(dt, params, state, sliding, direction, valid, nullptr, nullptr, &field) ==
           AdvanceStepResult::kStepOk);
    assert(wrapped.pos.x == state.pos.x && wrapped.speed.x == state.speed.x);
    assert(wrapped_sliding == sliding && wrapped_valid == valid);
    assert(wrapped_field.h.x == field.h.x && wrapped_field.h.y == field.h.y);
    assert(!sliding && valid);
    assert(real_fabs(static_cast<real_t>(state.pos.x) - (REAL_LITERAL(4.0) + dt)) < REAL_LITERAL(1e-6));
    assert(real_fabs(static_cast<real_t>(state.speed.x) - REAL_LITERAL(1.0)) < REAL_LITERAL(1e-6));
    assert(real_fabs(static_cast<real_t>(state.speed.x - params.alpha * state.omega.y)) < REAL_LITERAL(1e-6));
    const FieldResult expected = eval_mag_field(state.pos, state.down_disk_angle);
    assert(field.h.x == expected.h.x && field.h.y == expected.h.y && field.h.z == expected.h.z);
    for (int i = 0; i < 5; ++i)
        assert(field.hGrad3d.component[i] == expected.hGrad3d.component[i]);

    // Cross-step cache reuse gives the same accepted state as recomputation.
    State uncached = state;
    bool uncached_mode = sliding;
    Vec2 uncached_dir = direction;
    bool uncached_valid = false;
    assert(try_advance_1_step(dt, params, uncached, uncached_mode, uncached_dir, uncached_valid) ==
           AdvanceStepResult::kStepOk);
    assert(!uncached_valid);
    assert(try_advance_1_step(dt, params, state, sliding, direction, valid, nullptr, nullptr, &field) ==
           AdvanceStepResult::kStepOk);
    assert(static_cast<real_t>(state.pos.x) == static_cast<real_t>(uncached.pos.x));
    assert(static_cast<real_t>(state.q.x) == static_cast<real_t>(uncached.q.x));

    // Excessive drag makes the supplied Rolling mode inconsistent at entry.
    params.gamma_g = REAL_LITERAL(1.0);
    params.delta = REAL_LITERAL(1.0);
    const State before = state;
    const FieldResult field_before = field;
    assert(try_advance_1_step(dt, params, state, sliding, direction, valid, nullptr, nullptr, &field) ==
           AdvanceStepResult::kStepSolverSelfContradict);
    assert(static_cast<real_t>(state.pos.x) == static_cast<real_t>(before.pos.x));
    assert(static_cast<real_t>(state.speed.x) == static_cast<real_t>(before.speed.x));
    assert(!sliding && valid && direction.x == REAL_LITERAL(0.0) && direction.y == REAL_LITERAL(0.0));
    assert(field.h.x == field_before.h.x);
    assert(fwd1step(dt, params, state, sliding, direction, valid, nullptr, nullptr, &field) ==
           AdvanceStepResult::kStepSolverSelfContradict);
    assert(state.pos.x == before.pos.x && field.h.x == field_before.h.x);

    // Zero slip alone does not decide sticking: static friction must support the load.
    State contact{};
    contact.pos.x = REAL_LITERAL(4.0);
    contact.attitude.real = REAL_LITERAL(1.0);
    bool contact_sliding = true;
    Vec2 contact_direction = {REAL_LITERAL(1.0), REAL_LITERAL(0.0)};
    assert(resolve_zero_slip_mode(contact, params, contact_sliding, contact_direction) ==
           AdvanceStepResult::kStepOk);
    assert(!contact_sliding && contact_direction.x == REAL_LITERAL(0.0) &&
           contact_direction.y == REAL_LITERAL(0.0));

    // Unit quadratic drag plus the configured downslope load exceeds mu*N=0.02.
    // The launch direction follows their sum, including its transverse tilt component.
    contact.speed.x = REAL_LITERAL(1.0);
    contact.omega.y = REAL_LITERAL(1.0);
    assert(resolve_zero_slip_mode(contact, params, contact_sliding, contact_direction) ==
           AdvanceStepResult::kStepOk);
    const real_t tilt = kEnableInclinedDisk ? kTiltAngleRad * params.gamma_g : REAL_LITERAL(0.0);
    const Vec2 drive = make_Vec2(tilt * real_sin(kTiltAxisAzimuthRad) - REAL_LITERAL(1.0),
                                 -tilt * real_cos(kTiltAxisAzimuthRad));
    const Vec2 expected_direction = drive / length(drive);
    assert(contact_sliding);
    assert(length(contact_direction - expected_direction) < REAL_LITERAL(1e-6));

    // The returned zero-slip launch state is valid input for the next complete step.
    bool contact_cache_valid = false;
    assert(fwd1step(dt, params, contact, contact_sliding, contact_direction, contact_cache_valid) ==
           AdvanceStepResult::kStepOk);
    assert(contact_sliding && real_isfinite(contact_direction.x) && real_isfinite(contact_direction.y));
}
