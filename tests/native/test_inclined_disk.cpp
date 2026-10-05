// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

#include <cassert>
#include <initializer_list>

static void close(real_t actual, real_t expected)
{
    assert(real_fabs(actual - expected) <= REAL_LITERAL(2e-6) *
           (REAL_LITERAL(1.0) + real_fabs(expected)));
}

int main()
{
    const real_t theta = kEnableInclinedDisk ? kTiltAngleRad : REAL_LITERAL(0.0);
    const Vec2 downhill = make_Vec2(real_sin(kTiltAxisAzimuthRad), -real_cos(kTiltAxisAzimuthRad));
    SolverParams params = {REAL_LITERAL(1.0), REAL_LITERAL(0.0), REAL_LITERAL(10.0),
                           REAL_LITERAL(0.0), REAL_LITERAL(0.0)};

    // Downward magnetic attraction strengthens the drive; upward force weakens it.
    // Neither case changes the magnetic normal force or the existing horizontal force.
    for (real_t z : {REAL_LITERAL(-2.0), REAL_LITERAL(2.0)}) {
        const Vec3 magnetic = make_Vec3(REAL_LITERAL(3.0), REAL_LITERAL(4.0), z);
        const Vec3 effective = apply_inclined_disk_force(params, magnetic);
        close(effective.x, magnetic.x + theta * (params.gamma_g - z) * downhill.x);
        close(effective.y, magnetic.y + theta * (params.gamma_g - z) * downhill.y);
        assert(effective.z == magnetic.z);
    }

    State state{};
    state.pos.x = REAL_LITERAL(4.0);
    state.attitude.real = REAL_LITERAL(1.0);
    State derivative{};
    Vec3 force{}, torque{};
    const Vec2 drive = theta * params.gamma_g * downhill;

    // With no magnetic field force or drag, pure rolling has a = g*theta/(1+eta).
    assert(eval_rhs<true>(state, derivative, params, false, downhill,
                          nullptr, nullptr, nullptr, &force, &torque));
    close(force.x, drive.x);
    close(force.y, drive.y);
    assert(force.z == REAL_LITERAL(0.0));
    close(derivative.speed.x, drive.x / (REAL_LITERAL(1.0) + Eta));
    close(derivative.speed.y, drive.y / (REAL_LITERAL(1.0) + Eta));
    close(derivative.speed.x, params.alpha * derivative.omega.y);
    close(derivative.speed.y, -params.alpha * derivative.omega.x);

    // Sliding uses the same drive and retains mu*gamma_g as the friction magnitude.
    assert(eval_rhs<true>(state, derivative, params, true, downhill,
                          nullptr, nullptr, nullptr, &force, &torque));
    close(derivative.speed.x, drive.x - Mu * params.gamma_g * downhill.x);
    close(derivative.speed.y, drive.y - Mu * params.gamma_g * downhill.y);

    // Static contact must agree with the analytic rolling/sliding threshold.
    bool sliding = false;
    Vec2 direction{};
    assert(resolve_zero_slip_mode(state, params, sliding, direction) == AdvanceStepResult::kStepOk);
    const bool expected_sliding = real_fabs(theta) > (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu;
    assert(sliding == expected_sliding);
    if (sliding) {
        const real_t sign = theta < REAL_LITERAL(0.0) ? REAL_LITERAL(-1.0) : REAL_LITERAL(1.0);
        close(direction.x, sign * downhill.x);
        close(direction.y, sign * downhill.y);
    } else {
        // A full rolling step must use the same drive at its start, midpoint and end.
        const real_t dt = REAL_LITERAL(0.001);
        bool cache_valid = false;
        assert(fwd1step(dt, params, state, sliding, direction, cache_valid) == AdvanceStepResult::kStepOk);
        assert(!sliding);
        close(static_cast<real_t>(state.speed.x), dt * drive.x / (REAL_LITERAL(1.0) + Eta));
        close(static_cast<real_t>(state.speed.y), dt * drive.y / (REAL_LITERAL(1.0) + Eta));
    }
}
