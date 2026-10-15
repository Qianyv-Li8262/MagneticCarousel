// SPDX-License-Identifier: AGPL-3.0-only
#include "2dmodel-hysteresis.hpp"
#include <cassert>
#include <initializer_list>

int main()
{
    const SolverParams params = {REAL_LITERAL(1.0), REAL_LITERAL(0.0), REAL_LITERAL(10.0),
                                 REAL_LITERAL(0.0), REAL_LITERAL(0.0)};
    const Vec3 magnetic = make_Vec3(REAL_LITERAL(2.0), REAL_LITERAL(-1.0), REAL_LITERAL(3.0));
    const Vec3 force = make_Vec3(REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(-2.0));
    State state{};
    state.pos.x = REAL_LITERAL(4.0);
    state.attitude.real = REAL_LITERAL(1.0);

    // Passive at every speed, finite at zero, independent of normal-axis spin.
    for (real_t w : {REAL_LITERAL(0.0), REAL_LITERAL(1e-8), REAL_LITERAL(0.1),
                     REAL_LITERAL(1.5), REAL_LITERAL(20.0)}) {
        state.omega = make_Vec3(w, -w, REAL_LITERAL(4.0));
        const Vec3 total = add_gaussian_rolling_resistance(params, magnetic, state, force);
        const Vec3 resistance = total - magnetic;
        assert(real_isfinite(length(total)));
        assert(resistance * state.omega <= REAL_LITERAL(0.0));
        assert(total.z == magnetic.z);
        if constexpr (!kEnableRollingResistance)
            assert(total.x == magnetic.x && total.y == magnetic.y);
        if (w == REAL_LITERAL(0.0))
            assert(total.x == magnetic.x && total.y == magnetic.y);
    }

    // More magnetic normal attraction increases dry rolling loss, at fixed spin.
    state.omega = make_Vec3(REAL_LITERAL(0.0), REAL_LITERAL(1.0), REAL_LITERAL(0.0));
    const Vec3 zero{};
    const Vec3 weak = add_gaussian_rolling_resistance(params, zero, state, zero);
    const Vec3 strong = add_gaussian_rolling_resistance(params, zero, state, force);
    if constexpr (kEnableRollingResistance)
        assert(strong.y < weak.y);
    else
        assert(length(weak) == REAL_LITERAL(0.0) && length(strong) == REAL_LITERAL(0.0));

    // The same resistance enters the RHS and contact mode: an uphill pure-rolling
    // step dissipates energy and remains on v=(alpha*omega_y,-alpha*omega_x).
    state.speed = make_Vec2(-real_sin(kTiltAxisAzimuthRad), real_cos(kTiltAxisAzimuthRad));
    state.omega = make_Vec3(-state.speed.y, state.speed.x, REAL_LITERAL(0.0));
    bool sliding = true;
    Vec2 direction{};
    assert(resolve_zero_slip_mode(state, params, sliding, direction) == AdvanceStepResult::kStepOk);
    assert(!sliding);
    const real_t energy_before = state.speed * state.speed + Eta * (state.omega * state.omega);
    bool field_valid = false;
    assert(fwd1step(REAL_LITERAL(0.001), params, state, sliding, direction, field_valid) == AdvanceStepResult::kStepOk);
    const real_t energy_after = state.speed * state.speed + Eta * (state.omega * state.omega);
    assert(energy_after < energy_before);
    assert(!sliding);
    assert(real_fabs(static_cast<real_t>(state.speed.x - params.alpha * state.omega.y)) < REAL_LITERAL(1e-6));
    assert(real_fabs(static_cast<real_t>(state.speed.y + params.alpha * state.omega.x)) < REAL_LITERAL(1e-6));

    // Following air force changes direction with position, adds no normal load,
    // and vanishes exactly when switched off.
    SolverParams pushed = params;
    pushed.reverse_push_acceleration = REAL_LITERAL(0.2);
    for (Vec2 position : {make_Vec2(REAL_LITERAL(4.0), REAL_LITERAL(0.0)),
                          make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(4.0))}) {
        const Vec3 base = apply_inclined_disk_force(params, force, position);
        const Vec3 total = apply_inclined_disk_force(pushed, force, position);
        const Vec2 delta = make_Vec2(total.x - base.x, total.y - base.y);
        assert(real_fabs(delta * position) < REAL_LITERAL(1e-6));
        assert(position.x * delta.y - position.y * delta.x < REAL_LITERAL(0.0));
        assert(real_fabs(length(delta) - REAL_LITERAL(0.2)) < REAL_LITERAL(1e-6));
        assert(total.z == base.z);
    }
}
