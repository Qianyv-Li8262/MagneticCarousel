#include "1dmodel-hysteresis.hpp"

#include <stdexcept>
#include <string>

namespace
{

state evolve_impl(const state &initial, const real_t dt, const int steps, const solver_params &params, bool &sliding)
{
    if (dt <= REAL_LITERAL(0.0))
        throw std::invalid_argument("evolve: dt must be positive");
    if (steps < 0)
        throw std::invalid_argument("evolve: steps must be non-negative");

    state ball = initial;
    const real_t slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    sliding = real_fabs(slip_velocity) > REAL_LITERAL(1.0e-10);
    real_t slip_sign = sliding ? signum(slip_velocity) : REAL_LITERAL(0.0);
    if (!sliding) {
        const field_evaluation field_lab = evaluate_field(static_cast<real_t>(ball.phase));
        const real_t body_angle = -static_cast<real_t>(ball.theta);
        const vec2 h_ext = rotate(field_lab.field, body_angle);
        const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
        const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
        const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
        const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
        const real_t N = params.gamma_g - params.gamma * (A * (h_ext * h_ext_height_derivative) -
                                                          E * (ball.q * h_ext_height_derivative));
        if (N < REAL_LITERAL(0.0))
            throw std::runtime_error("evolve: N < 0 at the initial state");
        const real_t M_magnet = -params.gamma_m * E * cross(ball.q, h_ext);
        const real_t F_magnet = ETA * params.alpha * params.alpha * params.gamma_m *
                                (E * (ball.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
        const real_t theta_dot =
            ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet - F_magnet / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
            params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * ball.theta_dot * real_fabs(ball.theta_dot);
        if (real_fabs(theta_dot - M_magnet) * ETA * params.alpha > MU * N) {
            sliding = true;
            slip_sign = -signum(theta_dot - M_magnet);
        } else {
            slip_sign = REAL_LITERAL(0.0);
            project_pure_rolling(ball, params);
        }
    }

    for (int i = 0; i < steps; ++i) {
        const int flag = fwd1step(ball, dt, params, sliding, slip_sign);
        if (flag != 0)
            throw std::runtime_error("evolve: fwd1step failed at step " + std::to_string(i) + " with flag " +
                                     std::to_string(flag));
    }
    return ball;
}

} // namespace

state evolve(const state &initial, const real_t dt, const int steps, const solver_params &params)
{
    bool sliding;
    return evolve_impl(initial, dt, steps, params, sliding);
}

state evolve_with_mode(const state &initial, const real_t dt, const int steps, const solver_params &params,
                       bool &sliding)
{
    return evolve_impl(initial, dt, steps, params, sliding);
}
