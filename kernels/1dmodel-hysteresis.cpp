#include "1dmodel-hysteresis.hpp"

#include <stdexcept>
#include <string>

namespace {

state evolve_impl(const state &initial, const real_t dt, const int steps, const solver_params &params, bool &sliding)
{
    if (dt <= REAL_LITERAL(0.0)) throw std::invalid_argument("evolve: dt must be positive");
    if (steps < 0) throw std::invalid_argument("evolve: steps must be non-negative");

    state ball = initial;
    const real_t slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    sliding = real_fabs(slip_velocity) > REAL_LITERAL(1.0e-10);
    real_t slip_sign = sliding ? signum(slip_velocity) : REAL_LITERAL(0.0);
    if (!sliding) {
        vec2 h_ext = {H_EXT * real_cos(ball.phase), H_EXT * real_sin(ball.phase)};
        real_t N = params.gamma_g - params.gamma / (REAL_LITERAL(2.0) + MU_R) *
                                (-(MU_R - 1) * H_EXT * H_EXT +
                                 (1 - C) / (1 - XI_JA) * (ball.q * h_ext));
        if (N < REAL_LITERAL(0.0)) throw std::runtime_error("evolve: N < 0 at the initial state");
        real_t M_magnet = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(ball.q, h_ext);
        real_t theta_dot = ETA / (1 + ETA) * (1 - params.alpha) * M_magnet -
                           params.alpha / (1 + ETA) * params.delta * ball.theta_dot * real_fabs(ball.theta_dot);
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
