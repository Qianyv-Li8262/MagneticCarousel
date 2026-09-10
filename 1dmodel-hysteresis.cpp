#include "1dmodel-hysteresis.hpp"

#include <stdexcept>
#include <string>

state evolve(const state &initial, const float dt, const int steps)
{
    if (dt <= 0.0f) throw std::invalid_argument("evolve: dt must be positive");
    if (steps < 0) throw std::invalid_argument("evolve: steps must be non-negative");

    state ball = initial;
    const float slip_velocity = ball.xdot + ALPHA * ball.theta_dot;
    bool sliding = fabsf(slip_velocity) > 1.0e-10f;
    float slip_sign = sliding ? signum(slip_velocity) : 0.0f;
    if (!sliding) {
        vec2 q_rotated = rotate(ball.q, ball.theta);
        vec2 h_0 = {H_EXT * cosf(ball.t - ball.x), H_EXT * sinf(ball.t - ball.x)};
        float N = GAMMA_G - GAMMA / (2.0f + MU_R) *
                                (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - SMALL_KHI_JA) * (q_rotated * h_0));
        if (N < 0.0f) throw std::runtime_error("evolve: N < 0 at the initial state");
        float M_magnet = -GAMMA_M * (1 - C) / ((1 - SMALL_KHI_JA) * (MU_R + 2)) * cross(q_rotated, h_0);
        float theta_dot = ETA / (1 + ETA) * (1 - ALPHA) * M_magnet -
                          ALPHA / (1 + ETA) * DELTA * ball.theta_dot * fabsf(ball.theta_dot);
        if (fabsf(theta_dot - M_magnet) * ETA * ALPHA > MU * N) {
            sliding = true;
            slip_sign = -signum(theta_dot - M_magnet);
        } else {
            slip_sign = 0.0f;
            ball.theta_dot = -ball.xdot / ALPHA;
        }
    }

    for (int i = 0; i < steps; ++i) {
        const int flag = fwd1step(ball, dt, sliding, slip_sign);
        if (flag != 0)
            throw std::runtime_error("evolve: fwd1step failed at step " + std::to_string(i) + " with flag " +
                                     std::to_string(flag));
    }
    return ball;
}
