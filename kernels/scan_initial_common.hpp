// SPDX-License-Identifier: AGPL-3.0-only
// Shared terminal-window analysis for the initial-state CPU and CUDA scans.
#pragma once

#include "2dmodel-hysteresis.hpp"

namespace initial_scan {

constexpr int metric_count = 20;

DEVICE INLINE void update_mean_variance(real_t value, int count, real_t &mean, real_t &m2)
{
    const real_t delta = value - mean;
    mean += delta / static_cast<real_t>(count);
    m2 += delta * (value - mean);
}

DEVICE INLINE bool finite_state(const State<kEddyCurrentModeCount> &s)
{
    return real_isfinite(static_cast<real_t>(s.down_disk_angle)) &&
           real_isfinite(static_cast<real_t>(s.pos.x)) && real_isfinite(static_cast<real_t>(s.pos.y)) &&
           real_isfinite(static_cast<real_t>(s.speed.x)) && real_isfinite(static_cast<real_t>(s.speed.y)) &&
           real_isfinite(static_cast<real_t>(s.attitude.real)) &&
           real_isfinite(static_cast<real_t>(s.attitude.imagine.x)) &&
           real_isfinite(static_cast<real_t>(s.attitude.imagine.y)) &&
           real_isfinite(static_cast<real_t>(s.attitude.imagine.z)) &&
           real_isfinite(static_cast<real_t>(s.omega.x)) && real_isfinite(static_cast<real_t>(s.omega.y)) &&
           real_isfinite(static_cast<real_t>(s.omega.z)) &&
           real_isfinite(static_cast<real_t>(s.q.x)) && real_isfinite(static_cast<real_t>(s.q.y)) &&
           real_isfinite(static_cast<real_t>(s.q.z));
}

DEVICE INLINE bool observables(const State<kEddyCurrentModeCount> &s, const SolverParams &params,
                               real_t &v, real_t &omega, real_t &phase_rate,
                               real_t &abs_slip, real_t &abs_q)
{
    const real_t x = static_cast<real_t>(s.pos.x), y = static_cast<real_t>(s.pos.y);
    const real_t vx = static_cast<real_t>(s.speed.x), vy = static_cast<real_t>(s.speed.y);
    const real_t wx = static_cast<real_t>(s.omega.x), wy = static_cast<real_t>(s.omega.y);
    const real_t radius2 = x * x + y * y;
    if (!real_isfinite(radius2) || radius2 <= REAL_LITERAL(0.0)) return false;
    const real_t radius = real_sqrt(radius2);
    const real_t m = static_cast<real_t>(magnet_number / 2);
    const real_t cross_pos_speed = x * vy - y * vx;
    const real_t omega_radial = (x * wx + y * wy) / radius;
    const Vec2 slip = make_Vec2(vx - params.alpha * wy, vy + params.alpha * wx);
    v = cross_pos_speed / (radius * m);
    omega = omega_radial / m;
    phase_rate = REAL_LITERAL(1.0) - cross_pos_speed / radius2 - omega;
    abs_slip = length(slip) / m;
    abs_q = length(vec_cast<real_t>(s.q));
    return real_isfinite(v) && real_isfinite(omega) && real_isfinite(phase_rate) &&
           real_isfinite(abs_slip) && real_isfinite(abs_q);
}

DEVICE INLINE int analyze(real_t v0, real_t s0, real_t dt, int steps, int window_steps,
                          int sample_stride, const SolverParams &params, const LUTNode *field_lut,
                          real_t *output_state, real_t *metrics)
{
    if (!real_isfinite(v0) || !real_isfinite(s0) || !real_isfinite(params.alpha) ||
        !real_isfinite(params.delta) || !real_isfinite(params.gamma_g) ||
        !real_isfinite(params.gamma) || !real_isfinite(params.gamma_m) ||
        params.alpha <= REAL_LITERAL(0.0) || params.delta < REAL_LITERAL(0.0)) return -2;
    const real_t m = static_cast<real_t>(magnet_number / 2);
    const real_t omega0 = (s0 - v0) / params.alpha;
    if (!real_isfinite(omega0)) return -2;
    State<kEddyCurrentModeCount> ball{};
    ball.pos.x = static_cast<accumulator_t>(m);
    ball.speed.y = static_cast<accumulator_t>(m * v0);
    ball.omega.x = static_cast<accumulator_t>(m * omega0);
    ball.attitude.real = static_cast<accumulator_t>(REAL_LITERAL(1.0));
    bool sliding = false;
    Vec2 sliding_dir{REAL_LITERAL(0.0), REAL_LITERAL(0.0)};
    const Vec2 initial_slip = vec_cast<real_t>(ball.speed) +
        params.alpha * make_Vec2(-static_cast<real_t>(ball.omega.y),
                                  static_cast<real_t>(ball.omega.x));
    const real_t slip_magnitude = length(initial_slip);
    if (!real_isfinite(slip_magnitude)) return -2;
    sliding = slip_magnitude > REAL_LITERAL(1.0e-10);
    if (sliding) sliding_dir = initial_slip / slip_magnitude;
    else project_pure_rolling(ball, params);
    bool field_cache_valid = false;
    FieldResult field_cache{};
    FieldCache lut_cache{};
    const int window_start = steps > window_steps ? steps - window_steps : 0;
    const int split_step = window_start + (steps - window_start) / 2;
    int count = 0, first_count = 0, second_count = 0, sliding_count = 0;
    real_t mean_v = REAL_LITERAL(0.0), m2_v = REAL_LITERAL(0.0);
    real_t mean_omega = REAL_LITERAL(0.0), mean_phase_rate = REAL_LITERAL(0.0);
    real_t m2_phase_rate = REAL_LITERAL(0.0), mean_abs_slip = REAL_LITERAL(0.0);
    real_t mean_abs_q = REAL_LITERAL(0.0), mean_radius = REAL_LITERAL(0.0);
    real_t first_mean_v = REAL_LITERAL(0.0), second_mean_v = REAL_LITERAL(0.0);
    real_t first_mean_phase_rate = REAL_LITERAL(0.0), second_mean_phase_rate = REAL_LITERAL(0.0);
    real_t minimum_v = REAL_LITERAL(3.402823466e38), maximum_v = -REAL_LITERAL(3.402823466e38);
    real_t global_minimum_v = v0;
    int global_minimum_step = 0;
    bool velocity_recovered = false;
    int status = 0;
    for (int step = 0; step < steps; ++step) {
        const AdvanceStepResult result = fwd1step(dt, params, ball, sliding, sliding_dir,
                                                  field_cache_valid, &lut_cache, field_lut, &field_cache);
        if (result != AdvanceStepResult::kStepOk) { status = static_cast<int>(result) + 1; break; }
        if (!finite_state(ball)) { status = static_cast<int>(AdvanceStepResult::kStepNaNOrInf) + 1; break; }
        real_t v, omega, phase_rate, abs_slip, abs_q;
        if (!observables(ball, params, v, omega, phase_rate, abs_slip, abs_q)) {
            status = static_cast<int>(AdvanceStepResult::kStepNaNOrInf) + 1; break;
        }
        const int completed = step + 1;
        if (v < global_minimum_v) {
            global_minimum_v = v; global_minimum_step = completed; velocity_recovered = false;
        } else if (v > global_minimum_v) velocity_recovered = true;
        if (completed < window_start || (completed - window_start) % sample_stride != 0) continue;
        ++count;
        update_mean_variance(v, count, mean_v, m2_v);
        update_mean_variance(phase_rate, count, mean_phase_rate, m2_phase_rate);
        mean_omega += (omega - mean_omega) / static_cast<real_t>(count);
        mean_abs_slip += (abs_slip - mean_abs_slip) / static_cast<real_t>(count);
        mean_abs_q += (abs_q - mean_abs_q) / static_cast<real_t>(count);
        const real_t radius = real_sqrt(static_cast<real_t>(ball.pos * ball.pos));
        mean_radius += (radius - mean_radius) / static_cast<real_t>(count);
        sliding_count += sliding ? 1 : 0;
        if (v < minimum_v) minimum_v = v;
        if (v > maximum_v) maximum_v = v;
        if (completed < split_step) {
            ++first_count;
            first_mean_v += (v - first_mean_v) / static_cast<real_t>(first_count);
            first_mean_phase_rate += (phase_rate - first_mean_phase_rate) / static_cast<real_t>(first_count);
        } else {
            ++second_count;
            second_mean_v += (v - second_mean_v) / static_cast<real_t>(second_count);
            second_mean_phase_rate += (phase_rate - second_mean_phase_rate) / static_cast<real_t>(second_count);
        }
    }
    if (status == 0) {
        real_t final_v, final_omega, final_phase_rate, final_abs_slip, final_abs_q;
        if (count < 2 || first_count == 0 || second_count == 0 ||
            !observables(ball, params, final_v, final_omega, final_phase_rate, final_abs_slip, final_abs_q)) {
            status = -2;
        } else {
            metrics[0] = mean_v;
            metrics[1] = real_sqrt(m2_v / static_cast<real_t>(count - 1));
            metrics[2] = mean_omega;
            metrics[3] = mean_phase_rate;
            metrics[4] = real_sqrt(m2_phase_rate / static_cast<real_t>(count - 1));
            metrics[5] = mean_abs_slip;
            metrics[6] = static_cast<real_t>(sliding_count) / static_cast<real_t>(count);
            metrics[7] = mean_abs_q;
            metrics[8] = final_v;
            metrics[9] = final_omega;
            metrics[10] = final_phase_rate;
            metrics[11] = final_abs_q;
            metrics[12] = second_mean_v - first_mean_v;
            metrics[13] = second_mean_phase_rate - first_mean_phase_rate;
            metrics[14] = minimum_v;
            metrics[15] = maximum_v;
            metrics[16] = static_cast<real_t>(count);
            metrics[17] = static_cast<real_t>(global_minimum_step);
            metrics[18] = velocity_recovered ? REAL_LITERAL(1.0) : REAL_LITERAL(0.0);
            metrics[19] = mean_radius;
        }
    }
    output_state[0] = static_cast<real_t>(ball.down_disk_angle);
    output_state[1] = static_cast<real_t>(ball.pos.x);
    output_state[2] = static_cast<real_t>(ball.pos.y);
    output_state[3] = static_cast<real_t>(ball.speed.x);
    output_state[4] = static_cast<real_t>(ball.speed.y);
    output_state[5] = static_cast<real_t>(ball.attitude.real);
    output_state[6] = static_cast<real_t>(ball.attitude.imagine.x);
    output_state[7] = static_cast<real_t>(ball.attitude.imagine.y);
    output_state[8] = static_cast<real_t>(ball.attitude.imagine.z);
    output_state[9] = static_cast<real_t>(ball.omega.x);
    output_state[10] = static_cast<real_t>(ball.omega.y);
    output_state[11] = static_cast<real_t>(ball.omega.z);
    output_state[12] = static_cast<real_t>(ball.q.x);
    output_state[13] = static_cast<real_t>(ball.q.y);
    output_state[14] = static_cast<real_t>(ball.q.z);
    return status;
}
} // namespace initial_scan
