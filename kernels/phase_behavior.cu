#include "1dmodel-hysteresis.hpp"

#include <cuda_runtime.h>

namespace
{

enum phase_status {
    kPhaseOk = 0,
    kPhaseNegativeNormal = 1,
    kPhaseUndefinedSlip = 2,
    kPhaseNonFinite = 3,
    kPhaseInvalidInput = 4,
};

DEVICE INLINE bool phase_finite_state(const state &ball)
{
    return real_isfinite(ball.t) && real_isfinite(ball.x) && real_isfinite(ball.xdot) && real_isfinite(ball.theta) &&
           real_isfinite(ball.theta_dot) && real_isfinite(ball.phase) && real_isfinite(ball.q.x) && real_isfinite(ball.q.y);
}

DEVICE INLINE solver_params phase_scaled_params(const real_t sr, const real_t somega, const solver_params &reference)
{
    const real_t inverse_sr = REAL_LITERAL(1.0) / sr;
    const real_t inverse_somega_squared = REAL_LITERAL(1.0) / (somega * somega);
    return {
        reference.alpha * sr,
        reference.delta * inverse_sr,
        reference.gamma_g * inverse_somega_squared,
        reference.gamma * inverse_somega_squared,
        reference.gamma_m * inverse_sr * inverse_sr * inverse_somega_squared,
    };
}

DEVICE INLINE int phase_initialize(state &ball, const solver_params &params, bool &sliding, real_t &slip_sign)
{
    const real_t slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    sliding = real_fabs(slip_velocity) > REAL_LITERAL(1.0e-10);
    slip_sign = sliding ? signum(slip_velocity) : REAL_LITERAL(0.0);
    if (sliding)
        return kPhaseOk;

    const field_evaluation field_lab = evaluate_field(ball.phase);
    const real_t body_angle = -ball.theta;
    const vec2 h_ext = rotate(field_lab.field, body_angle);
    const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
    const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
    const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
    const real_t E = (REAL_LITERAL(1.0) - C) / ((MU_R + REAL_LITERAL(2.0)) * (REAL_LITERAL(1.0) - XI_JA));
    const real_t normal_force =
        params.gamma_g -
        params.gamma * (A * (h_ext * h_ext_height_derivative) - E * (ball.q * h_ext_height_derivative));
    if (!real_isfinite(normal_force) || normal_force < REAL_LITERAL(0.0))
        return kPhaseNegativeNormal;

    const real_t magnetic_torque = -params.gamma_m * E * cross(ball.q, h_ext);
    const real_t magnetic_force = ETA * params.alpha * params.alpha * params.gamma_m *
                                 (E * (ball.q * h_ext_phase_derivative) -
                                  A * (h_ext * h_ext_phase_derivative));
    const real_t theta_acceleration = ETA / (REAL_LITERAL(1.0) + ETA) * magnetic_torque -
                                     magnetic_force / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
                                     params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * ball.theta_dot *
                                         real_fabs(ball.theta_dot);
    if (!real_isfinite(magnetic_torque) || !real_isfinite(magnetic_force) || !real_isfinite(theta_acceleration))
        return kPhaseNonFinite;

    if (real_fabs(theta_acceleration - magnetic_torque) * ETA * params.alpha > MU * normal_force) {
        sliding = true;
        slip_sign = -signum(theta_acceleration - magnetic_torque);
        if (slip_sign == REAL_LITERAL(0.0))
            return kPhaseUndefinedSlip;
    } else {
        project_pure_rolling(ball, params);
    }
    return kPhaseOk;
}

DEVICE INLINE void update_mean_variance(const real_t value, const int count, real_t &mean, real_t &m2)
{
    const real_t delta = value - mean;
    mean += delta / static_cast<real_t>(count);
    m2 += delta * (value - mean);
}

// Metrics layout:
//  0 mean(v), 1 std(v), 2 mean(omega), 3 mean(phase rate),
//  4 std(phase rate), 5 mean(abs(slip)), 6 sliding fraction,
//  7 mean(abs(q)), 8 final(v), 9 final(omega), 10 final(phase rate),
// 11 final(abs(q)), 12 second-half minus first-half mean(v),
// 13 second-half minus first-half mean(phase rate), 14 min(v), 15 max(v),
// 16 sample count, 17 completed step of the global minimum(v),
// 18 whether v rose after that minimum.
DEVICE INLINE int phase_analyze(state &ball, const real_t dt, const int steps, const int window_steps,
                                const int sample_stride, const solver_params &params, real_t *metrics)
{
    bool sliding = false;
    real_t slip_sign = REAL_LITERAL(0.0);
    int status = phase_initialize(ball, params, sliding, slip_sign);
    if (status != kPhaseOk)
        return status;

    const int window_start = steps > window_steps ? steps - window_steps : 0;
    const int split_step = window_start + (steps - window_start) / 2;
    int count = 0;
    int first_count = 0;
    int second_count = 0;
    int sliding_count = 0;
    real_t mean_v = REAL_LITERAL(0.0);
    real_t m2_v = REAL_LITERAL(0.0);
    real_t mean_omega = REAL_LITERAL(0.0);
    real_t mean_phase_rate = REAL_LITERAL(0.0);
    real_t m2_phase_rate = REAL_LITERAL(0.0);
    real_t mean_abs_slip = REAL_LITERAL(0.0);
    real_t mean_abs_q = REAL_LITERAL(0.0);
    real_t first_mean_v = REAL_LITERAL(0.0);
    real_t second_mean_v = REAL_LITERAL(0.0);
    real_t first_mean_phase_rate = REAL_LITERAL(0.0);
    real_t second_mean_phase_rate = REAL_LITERAL(0.0);
    real_t minimum_v = REAL_LITERAL(3.402823466e38);
    real_t maximum_v = -REAL_LITERAL(3.402823466e38);
    real_t global_minimum_v = ball.xdot;
    int global_minimum_step = 0;
    bool velocity_recovered = false;

    for (int step = 0; step < steps; ++step) {
        const int flag = fwd1step(ball, dt, params, sliding, slip_sign);
        if (flag != 0)
            return flag;
        if (!phase_finite_state(ball))
            return kPhaseNonFinite;

        const int completed = step + 1;
        if (ball.xdot < global_minimum_v) {
            global_minimum_v = ball.xdot;
            global_minimum_step = completed;
            velocity_recovered = false;
        } else if (ball.xdot > global_minimum_v) {
            velocity_recovered = true;
        }
        if (completed < window_start || (completed - window_start) % sample_stride != 0)
            continue;

        const real_t phase_rate = REAL_LITERAL(1.0) - ball.xdot - ball.theta_dot;
        const real_t slip = ball.xdot + params.alpha * ball.theta_dot;
        ++count;
        update_mean_variance(ball.xdot, count, mean_v, m2_v);
        update_mean_variance(phase_rate, count, mean_phase_rate, m2_phase_rate);
        mean_omega += (ball.theta_dot - mean_omega) / static_cast<real_t>(count);
        mean_abs_slip += (real_fabs(slip) - mean_abs_slip) / static_cast<real_t>(count);
        mean_abs_q += (length(ball.q) - mean_abs_q) / static_cast<real_t>(count);
        sliding_count += sliding ? 1 : 0;
        minimum_v = ball.xdot < minimum_v ? ball.xdot : minimum_v;
        maximum_v = ball.xdot > maximum_v ? ball.xdot : maximum_v;

        if (completed < split_step) {
            ++first_count;
            first_mean_v += (ball.xdot - first_mean_v) / static_cast<real_t>(first_count);
            first_mean_phase_rate +=
                (phase_rate - first_mean_phase_rate) / static_cast<real_t>(first_count);
        } else {
            ++second_count;
            second_mean_v += (ball.xdot - second_mean_v) / static_cast<real_t>(second_count);
            second_mean_phase_rate +=
                (phase_rate - second_mean_phase_rate) / static_cast<real_t>(second_count);
        }
    }

    if (count < 2 || first_count == 0 || second_count == 0)
        return kPhaseInvalidInput;
    metrics[0] = mean_v;
    metrics[1] = real_sqrt(m2_v / static_cast<real_t>(count - 1));
    metrics[2] = mean_omega;
    metrics[3] = mean_phase_rate;
    metrics[4] = real_sqrt(m2_phase_rate / static_cast<real_t>(count - 1));
    metrics[5] = mean_abs_slip;
    metrics[6] = static_cast<real_t>(sliding_count) / static_cast<real_t>(count);
    metrics[7] = mean_abs_q;
    metrics[8] = ball.xdot;
    metrics[9] = ball.theta_dot;
    metrics[10] = REAL_LITERAL(1.0) - ball.xdot - ball.theta_dot;
    metrics[11] = length(ball.q);
    metrics[12] = second_mean_v - first_mean_v;
    metrics[13] = second_mean_phase_rate - first_mean_phase_rate;
    metrics[14] = minimum_v;
    metrics[15] = maximum_v;
    metrics[16] = static_cast<real_t>(count);
    metrics[17] = static_cast<real_t>(global_minimum_step);
    metrics[18] = velocity_recovered ? REAL_LITERAL(1.0) : REAL_LITERAL(0.0);
    return kPhaseOk;
}

} // namespace

extern "C" __launch_bounds__(128, 4) GLOBAL  void AnalyzeParameterGrid(const real_t *sr_values, const real_t *somega_values, const int k,
                                             const real_t dt, const int steps, const int window_steps,
                                             const int sample_stride, const real_t reference_alpha,
                                             const real_t reference_delta, const real_t reference_gamma_g,
                                             const real_t reference_gamma, const real_t reference_gamma_m,
                                             int *output_status, real_t *output_metrics)
{
    const int sr_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int somega_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (sr_index >= k || somega_index >= k)
        return;
    const int index = somega_index * k + sr_index;
    output_status[index] = kPhaseInvalidInput;
    const real_t sr = sr_values[sr_index];
    const real_t somega = somega_values[somega_index];
    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) || somega <= REAL_LITERAL(0.0) || dt <= REAL_LITERAL(0.0) || steps <= 0 ||
        window_steps <= 1 || sample_stride <= 0)
        return;
    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    const solver_params params = phase_scaled_params(sr, somega, reference);
    state ball = {REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)}};
    output_status[index] =
        phase_analyze(ball, dt, steps, window_steps, sample_stride, params, output_metrics + 19 * index);
}

extern "C" GLOBAL void AnalyzeInitialGrid(
    const real_t sr, const real_t somega, const real_t *v0_values, const real_t *s0_values, const int k,
    const real_t dt, const int steps, const int window_steps, const int sample_stride,
    const real_t reference_alpha, const real_t reference_delta, const real_t reference_gamma_g,
    const real_t reference_gamma, const real_t reference_gamma_m, int *output_status, real_t *output_metrics)
{
    const int v0_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int s0_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (v0_index >= k || s0_index >= k)
        return;
    const int index = s0_index * k + v0_index;
    output_status[index] = kPhaseInvalidInput;
    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) || somega <= REAL_LITERAL(0.0) || dt <= REAL_LITERAL(0.0) || steps <= 0 ||
        window_steps <= 1 || sample_stride <= 0)
        return;
    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    const solver_params params = phase_scaled_params(sr, somega, reference);
    const real_t v0 = v0_values[v0_index];
    const real_t s0 = s0_values[s0_index];
    const real_t omega0 = (s0 - v0) / params.alpha;
    if (!real_isfinite(v0) || !real_isfinite(s0) || !real_isfinite(omega0))
        return;
    state ball = {REAL_LITERAL(0.0), REAL_LITERAL(0.0), v0, REAL_LITERAL(0.0), omega0, REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)}};
    output_status[index] =
        phase_analyze(ball, dt, steps, window_steps, sample_stride, params, output_metrics + 19 * index);
}

// One-dimensional high-resolution cut through the (v0, S0) basin map.
// Each thread evolves one S0 value at a shared v0 and writes the same 19
// terminal-window metrics as AnalyzeInitialGrid.
extern "C" GLOBAL void AnalyzeInitialLine(
    const real_t sr, const real_t somega, const real_t v0, const real_t *s0_values, const int count,
    const real_t dt, const int steps, const int window_steps, const int sample_stride,
    const real_t reference_alpha, const real_t reference_delta, const real_t reference_gamma_g,
    const real_t reference_gamma, const real_t reference_gamma_m, int *output_status,
    real_t *output_metrics)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    output_status[index] = kPhaseInvalidInput;
    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) || somega <= REAL_LITERAL(0.0) || !real_isfinite(v0) ||
        dt <= REAL_LITERAL(0.0) || steps <= 0 || window_steps <= 1 || sample_stride <= 0)
        return;

    const real_t s0 = s0_values[index];
    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    const solver_params params = phase_scaled_params(sr, somega, reference);
    const real_t omega0 = (s0 - v0) / params.alpha;
    if (!real_isfinite(s0) || !real_isfinite(omega0))
        return;

    state ball = {REAL_LITERAL(0.0), REAL_LITERAL(0.0), v0, REAL_LITERAL(0.0), omega0, REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)}};
    output_status[index] =
        phase_analyze(ball, dt, steps, window_steps, sample_stride, params, output_metrics + 19 * index);
}
