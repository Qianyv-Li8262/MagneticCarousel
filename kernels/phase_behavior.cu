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
    return isfinite(ball.t) && isfinite(ball.x) && isfinite(ball.xdot) && isfinite(ball.theta) &&
           isfinite(ball.theta_dot) && isfinite(ball.phase) && isfinite(ball.q.x) && isfinite(ball.q.y);
}

DEVICE INLINE solver_params phase_scaled_params(const float sr, const float somega, const solver_params &reference)
{
    const float inverse_sr = 1.0f / sr;
    const float inverse_somega_squared = 1.0f / (somega * somega);
    return {
        reference.alpha * sr,
        reference.delta * inverse_sr,
        reference.gamma_g * inverse_somega_squared,
        reference.gamma * inverse_somega_squared,
        reference.gamma_m * inverse_sr * inverse_sr * inverse_somega_squared,
    };
}

DEVICE INLINE int phase_initialize(state &ball, const solver_params &params, bool &sliding, float &slip_sign)
{
    const float slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    sliding = fabsf(slip_velocity) > 1.0e-10f;
    slip_sign = sliding ? signum(slip_velocity) : 0.0f;
    if (sliding)
        return kPhaseOk;

    const vec2 h_ext = {H_EXT * cosf(ball.phase), H_EXT * sinf(ball.phase)};
    const float normal_force =
        params.gamma_g - params.gamma / (2.0f + MU_R) *
                             (-(MU_R - 1.0f) * H_EXT * H_EXT +
                              (1.0f - C) / (1.0f - XI_JA) * (ball.q * h_ext));
    if (!isfinite(normal_force) || normal_force < 0.0f)
        return kPhaseNegativeNormal;

    const float magnetic_torque =
        -params.gamma_m * (1.0f - C) / ((1.0f - XI_JA) * (MU_R + 2.0f)) * cross(ball.q, h_ext);
    const float theta_acceleration =
        ETA / (1.0f + ETA) * (1.0f - params.alpha) * magnetic_torque -
        params.alpha / (1.0f + ETA) * params.delta * ball.theta_dot * fabsf(ball.theta_dot);
    if (!isfinite(magnetic_torque) || !isfinite(theta_acceleration))
        return kPhaseNonFinite;

    if (fabsf(theta_acceleration - magnetic_torque) * ETA * params.alpha > MU * normal_force) {
        sliding = true;
        slip_sign = -signum(theta_acceleration - magnetic_torque);
        if (slip_sign == 0.0f)
            return kPhaseUndefinedSlip;
    } else {
        project_pure_rolling(ball, params);
    }
    return kPhaseOk;
}

DEVICE INLINE void update_mean_variance(const float value, const int count, float &mean, float &m2)
{
    const float delta = value - mean;
    mean += delta / static_cast<float>(count);
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
DEVICE INLINE int phase_analyze(state &ball, const float dt, const int steps, const int window_steps,
                                const int sample_stride, const solver_params &params, float *metrics)
{
    bool sliding = false;
    float slip_sign = 0.0f;
    int status = phase_initialize(ball, params, sliding, slip_sign);
    if (status != kPhaseOk)
        return status;

    const int window_start = steps > window_steps ? steps - window_steps : 0;
    const int split_step = window_start + (steps - window_start) / 2;
    int count = 0;
    int first_count = 0;
    int second_count = 0;
    int sliding_count = 0;
    float mean_v = 0.0f;
    float m2_v = 0.0f;
    float mean_omega = 0.0f;
    float mean_phase_rate = 0.0f;
    float m2_phase_rate = 0.0f;
    float mean_abs_slip = 0.0f;
    float mean_abs_q = 0.0f;
    float first_mean_v = 0.0f;
    float second_mean_v = 0.0f;
    float first_mean_phase_rate = 0.0f;
    float second_mean_phase_rate = 0.0f;
    float minimum_v = 3.402823466e38f;
    float maximum_v = -3.402823466e38f;
    float global_minimum_v = ball.xdot;
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

        const float phase_rate = 1.0f - ball.xdot - ball.theta_dot;
        const float slip = ball.xdot + params.alpha * ball.theta_dot;
        ++count;
        update_mean_variance(ball.xdot, count, mean_v, m2_v);
        update_mean_variance(phase_rate, count, mean_phase_rate, m2_phase_rate);
        mean_omega += (ball.theta_dot - mean_omega) / static_cast<float>(count);
        mean_abs_slip += (fabsf(slip) - mean_abs_slip) / static_cast<float>(count);
        mean_abs_q += (length(ball.q) - mean_abs_q) / static_cast<float>(count);
        sliding_count += sliding ? 1 : 0;
        minimum_v = fminf(minimum_v, ball.xdot);
        maximum_v = fmaxf(maximum_v, ball.xdot);

        if (completed < split_step) {
            ++first_count;
            first_mean_v += (ball.xdot - first_mean_v) / static_cast<float>(first_count);
            first_mean_phase_rate +=
                (phase_rate - first_mean_phase_rate) / static_cast<float>(first_count);
        } else {
            ++second_count;
            second_mean_v += (ball.xdot - second_mean_v) / static_cast<float>(second_count);
            second_mean_phase_rate +=
                (phase_rate - second_mean_phase_rate) / static_cast<float>(second_count);
        }
    }

    if (count < 2 || first_count == 0 || second_count == 0)
        return kPhaseInvalidInput;
    metrics[0] = mean_v;
    metrics[1] = sqrtf(m2_v / static_cast<float>(count - 1));
    metrics[2] = mean_omega;
    metrics[3] = mean_phase_rate;
    metrics[4] = sqrtf(m2_phase_rate / static_cast<float>(count - 1));
    metrics[5] = mean_abs_slip;
    metrics[6] = static_cast<float>(sliding_count) / static_cast<float>(count);
    metrics[7] = mean_abs_q;
    metrics[8] = ball.xdot;
    metrics[9] = ball.theta_dot;
    metrics[10] = 1.0f - ball.xdot - ball.theta_dot;
    metrics[11] = length(ball.q);
    metrics[12] = second_mean_v - first_mean_v;
    metrics[13] = second_mean_phase_rate - first_mean_phase_rate;
    metrics[14] = minimum_v;
    metrics[15] = maximum_v;
    metrics[16] = static_cast<float>(count);
    metrics[17] = static_cast<float>(global_minimum_step);
    metrics[18] = velocity_recovered ? 1.0f : 0.0f;
    return kPhaseOk;
}

} // namespace

extern "C" GLOBAL void AnalyzeParameterGrid(const float *sr_values, const float *somega_values, const int k,
                                             const float dt, const int steps, const int window_steps,
                                             const int sample_stride, const float reference_alpha,
                                             const float reference_delta, const float reference_gamma_g,
                                             const float reference_gamma, const float reference_gamma_m,
                                             int *output_status, float *output_metrics)
{
    const int sr_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int somega_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (sr_index >= k || somega_index >= k)
        return;
    const int index = somega_index * k + sr_index;
    output_status[index] = kPhaseInvalidInput;
    const float sr = sr_values[sr_index];
    const float somega = somega_values[somega_index];
    if (!isfinite(sr) || sr <= 0.0f || !isfinite(somega) || somega <= 0.0f || dt <= 0.0f || steps <= 0 ||
        window_steps <= 1 || sample_stride <= 0)
        return;
    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    const solver_params params = phase_scaled_params(sr, somega, reference);
    state ball = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {0.0f, 0.0f}};
    output_status[index] =
        phase_analyze(ball, dt, steps, window_steps, sample_stride, params, output_metrics + 19 * index);
}

extern "C" GLOBAL void AnalyzeInitialGrid(
    const float sr, const float somega, const float *v0_values, const float *s0_values, const int k,
    const float dt, const int steps, const int window_steps, const int sample_stride,
    const float reference_alpha, const float reference_delta, const float reference_gamma_g,
    const float reference_gamma, const float reference_gamma_m, int *output_status, float *output_metrics)
{
    const int v0_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int s0_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (v0_index >= k || s0_index >= k)
        return;
    const int index = s0_index * k + v0_index;
    output_status[index] = kPhaseInvalidInput;
    if (!isfinite(sr) || sr <= 0.0f || !isfinite(somega) || somega <= 0.0f || dt <= 0.0f || steps <= 0 ||
        window_steps <= 1 || sample_stride <= 0)
        return;
    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    const solver_params params = phase_scaled_params(sr, somega, reference);
    const float v0 = v0_values[v0_index];
    const float s0 = s0_values[s0_index];
    const float omega0 = (s0 - v0) / params.alpha;
    if (!isfinite(v0) || !isfinite(s0) || !isfinite(omega0))
        return;
    state ball = {0.0f, 0.0f, v0, 0.0f, omega0, 0.0f, {0.0f, 0.0f}};
    output_status[index] =
        phase_analyze(ball, dt, steps, window_steps, sample_stride, params, output_metrics + 19 * index);
}
