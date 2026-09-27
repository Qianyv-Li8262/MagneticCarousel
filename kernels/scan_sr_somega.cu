// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

namespace {

constexpr int kMetricCount = 20;

DEVICE INLINE void update_mean_variance(const real_t value, const int count, real_t &mean, real_t &m2)
{
    const real_t delta = value - mean;
    mean += delta / static_cast<real_t>(count);
    m2 += delta * (value - mean);
}

DEVICE INLINE bool finite_state(const State &s)
{
    return real_isfinite(static_cast<real_t>(s.down_disk_angle)) &&
           real_isfinite(static_cast<real_t>(s.pos.x)) && real_isfinite(static_cast<real_t>(s.pos.y)) &&
           real_isfinite(static_cast<real_t>(s.speed.x)) && real_isfinite(static_cast<real_t>(s.speed.y)) &&
           real_isfinite(static_cast<real_t>(s.attitude.real)) &&
           real_isfinite(static_cast<real_t>(s.attitude.imagine.x)) &&
           real_isfinite(static_cast<real_t>(s.attitude.imagine.y)) &&
           real_isfinite(static_cast<real_t>(s.attitude.imagine.z)) &&
           real_isfinite(static_cast<real_t>(s.omega.x)) &&
           real_isfinite(static_cast<real_t>(s.omega.y)) &&
           real_isfinite(static_cast<real_t>(s.omega.z)) &&
           real_isfinite(static_cast<real_t>(s.q.x)) &&
           real_isfinite(static_cast<real_t>(s.q.y)) &&
           real_isfinite(static_cast<real_t>(s.q.z));
}

// The first Fourier harmonic has azimuthal order m=N/2. Its phase in the
// local radial/tangential frame advances at m*(1-phi_dot) relative to the
// rotating disk. The ball's spin about the radial axis subtracts omega_r.
// Divide by m to express the result in the same phase-time units as 1D:
//     phase_rate = 1 - phi_dot - omega_r/m.
// At radius r=m this reduces to the 1D 1-v-omega definition, with
// v=v_tangent/m and omega=omega_r/m.
DEVICE INLINE bool observables(const State &s, const SolverParams &params,
                               real_t &v, real_t &omega, real_t &phase_rate,
                               real_t &abs_slip, real_t &abs_q)
{
    const real_t x = static_cast<real_t>(s.pos.x);
    const real_t y = static_cast<real_t>(s.pos.y);
    const real_t vx = static_cast<real_t>(s.speed.x);
    const real_t vy = static_cast<real_t>(s.speed.y);
    const real_t wx = static_cast<real_t>(s.omega.x);
    const real_t wy = static_cast<real_t>(s.omega.y);
    const real_t radius2 = x * x + y * y;
    if (!real_isfinite(radius2) || radius2 <= REAL_LITERAL(0.0))
        return false;
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

} // namespace

// One CUDA thread advances one point of the (sr, somega) parameter grid.
// In Fourier-spline mode the CMake-generated field LUT is copied once per CTA into shared memory;
// with the current 8-magnet / 7-mode / 128-node FP64 configuration this is
// 28 KiB per CTA, leaving room for three CTAs per V100S SM.
// The first 19 output_metrics columns match 1D phase_behavior.cu;
// column 19 is the terminal-window mean orbital radius.
// Column 3 is the terminal-window mean phase derivative, column 4 its sample
// standard deviation, and column 13 the second-half minus first-half mean.
extern "C" __launch_bounds__(128, 3) GLOBAL void scan_sr_somega_cuda(
    const real_t sr_lower, const real_t sr_upper,
    const real_t somega_lower, const real_t somega_upper,
    const int k, const real_t dt, const int steps,
    const int window_steps, const int sample_stride,
    const real_t reference_alpha, const real_t reference_delta,
    const real_t reference_gamma_g, const real_t reference_gamma,
    const real_t reference_gamma_m, const LUTNode *lut_global,
    real_t *output_state, int *output_status, real_t *output_metrics)
{
#if FIELD_EVAL_MODE == 1
    constexpr int lut_count = kFieldFourierModeCount * kSplineNodes;
    __shared__ LUTNode lut_shared[lut_count];

    const int thread_in_block = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int threads_in_block = static_cast<int>(blockDim.x * blockDim.y);
    for (int i = thread_in_block; i < lut_count; i += threads_in_block)
        lut_shared[i] = lut_global[i];
    __syncthreads();
    const LUTNode *field_lut = lut_shared;
#else
    const LUTNode *field_lut = nullptr;
#endif

    const int sr_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int somega_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (k <= 0 || sr_index >= k || somega_index >= k)
        return;

    const int output_index = somega_index * k + sr_index;
    output_status[output_index] = -1;
    const int state_offset = output_index * 15;
    for (int i = 0; i < 15; ++i)
        output_state[state_offset + i] = REAL_LITERAL(0.0);

    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || steps <= 0 ||
        window_steps <= 1 || sample_stride <= 0 ||
        !real_isfinite(sr_lower) || !real_isfinite(sr_upper) ||
        !real_isfinite(somega_lower) || !real_isfinite(somega_upper)) {
        output_status[output_index] = -2;
        return;
    }

    const real_t sr_fraction = k == 1 ? REAL_LITERAL(0.0)
                                       : static_cast<real_t>(sr_index) / static_cast<real_t>(k - 1);
    const real_t somega_fraction = k == 1 ? REAL_LITERAL(0.0)
                                           : static_cast<real_t>(somega_index) / static_cast<real_t>(k - 1);
    const real_t sr = sr_lower + (sr_upper - sr_lower) * sr_fraction;
    const real_t somega = somega_lower + (somega_upper - somega_lower) * somega_fraction;
    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) ||
        !real_isfinite(somega) || somega <= REAL_LITERAL(0.0)) {
        output_status[output_index] = -2;
        return;
    }

    // Keep the 8-magnet reference physics fixed while scaling radius and
    // drive frequency, matching scaled_model_parameters() in the CPU runner.
    const real_t inv_sr = REAL_LITERAL(1.0) / sr;
    const real_t inv_somega2 = REAL_LITERAL(1.0) / (somega * somega);
    const real_t magnet_time_scale = static_cast<real_t>(magnet_number) / REAL_LITERAL(8.0);
    const real_t magnet_time_scale2 = magnet_time_scale * magnet_time_scale;
    const SolverParams params = {
        reference_alpha * sr,
        reference_delta * inv_sr,
        reference_gamma_g * inv_somega2 * magnet_time_scale2,
        reference_gamma * inv_somega2 * magnet_time_scale2,
        reference_gamma_m * inv_sr * inv_sr * inv_somega2 * magnet_time_scale2,
    };
    if (!real_isfinite(params.alpha) || params.alpha <= REAL_LITERAL(0.0) ||
        !real_isfinite(params.delta) || params.delta < REAL_LITERAL(0.0) ||
        !real_isfinite(params.gamma_g) || !real_isfinite(params.gamma) ||
        !real_isfinite(params.gamma_m)) {
        output_status[output_index] = -2;
        return;
    }

    // Match make_initial_state() in animate_one_trajectory.py: the sphere
    // starts at the magnet radius with identity attitude and otherwise rests.
    State ball{};
    ball.pos.x = static_cast<accumulator_t>(magnet_number / 2);
    ball.attitude.real = static_cast<accumulator_t>(REAL_LITERAL(1.0));
    bool sliding = false;
    Vec2 sliding_dir{REAL_LITERAL(0.0), REAL_LITERAL(0.0)};
    bool field_cache_valid = false;
    FieldResult field_cache{};
    FieldCache lut_cache{};

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
    real_t mean_radius = REAL_LITERAL(0.0);
    real_t first_mean_v = REAL_LITERAL(0.0);
    real_t second_mean_v = REAL_LITERAL(0.0);
    real_t first_mean_phase_rate = REAL_LITERAL(0.0);
    real_t second_mean_phase_rate = REAL_LITERAL(0.0);
    real_t minimum_v = REAL_LITERAL(3.402823466e38);
    real_t maximum_v = -REAL_LITERAL(3.402823466e38);
    real_t global_minimum_v = REAL_LITERAL(0.0);
    int global_minimum_step = 0;
    bool velocity_recovered = false;

    int status = 0;
    for (int step = 0; step < steps; ++step) {
        const AdvanceStepResult result = fwd1step(dt, params, ball, sliding, sliding_dir,
                                                   field_cache_valid, &lut_cache, field_lut,
                                                   &field_cache);
        if (result != AdvanceStepResult::kStepOk) {
            status = static_cast<int>(result) + 1;
            break;
        }
        if (!finite_state(ball)) {
            status = static_cast<int>(AdvanceStepResult::kStepNaNOrInf) + 1;
            break;
        }

        real_t v, omega, phase_rate, abs_slip, abs_q;
        if (!observables(ball, params, v, omega, phase_rate, abs_slip, abs_q)) {
            status = static_cast<int>(AdvanceStepResult::kStepNaNOrInf) + 1;
            break;
        }
        const int completed = step + 1;
        if (v < global_minimum_v) {
            global_minimum_v = v;
            global_minimum_step = completed;
            velocity_recovered = false;
        } else if (v > global_minimum_v) {
            velocity_recovered = true;
        }
        if (completed < window_start || (completed - window_start) % sample_stride != 0)
            continue;

        ++count;
        update_mean_variance(v, count, mean_v, m2_v);
        update_mean_variance(phase_rate, count, mean_phase_rate, m2_phase_rate);
        mean_omega += (omega - mean_omega) / static_cast<real_t>(count);
        mean_abs_slip += (abs_slip - mean_abs_slip) / static_cast<real_t>(count);
        mean_abs_q += (abs_q - mean_abs_q) / static_cast<real_t>(count);
        const real_t radius = real_sqrt(static_cast<real_t>(ball.pos * ball.pos));
        mean_radius += (radius - mean_radius) / static_cast<real_t>(count);
        sliding_count += sliding ? 1 : 0;
        if (v < minimum_v)
            minimum_v = v;
        if (v > maximum_v)
            maximum_v = v;
        if (completed < split_step) {
            ++first_count;
            first_mean_v += (v - first_mean_v) / static_cast<real_t>(first_count);
            first_mean_phase_rate +=
                (phase_rate - first_mean_phase_rate) / static_cast<real_t>(first_count);
        } else {
            ++second_count;
            second_mean_v += (v - second_mean_v) / static_cast<real_t>(second_count);
            second_mean_phase_rate +=
                (phase_rate - second_mean_phase_rate) / static_cast<real_t>(second_count);
        }
    }

    if (status == 0) {
        real_t final_v, final_omega, final_phase_rate, final_abs_slip, final_abs_q;
        if (count < 2 || first_count == 0 || second_count == 0 ||
            !observables(ball, params, final_v, final_omega, final_phase_rate,
                         final_abs_slip, final_abs_q)) {
            status = -2;
        } else {
            real_t *metrics = output_metrics + kMetricCount * output_index;
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

    output_state[state_offset + 0] = static_cast<real_t>(ball.down_disk_angle);
    output_state[state_offset + 1] = static_cast<real_t>(ball.pos.x);
    output_state[state_offset + 2] = static_cast<real_t>(ball.pos.y);
    output_state[state_offset + 3] = static_cast<real_t>(ball.speed.x);
    output_state[state_offset + 4] = static_cast<real_t>(ball.speed.y);
    output_state[state_offset + 5] = static_cast<real_t>(ball.attitude.real);
    output_state[state_offset + 6] = static_cast<real_t>(ball.attitude.imagine.x);
    output_state[state_offset + 7] = static_cast<real_t>(ball.attitude.imagine.y);
    output_state[state_offset + 8] = static_cast<real_t>(ball.attitude.imagine.z);
    output_state[state_offset + 9] = static_cast<real_t>(ball.omega.x);
    output_state[state_offset + 10] = static_cast<real_t>(ball.omega.y);
    output_state[state_offset + 11] = static_cast<real_t>(ball.omega.z);
    output_state[state_offset + 12] = static_cast<real_t>(ball.q.x);
    output_state[state_offset + 13] = static_cast<real_t>(ball.q.y);
    output_state[state_offset + 14] = static_cast<real_t>(ball.q.z);
    output_status[output_index] = status;
}
