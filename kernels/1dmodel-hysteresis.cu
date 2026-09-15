#include "1dmodel-hysteresis.hpp"

#include <cuda_runtime.h>

namespace
{

// The two grid coordinates are ratios relative to a runtime-supplied reference
// point:
//
//     sr     = radius / reference_radius
//     somega = drive_omega / reference_drive_omega
//
// The material, field wavenumber, field amplitude, and drag coefficient are
// held fixed while the radius and drive angular speed are varied.

enum gpu_evolve_status {
    kGpuEvolveOk = 0,
    kGpuEvolveInvalidInput = 1,
    kGpuEvolveStepFailed = 2,
    kGpuEvolveNonFiniteState = 3,
};

DEVICE INLINE bool finite_state(const state &ball)
{
    return isfinite(ball.t) && isfinite(ball.x) && isfinite(ball.xdot) && isfinite(ball.theta) &&
           isfinite(ball.theta_dot) && isfinite(ball.phase) && isfinite(ball.q.x) && isfinite(ball.q.y);
}

DEVICE INLINE bool finite_params(const solver_params &params)
{
    return isfinite(params.alpha) && params.alpha > 0.0f && isfinite(params.delta) && params.delta >= 0.0f &&
           isfinite(params.gamma_g) && isfinite(params.gamma) && isfinite(params.gamma_m);
}

DEVICE INLINE solver_params scaled_params(const float sr, const float somega, const solver_params &reference)
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

// GPU counterpart of the initialization in 1dmodel-hysteresis.cpp.  CUDA
// device code cannot throw C++ exceptions, so failures are returned as flags.
DEVICE INLINE int initialize_gpu_mode(state &ball, const solver_params &params, bool &sliding, float &slip_sign)
{
    const float slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    sliding = fabsf(slip_velocity) > 1.0e-10f;
    slip_sign = sliding ? signum(slip_velocity) : 0.0f;

    if (sliding)
        return kGpuEvolveOk;

    const field_evaluation field_lab = evaluate_field(ball.phase);
    const float body_angle = -ball.theta;
    const vec2 h_ext = rotate(field_lab.field, body_angle);
    const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
    const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
    const float A = (MU_R - 1.0f) / (MU_R + 2.0f);
    const float E = (1.0f - C) / ((MU_R + 2.0f) * (1.0f - XI_JA));
    const float normal_force =
        params.gamma_g -
        params.gamma * (A * (h_ext * h_ext_height_derivative) - E * (ball.q * h_ext_height_derivative));
    if (!isfinite(normal_force) || normal_force < 0.0f)
        return kGpuEvolveStepFailed;

    const float magnetic_torque = -params.gamma_m * E * cross(ball.q, h_ext);
    const float magnetic_force = ETA * params.alpha * params.alpha * params.gamma_m *
                                 (E * (ball.q * h_ext_phase_derivative) -
                                  A * (h_ext * h_ext_phase_derivative));
    const float theta_acceleration = ETA / (1.0f + ETA) * magnetic_torque -
                                     magnetic_force / ((1.0f + ETA) * params.alpha) -
                                     params.alpha / (1.0f + ETA) * params.delta * ball.theta_dot *
                                         fabsf(ball.theta_dot);
    if (!isfinite(magnetic_torque) || !isfinite(magnetic_force) || !isfinite(theta_acceleration))
        return kGpuEvolveNonFiniteState;

    if (fabsf(theta_acceleration - magnetic_torque) * ETA * params.alpha > MU * normal_force) {
        sliding = true;
        slip_sign = -signum(theta_acceleration - magnetic_torque);
        if (slip_sign == 0.0f)
            return kGpuEvolveStepFailed;
    } else {
        slip_sign = 0.0f;
        project_pure_rolling(ball, params);
    }

    return kGpuEvolveOk;
}

DEVICE INLINE void update_velocity_minimum(const state &ball, const int completed_step, bool &velocity_recovered,
                                           float &minimum_x, int &minimum_step, float &minimum_velocity)
{
    if (ball.xdot < minimum_velocity) {
        minimum_velocity = ball.xdot;
        minimum_x = ball.x;
        minimum_step = completed_step;
        velocity_recovered = false;
    } else if (ball.xdot > minimum_velocity) {
        velocity_recovered = true;
    }
}

// GPU analogue of the CPU evolve loop. It maintains sliding, slip_sign, and
// the signed minimum center-of-mass velocity across integration steps.
DEVICE INLINE int evolve_gpu(state &ball, const float dt, const float total_time, const solver_params &params,
                             bool &velocity_recovered, float &minimum_x, int &minimum_step, float &minimum_velocity)
{
    if (!isfinite(dt) || dt <= 0.0f || !isfinite(total_time) || total_time < 0.0f || !finite_params(params))
        return kGpuEvolveInvalidInput;

    const float step_count = floorf(total_time / dt);
    if (!isfinite(step_count) || step_count > 2147483000.0f)
        return kGpuEvolveInvalidInput;

    bool sliding = false;
    float slip_sign = 0.0f;
    int status = initialize_gpu_mode(ball, params, sliding, slip_sign);
    if (status != kGpuEvolveOk)
        return status;

    // The supplied initial state is step zero and participates in the minimum.
    velocity_recovered = false;
    minimum_x = ball.x;
    minimum_step = 0;
    minimum_velocity = ball.xdot;

    const int full_steps = static_cast<int>(step_count);
    for (int step = 0; step < full_steps; ++step) {
        if (fwd1step(ball, dt, params, sliding, slip_sign) != 0)
            return kGpuEvolveStepFailed;
        if (!finite_state(ball))
            return kGpuEvolveNonFiniteState;
        update_velocity_minimum(ball, step + 1, velocity_recovered, minimum_x, minimum_step, minimum_velocity);
    }

    // Do not silently discard a final fractional step when total_time is not
    // an integer multiple of dt.
    const float remainder = total_time - static_cast<float>(full_steps) * dt;
    if (remainder > 0.0f) {
        if (fwd1step(ball, remainder, params, sliding, slip_sign) != 0)
            return kGpuEvolveStepFailed;
        if (!finite_state(ball))
            return kGpuEvolveNonFiniteState;
        update_velocity_minimum(ball, full_steps + 1, velocity_recovered, minimum_x, minimum_step, minimum_velocity);
    }

    return kGpuEvolveOk;
}

DEVICE INLINE float interpolate_grid_value(const float lower, const float upper, const int index, const int count)
{
    if (count == 1)
        return lower;
    const float fraction = static_cast<float>(index) / static_cast<float>(count - 1);
    return lower + (upper - lower) * fraction;
}

} // namespace

// Two-dimensional grid layout:
//   x thread coordinate -> sr
//   y thread coordinate -> somega
//   flat index          -> somega_index * k + sr_index
//
// output_status is always written: 0 means success and 1 means failure. The
// other four arrays are written only after a complete successful evolution.
extern "C" GLOBAL void EvolveHysteresisGrid(const float sr_lower, const float sr_upper, const float somega_lower,
                                            const float somega_upper, const int k, const float dt,
                                            const float total_time, const float reference_alpha,
                                            const float reference_delta, const float reference_gamma_g,
                                            const float reference_gamma, const float reference_gamma_m,
                                            int *output_status, int *output_velocity_recovered, float *output_minimum_x,
                                            int *output_minimum_step, float *output_minimum_velocity)
{
    const int sr_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int somega_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (k <= 0 || sr_index >= k || somega_index >= k)
        return;

    const int output_index = somega_index * k + sr_index;
    output_status[output_index] = 1;

    const float sr = interpolate_grid_value(sr_lower, sr_upper, sr_index, k);
    const float somega = interpolate_grid_value(somega_lower, somega_upper, somega_index, k);
    if (!isfinite(sr) || sr <= 0.0f || !isfinite(somega) || somega <= 0.0f)
        return;

    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    if (!finite_params(reference))
        return;

    const solver_params params = scaled_params(sr, somega, reference);
    state ball = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {0.0f, 0.0f},
    };

    bool velocity_recovered = false;
    float minimum_x = 0.0f;
    int minimum_step = 0;
    float minimum_velocity = 0.0f;
    if (evolve_gpu(ball, dt, total_time, params, velocity_recovered, minimum_x, minimum_step, minimum_velocity) !=
        kGpuEvolveOk)
        return;

    output_velocity_recovered[output_index] = velocity_recovered ? 1 : 0;
    output_minimum_x[output_index] = minimum_x;
    output_minimum_step[output_index] = minimum_step;
    output_minimum_velocity[output_index] = minimum_velocity;
    output_status[output_index] = 0;
}

// Two-dimensional initial-condition grid layout:
//   x thread coordinate -> initial center-of-mass velocity v0
//   y thread coordinate -> initial contact-point slip velocity S0
//   flat index          -> S0_index * k + v0_index
//
// sr and somega are fixed for the whole launch.  Each thread recovers the
// initial angular velocity from
//
//     S0 = v0 + alpha * omega0.
//
// The initial position, body angle, magnetic phase, and hysteresis vector are
// zero.  Outputs use exactly the same finite-time velocity-minimum criterion
// as EvolveHysteresisGrid.
extern "C" GLOBAL void EvolveHysteresisInitialGrid(
    const float sr, const float somega, const float v0_lower, const float v0_upper, const float s0_lower,
    const float s0_upper, const int k, const float dt, const float total_time, const float reference_alpha,
    const float reference_delta, const float reference_gamma_g, const float reference_gamma,
    const float reference_gamma_m, int *output_status, int *output_velocity_recovered, float *output_minimum_x,
    int *output_minimum_step, float *output_minimum_velocity)
{
    const int v0_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int s0_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (k <= 0 || v0_index >= k || s0_index >= k)
        return;

    const int output_index = s0_index * k + v0_index;
    output_status[output_index] = 1;

    if (!isfinite(sr) || sr <= 0.0f || !isfinite(somega) || somega <= 0.0f)
        return;

    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    if (!finite_params(reference))
        return;

    const solver_params params = scaled_params(sr, somega, reference);
    if (!finite_params(params))
        return;

    const float v0 = interpolate_grid_value(v0_lower, v0_upper, v0_index, k);
    const float s0 = interpolate_grid_value(s0_lower, s0_upper, s0_index, k);
    const float omega0 = (s0 - v0) / params.alpha;
    if (!isfinite(v0) || !isfinite(s0) || !isfinite(omega0))
        return;

    state ball = {
        0.0f, 0.0f, v0, 0.0f, omega0, 0.0f, {0.0f, 0.0f},
    };

    bool velocity_recovered = false;
    float minimum_x = 0.0f;
    int minimum_step = 0;
    float minimum_velocity = v0;
    if (evolve_gpu(ball, dt, total_time, params, velocity_recovered, minimum_x, minimum_step, minimum_velocity) !=
        kGpuEvolveOk)
        return;

    output_velocity_recovered[output_index] = velocity_recovered ? 1 : 0;
    output_minimum_x[output_index] = minimum_x;
    output_minimum_step[output_index] = minimum_step;
    output_minimum_velocity[output_index] = minimum_velocity;
    output_status[output_index] = 0;
}
