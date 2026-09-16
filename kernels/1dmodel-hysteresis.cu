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
    return real_isfinite(ball.t) && real_isfinite(ball.x) && real_isfinite(ball.xdot) && real_isfinite(ball.theta) &&
           real_isfinite(ball.theta_dot) && real_isfinite(ball.phase) && real_isfinite(ball.q.x) && real_isfinite(ball.q.y);
}

DEVICE INLINE bool finite_params(const solver_params &params)
{
    return real_isfinite(params.alpha) && params.alpha > REAL_LITERAL(0.0) && real_isfinite(params.delta) && params.delta >= REAL_LITERAL(0.0) &&
           real_isfinite(params.gamma_g) && real_isfinite(params.gamma) && real_isfinite(params.gamma_m);
}

DEVICE INLINE solver_params scaled_params(const real_t sr, const real_t somega, const solver_params &reference)
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

// GPU counterpart of the initialization in 1dmodel-hysteresis.cpp.  CUDA
// device code cannot throw C++ exceptions, so failures are returned as flags.
DEVICE INLINE int initialize_gpu_mode(state &ball, const solver_params &params, bool &sliding, real_t &slip_sign)
{
    const real_t slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    sliding = real_fabs(slip_velocity) > REAL_LITERAL(1.0e-10);
    slip_sign = sliding ? signum(slip_velocity) : REAL_LITERAL(0.0);

    if (sliding)
        return kGpuEvolveOk;

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
        return kGpuEvolveStepFailed;

    const real_t magnetic_torque = -params.gamma_m * E * cross(ball.q, h_ext);
    const real_t magnetic_force = ETA * params.alpha * params.alpha * params.gamma_m *
                                 (E * (ball.q * h_ext_phase_derivative) -
                                  A * (h_ext * h_ext_phase_derivative));
    const real_t theta_acceleration = ETA / (REAL_LITERAL(1.0) + ETA) * magnetic_torque -
                                     magnetic_force / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
                                     params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * ball.theta_dot *
                                         real_fabs(ball.theta_dot);
    if (!real_isfinite(magnetic_torque) || !real_isfinite(magnetic_force) || !real_isfinite(theta_acceleration))
        return kGpuEvolveNonFiniteState;

    if (real_fabs(theta_acceleration - magnetic_torque) * ETA * params.alpha > MU * normal_force) {
        sliding = true;
        slip_sign = -signum(theta_acceleration - magnetic_torque);
        if (slip_sign == REAL_LITERAL(0.0))
            return kGpuEvolveStepFailed;
    } else {
        slip_sign = REAL_LITERAL(0.0);
        project_pure_rolling(ball, params);
    }

    return kGpuEvolveOk;
}

DEVICE INLINE void update_velocity_minimum(const state &ball, const int completed_step, bool &velocity_recovered,
                                           real_t &minimum_x, int &minimum_step, real_t &minimum_velocity)
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
DEVICE INLINE int evolve_gpu(state &ball, const real_t dt, const real_t total_time, const solver_params &params,
                             bool &velocity_recovered, real_t &minimum_x, int &minimum_step, real_t &minimum_velocity)
{
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || !real_isfinite(total_time) || total_time < REAL_LITERAL(0.0) || !finite_params(params))
        return kGpuEvolveInvalidInput;

    const real_t step_count = real_floor(total_time / dt);
    if (!real_isfinite(step_count) || step_count > REAL_LITERAL(2147483000.0))
        return kGpuEvolveInvalidInput;

    bool sliding = false;
    real_t slip_sign = REAL_LITERAL(0.0);
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
    const real_t remainder = total_time - static_cast<real_t>(full_steps) * dt;
    if (remainder > REAL_LITERAL(0.0)) {
        if (fwd1step(ball, remainder, params, sliding, slip_sign) != 0)
            return kGpuEvolveStepFailed;
        if (!finite_state(ball))
            return kGpuEvolveNonFiniteState;
        update_velocity_minimum(ball, full_steps + 1, velocity_recovered, minimum_x, minimum_step, minimum_velocity);
    }

    return kGpuEvolveOk;
}

DEVICE INLINE real_t interpolate_grid_value(const real_t lower, const real_t upper, const int index, const int count)
{
    if (count == 1)
        return lower;
    const real_t fraction = static_cast<real_t>(index) / static_cast<real_t>(count - 1);
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
extern "C" GLOBAL void EvolveHysteresisGrid(const real_t sr_lower, const real_t sr_upper, const real_t somega_lower,
                                            const real_t somega_upper, const int k, const real_t dt,
                                            const real_t total_time, const real_t reference_alpha,
                                            const real_t reference_delta, const real_t reference_gamma_g,
                                            const real_t reference_gamma, const real_t reference_gamma_m,
                                            int *output_status, int *output_velocity_recovered, real_t *output_minimum_x,
                                            int *output_minimum_step, real_t *output_minimum_velocity)
{
    const int sr_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int somega_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (k <= 0 || sr_index >= k || somega_index >= k)
        return;

    const int output_index = somega_index * k + sr_index;
    output_status[output_index] = 1;

    const real_t sr = interpolate_grid_value(sr_lower, sr_upper, sr_index, k);
    const real_t somega = interpolate_grid_value(somega_lower, somega_upper, somega_index, k);
    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) || somega <= REAL_LITERAL(0.0))
        return;

    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    if (!finite_params(reference))
        return;

    const solver_params params = scaled_params(sr, somega, reference);
    state ball = {
        REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)},
    };

    bool velocity_recovered = false;
    real_t minimum_x = REAL_LITERAL(0.0);
    int minimum_step = 0;
    real_t minimum_velocity = REAL_LITERAL(0.0);
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
    const real_t sr, const real_t somega, const real_t v0_lower, const real_t v0_upper, const real_t s0_lower,
    const real_t s0_upper, const int k, const real_t dt, const real_t total_time, const real_t reference_alpha,
    const real_t reference_delta, const real_t reference_gamma_g, const real_t reference_gamma,
    const real_t reference_gamma_m, int *output_status, int *output_velocity_recovered, real_t *output_minimum_x,
    int *output_minimum_step, real_t *output_minimum_velocity)
{
    const int v0_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int s0_index = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (k <= 0 || v0_index >= k || s0_index >= k)
        return;

    const int output_index = s0_index * k + v0_index;
    output_status[output_index] = 1;

    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) || somega <= REAL_LITERAL(0.0))
        return;

    const solver_params reference = {
        reference_alpha, reference_delta, reference_gamma_g, reference_gamma, reference_gamma_m,
    };
    if (!finite_params(reference))
        return;

    const solver_params params = scaled_params(sr, somega, reference);
    if (!finite_params(params))
        return;

    const real_t v0 = interpolate_grid_value(v0_lower, v0_upper, v0_index, k);
    const real_t s0 = interpolate_grid_value(s0_lower, s0_upper, s0_index, k);
    const real_t omega0 = (s0 - v0) / params.alpha;
    if (!real_isfinite(v0) || !real_isfinite(s0) || !real_isfinite(omega0))
        return;

    state ball = {
        REAL_LITERAL(0.0), REAL_LITERAL(0.0), v0, REAL_LITERAL(0.0), omega0, REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)},
    };

    bool velocity_recovered = false;
    real_t minimum_x = REAL_LITERAL(0.0);
    int minimum_step = 0;
    real_t minimum_velocity = v0;
    if (evolve_gpu(ball, dt, total_time, params, velocity_recovered, minimum_x, minimum_step, minimum_velocity) !=
        kGpuEvolveOk)
        return;

    output_velocity_recovered[output_index] = velocity_recovered ? 1 : 0;
    output_minimum_x[output_index] = minimum_x;
    output_minimum_step[output_index] = minimum_step;
    output_minimum_velocity[output_index] = minimum_velocity;
    output_status[output_index] = 0;
}
