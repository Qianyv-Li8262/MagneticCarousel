// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include <cuda_runtime.h>

#include <math_constants.h>

namespace
{

// Parameters which are not swept by the kernel.
constexpr float kBaseGamma = 2.5f;
constexpr float kBaseEpsilon = 50.0f;
constexpr float kEta = 0.4f;
constexpr float kMuS = 0.02f;
constexpr float kMuK = 0.02f;
constexpr float kDelta = 6.4e-5f;
constexpr float kSlipTolerance = 1.0e-10f;
constexpr int kStickWindow = 300;

struct State {
    float x;
    float v;
    float theta;
    float omega;
    // Auxiliary phase = t - x - theta. Keeping it as a state avoids
    // catastrophic FP32 cancellation when x and theta become large.
    float phase;
};

__device__ __forceinline__ float signum(const float value)
{
    return static_cast<float>((value > 0.0f) - (value < 0.0f));
}

__device__ __forceinline__ State addScaled(const State &base, const State &rhs, const float scale)
{
    return {base.x + scale * rhs.x, base.v + scale * rhs.v, base.theta + scale * rhs.theta,
            base.omega + scale * rhs.omega, base.phase + scale * rhs.phase};
}

__device__ __forceinline__ float normalForce(const State &state, const float time, const float gamma,
                                             const float epsilon)
{
    (void)time;
    return gamma + epsilon * cosf(state.phase);
}

// Device equivalent of solve_toymodel_animated.py::evalRhs.
// False means that a negative normal force was encountered.
__device__ __forceinline__ bool evalRhs(const State &input, const bool sliding, const float time, float slip_sign,
                                        const float alpha, const float delta, const float gamma, const float epsilon,
                                        State &ans)
{
    ans = {input.v, 0.0f, input.omega, 0.0f,
           1.0f - input.v - input.omega};

    (void)time;
    const float phase = input.phase;
    const float N = gamma + epsilon * cosf(phase);
    if (N < 0.0f) {
        return false;
    }

    if (sliding) {
        const float S = input.v + alpha * input.omega;
        if (slip_sign == 0.0f) {
            slip_sign = signum(S);
        }

        const float F = -slip_sign * kMuK * N;
        ans.v = epsilon * sinf(phase) + F - delta * input.v * fabsf(input.v);
        ans.omega = epsilon / (kEta * alpha * alpha) * sinf(phase) + F / (kEta * alpha);
    } else {
        ans.v = (epsilon * (1.0f - 1.0f / alpha) * sinf(phase) - delta * input.v * fabsf(input.v)) / (1.0f + kEta);
        ans.omega = -ans.v / alpha;
    }

    return true;
}

// Device equivalent of solve_toymodel_animated.py::fwd1Step.
// State/mode are committed only if all sampled normal forces are nonnegative.
__device__ __forceinline__ bool fwd1Step(State &state, bool &sliding, float &slip_sign, const float time,
                                         const float dt, const float alpha, const float delta, const float gamma,
                                         const float epsilon)
{
    const State input = state;

    State rhs;
    if (!evalRhs(input, sliding, time, slip_sign, alpha, delta, gamma, epsilon, rhs)) {
        return false;
    }
    const State midpoint_trial = addScaled(input, rhs, 0.5f * dt);

    if (!evalRhs(midpoint_trial, sliding, time + 0.5f * dt, slip_sign, alpha, delta, gamma, epsilon, rhs)) {
        return false;
    }
    const State final_trial = addScaled(input, rhs, dt);
    if (normalForce(final_trial, time + dt, gamma, epsilon) < 0.0f) {
        return false;
    }

    // Current mode: sliding.
    if (sliding) {
        const float S_init = input.v + alpha * input.omega;
        const float S_final = final_trial.v + alpha * final_trial.omega;

        if (slip_sign == 0.0f) {
            slip_sign = signum(S_init);
        }

        // S did not cross zero: accept the complete midpoint step.
        if (S_init * S_final >= 0.0f) {
            state = final_trial;
            return true;
        }

        // Locate S == 0 linearly, then reintegrate to the event with midpoint RK2.
        const float t_star = dt * S_init / (S_init - S_final);

        if (!evalRhs(input, true, time, slip_sign, alpha, delta, gamma, epsilon, rhs)) {
            return false;
        }
        const State midpoint_star = addScaled(input, rhs, 0.5f * t_star);

        if (!evalRhs(midpoint_star, true, time + 0.5f * t_star, slip_sign, alpha, delta, gamma, epsilon, rhs)) {
            return false;
        }
        State event_tstar = addScaled(input, rhs, t_star);

        // Project the event state back onto the rolling constraint S == 0.
        const float u = event_tstar.v;
        const float omega = event_tstar.omega;
        event_tstar.v = (u - kEta * alpha * omega) / (1.0f + kEta);
        event_tstar.omega = -event_tstar.v / alpha;

        const float event_time = time + t_star;
        const float phase = event_tstar.phase;
        const float N = gamma + epsilon * cosf(phase);
        if (N < 0.0f) {
            return false;
        }

        const float F_req =
            (kEta * delta * event_tstar.v * fabsf(event_tstar.v) - epsilon * (kEta + 1.0f / alpha) * sinf(phase)) /
            (1.0f + kEta);
        const float remain = dt - t_star;

        if (fabsf(F_req) <= kMuS * N) {
            // slip -> stick
            if (!evalRhs(event_tstar, false, event_time, 0.0f, alpha, delta, gamma, epsilon, rhs)) {
                return false;
            }
            const State midpoint_remain = addScaled(event_tstar, rhs, 0.5f * remain);

            if (!evalRhs(midpoint_remain, false, event_time + 0.5f * remain, 0.0f, alpha, delta, gamma, epsilon, rhs)) {
                return false;
            }
            State final_state = addScaled(event_tstar, rhs, remain);
            if (normalForce(final_state, time + dt, gamma, epsilon) < 0.0f) {
                return false;
            }
            final_state.omega = -final_state.v / alpha;

            state = final_state;
            sliding = false;
            slip_sign = 0.0f;
            return true;
        }

        // Static friction is insufficient: cross S == 0 and keep sliding.
        const float new_slip_sign = -signum(F_req);
        if (!evalRhs(event_tstar, true, event_time, new_slip_sign, alpha, delta, gamma, epsilon, rhs)) {
            return false;
        }
        const State midpoint_remain = addScaled(event_tstar, rhs, 0.5f * remain);

        if (!evalRhs(midpoint_remain, true, event_time + 0.5f * remain, new_slip_sign, alpha, delta, gamma, epsilon,
                     rhs)) {
            return false;
        }
        const State final_state = addScaled(event_tstar, rhs, remain);
        if (normalForce(final_state, time + dt, gamma, epsilon) < 0.0f) {
            return false;
        }

        state = final_state;
        sliding = true;
        slip_sign = new_slip_sign;
        return true;
    }

    // Current mode: sticking.
    const float phase_init = input.phase;
    const float N_init = gamma + epsilon * cosf(phase_init);
    if (N_init < 0.0f) {
        return false;
    }

    const float F_req_init =
        (kEta * delta * input.v * fabsf(input.v) - epsilon * (kEta + 1.0f / alpha) * sinf(phase_init)) / (1.0f + kEta);
    const float g_init = kMuS * N_init - fabsf(F_req_init);

    // FP32 roundoff can put a nominally sticking state just beyond the static
    // friction boundary at the beginning of a step. In that case there is no
    // bracket [g_init >= 0, g_final < 0] for event interpolation: start the
    // complete step in sliding mode instead of evaluating an invalid t_star.
    if (g_init < 0.0f) {
        const float new_slip_sign = -signum(F_req_init);
        if (!evalRhs(input, true, time, new_slip_sign,
                     alpha, delta, gamma, epsilon, rhs)) {
            return false;
        }
        const State midpoint_sliding = addScaled(input, rhs, 0.5f * dt);
        if (!evalRhs(midpoint_sliding, true, time + 0.5f * dt,
                     new_slip_sign, alpha, delta, gamma, epsilon, rhs)) {
            return false;
        }
        const State final_sliding = addScaled(input, rhs, dt);
        if (normalForce(final_sliding, time + dt, gamma, epsilon) < 0.0f) {
            return false;
        }

        state = final_sliding;
        sliding = true;
        slip_sign = new_slip_sign;
        return true;
    }

    const float phase_final = final_trial.phase;
    const float N_final = gamma + epsilon * cosf(phase_final);
    if (N_final < 0.0f) {
        return false;
    }

    const float F_req_final =
        (kEta * delta * final_trial.v * fabsf(final_trial.v) - epsilon * (kEta + 1.0f / alpha) * sinf(phase_final)) /
        (1.0f + kEta);
    const float g_final = kMuS * N_final - fabsf(F_req_final);

    if (g_final >= 0.0f) {
        State final_stick = final_trial;
        final_stick.omega = -final_stick.v / alpha;
        state = final_stick;
        sliding = false;
        slip_sign = 0.0f;
        return true;
    }

    // stick -> slip
    const float t_star = dt * g_init / (g_init - g_final);

    if (!evalRhs(input, false, time, 0.0f, alpha, delta, gamma, epsilon, rhs)) {
        return false;
    }
    const State midpoint_star = addScaled(input, rhs, 0.5f * t_star);

    if (!evalRhs(midpoint_star, false, time + 0.5f * t_star, 0.0f, alpha, delta, gamma, epsilon, rhs)) {
        return false;
    }
    const State event_tstar = addScaled(input, rhs, t_star);

    const float event_time = time + t_star;
    const float phase = event_tstar.phase;
    const float N = gamma + epsilon * cosf(phase);
    if (N < 0.0f) {
        return false;
    }

    const float F_req =
        (kEta * delta * event_tstar.v * fabsf(event_tstar.v) - epsilon * (kEta + 1.0f / alpha) * sinf(phase)) /
        (1.0f + kEta);
    const float new_slip_sign = -signum(F_req);
    const float remain = dt - t_star;

    if (!evalRhs(event_tstar, true, event_time, new_slip_sign, alpha, delta, gamma, epsilon, rhs)) {
        return false;
    }
    const State midpoint_remain = addScaled(event_tstar, rhs, 0.5f * remain);

    if (!evalRhs(midpoint_remain, true, event_time + 0.5f * remain, new_slip_sign, alpha, delta, gamma, epsilon, rhs)) {
        return false;
    }
    const State final_state = addScaled(event_tstar, rhs, remain);
    if (normalForce(final_state, time + dt, gamma, epsilon) < 0.0f) {
        return false;
    }

    state = final_state;
    sliding = true;
    slip_sign = new_slip_sign;
    return true;
}

constexpr int kTraceValueCount = 10;

__device__ __forceinline__ void writeTraceRecord(
    const int step,
    const int event,
    const bool sliding,
    const float time,
    const State &state,
    const float slip_sign,
    const float alpha,
    const float delta,
    const float gamma,
    const float epsilon,
    const int capacity,
    int &count,
    int *log_steps,
    int *log_events,
    int *log_modes,
    float *log_values)
{
    if (count >= capacity) {
        return;
    }

    const float N = gamma + epsilon * cosf(state.phase);
    const float F_req =
        (kEta * delta * state.v * fabsf(state.v) -
         epsilon * (kEta + 1.0f / alpha) * sinf(state.phase)) /
        (1.0f + kEta);
    const int offset = count * kTraceValueCount;

    log_steps[count] = step;
    log_events[count] = event;
    log_modes[count] = sliding ? 1 : 0;
    log_values[offset + 0] = time;
    log_values[offset + 1] = state.x;
    log_values[offset + 2] = state.v;
    log_values[offset + 3] = state.theta;
    log_values[offset + 4] = state.omega;
    log_values[offset + 5] = state.phase;
    log_values[offset + 6] = N;
    log_values[offset + 7] = state.v + alpha * state.omega;
    log_values[offset + 8] = F_req;
    log_values[offset + 9] = slip_sign;
    ++count;
}

} // namespace

// Launch with a two-dimensional grid. The global x coordinate selects alpha,
// and the global y coordinate selects the driving omega. Output layout is
// [omega_index * k + alpha_index].
//
// output_stick300 and output_x_sign must point to int32 arrays; output_v must
// point to a float32 array. If a negative normal force is found, only
// output_stick300[index] is written (-1), and the thread returns immediately.
extern "C" __global__ void Evolve(const float alpha_lower, const float alpha_upper, const int k,
                                  const float omega_lower, const float omega_upper, const float total_time,
                                  const float dt, int *output_stick300, int *output_x_sign, float *output_v)
{
    const int alpha_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int omega_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (alpha_index >= k || omega_index >= k) {
        return;
    }

    const int index = omega_index * k + alpha_index;
    const float alpha_fraction = k > 1 ? static_cast<float>(alpha_index) / static_cast<float>(k - 1) : 0.0f;
    const float omega_fraction = k > 1 ? static_cast<float>(omega_index) / static_cast<float>(k - 1) : 0.0f;
    const float alpha = alpha_lower + (alpha_upper - alpha_lower) * alpha_fraction;
    const float drive_omega = omega_lower + (omega_upper - omega_lower) * omega_fraction;
    const float omega_squared = drive_omega * drive_omega;
    const float epsilon = kBaseEpsilon / omega_squared;
    const float gamma = kBaseGamma / omega_squared;
    const float delta = kDelta;

    // Common initial state: [x, v, theta, omega] = [0, 0, 0, 0].
    State state = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool sliding;
    float slip_sign;

    const float S = state.v + alpha * state.omega;
    if (fabsf(S) > kSlipTolerance) {
        sliding = true;
        slip_sign = signum(S);
    } else {
        const float phase = state.phase;
        const float N = gamma + epsilon * cosf(phase);
        if (N < 0.0f) {
            output_stick300[index] = -1;
            return;
        }

        const float F_req =
            (kEta * delta * state.v * fabsf(state.v) - epsilon * (kEta + 1.0f / alpha) * sinf(phase)) / (1.0f + kEta);

        if (fabsf(F_req) <= kMuS * N) {
            sliding = false;
            slip_sign = 0.0f;
            state.omega = -state.v / alpha;
        } else {
            sliding = true;
            slip_sign = -signum(F_req);
        }
    }

    // A run shorter than 300 completed steps cannot satisfy "the past 300
    // steps were all sticking". Capping avoids overflow in very long runs.
    int consecutive_stick_steps = 0;
    // Correct the possible one-ULP underestimate when total_time is intended
    // to be an integer multiple of dt (for example, 1.0f / 0.001f).
    const float step_ratio = total_time / dt;
    const long long steps = static_cast<long long>(nextafterf(step_ratio, CUDART_INF_F));

    for (long long step = 0; step < steps; ++step) {
        // Multiplication avoids the large drift from repeatedly adding dt in
        // a long, entirely FP32 evolution.
        const float time = static_cast<float>(step) * dt;
        if (!fwd1Step(state, sliding, slip_sign, time, dt, alpha, delta, gamma, epsilon)) {
            output_stick300[index] = -1;
            return;
        }
        if (sliding) {
            consecutive_stick_steps = 0;
        } else if (consecutive_stick_steps < kStickWindow) {
            ++consecutive_stick_steps;
        }
    }

    output_stick300[index] = consecutive_stick_steps >= kStickWindow ? 1 : 0;
    output_x_sign[index] = static_cast<int>(signum(state.x));
    output_v[index] = state.v;
}

// Single-thread diagnostic kernel. Event codes:
// 0 periodic/final, 1 stick->slip, 2 slip->stick,
// 3 sliding direction reversal, 4 negative normal force.
extern "C" __global__ void TraceEvolve(
    const float alpha,
    const float drive_omega,
    const float total_time,
    const float dt,
    const float initial_x,
    const float initial_v,
    const float initial_theta,
    const float initial_state_omega,
    const int log_every,
    const int log_capacity,
    int *log_steps,
    int *log_events,
    int *log_modes,
    float *log_values,
    int *log_count)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const float omega_squared = drive_omega * drive_omega;
    const float epsilon = kBaseEpsilon / omega_squared;
    const float gamma = kBaseGamma / omega_squared;
    const float delta = kDelta;
    State state = {
        initial_x,
        initial_v,
        initial_theta,
        initial_state_omega,
        -initial_x - initial_theta
    };

    bool sliding;
    float slip_sign;
    const float initial_S = state.v + alpha * state.omega;
    const float initial_N = gamma + epsilon * cosf(state.phase);
    const float initial_F_req =
        (kEta * delta * state.v * fabsf(state.v) -
         epsilon * (kEta + 1.0f / alpha) * sinf(state.phase)) /
        (1.0f + kEta);

    if (fabsf(initial_S) > kSlipTolerance) {
        sliding = true;
        slip_sign = signum(initial_S);
    } else if (initial_N >= 0.0f && fabsf(initial_F_req) <= kMuS * initial_N) {
        sliding = false;
        slip_sign = 0.0f;
        state.omega = -state.v / alpha;
    } else {
        sliding = true;
        slip_sign = -signum(initial_F_req);
    }

    int count = 0;
    writeTraceRecord(
        0, initial_N < 0.0f ? 4 : 0, sliding, 0.0f,
        state, slip_sign, alpha, delta, gamma, epsilon,
        log_capacity, count, log_steps, log_events, log_modes, log_values
    );
    if (initial_N < 0.0f) {
        *log_count = count;
        return;
    }

    const float step_ratio = total_time / dt;
    const long long steps = static_cast<long long>(
        nextafterf(step_ratio, CUDART_INF_F)
    );

    for (long long step = 0; step < steps; ++step) {
        const float time = static_cast<float>(step) * dt;
        const bool was_sliding = sliding;
        const float old_slip_sign = slip_sign;
        const bool valid = fwd1Step(
            state, sliding, slip_sign, time, dt,
            alpha, delta, gamma, epsilon
        );

        int event = 0;
        if (!valid) {
            event = 4;
        } else if (!was_sliding && sliding) {
            event = 1;
        } else if (was_sliding && !sliding) {
            event = 2;
        } else if (sliding && old_slip_sign != slip_sign) {
            event = 3;
        }

        const int completed_step = static_cast<int>(step + 1);
        const bool periodic = log_every > 0 && completed_step % log_every == 0;
        if (event != 0 || periodic || step + 1 == steps) {
            writeTraceRecord(
                completed_step, event, sliding, time + dt,
                state, slip_sign, alpha, delta, gamma, epsilon,
                log_capacity, count, log_steps, log_events,
                log_modes, log_values
            );
        }

        if (!valid) {
            break;
        }
    }

    *log_count = count;
}
