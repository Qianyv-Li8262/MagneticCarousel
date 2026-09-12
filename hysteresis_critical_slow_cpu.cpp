#include "hysteresis_critical_slow_cpu.hpp"

#include <stdexcept>
#include <deque>
#include <algorithm>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace {
void validate_inputs(const double dt, const std::int64_t max_confirm, const solver_params &params)
{
    if (!std::isfinite(dt) || dt <= 0.0 || max_confirm < 0)
        throw std::invalid_argument("dt must be finite and positive; max_confirm must be nonnegative");
    if (!std::isfinite(params.alpha) || params.alpha == 0.0 || params.alpha == 1.0 ||
        !std::isfinite(params.delta) || !std::isfinite(params.gamma_g) ||
        !std::isfinite(params.gamma) || !std::isfinite(params.gamma_m))
        throw std::invalid_argument("parameters must be finite and alpha must differ from 0 and 1");

    const double frame_time = critical_slow_confirm_evolve_steps * dt;
    if (!std::isfinite(frame_time) ||
        !std::isfinite(static_cast<double>(max_confirm) * frame_time))
        throw std::invalid_argument("requested duration exceeds FP64 range");
}

solver_params scaled_params(const double sr, const double somega, const solver_params &reference)
{
    if (!std::isfinite(sr) || sr <= 0.0 || !std::isfinite(somega) || somega <= 0.0)
        throw std::invalid_argument("sr and somega must be finite and positive");
    const double inverse_sr = 1.0 / sr;
    const double inverse_somega_squared = 1.0 / (somega * somega);
    return {
        reference.alpha * sr,
        reference.delta * inverse_sr,
        reference.gamma_g * inverse_somega_squared,
        reference.gamma * inverse_somega_squared,
        reference.gamma_m * inverse_sr * inverse_sr * inverse_somega_squared,
    };
}
} // namespace

critical_slow_result evolve_critical_slow(const double dt, const std::int64_t max_confirm,
                                         const solver_params &params)
{
    validate_inputs(dt, max_confirm, params);
    const double frame_time = critical_slow_confirm_evolve_steps * dt;

    state ball = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {0.0, 0.0}};
    bool sliding = false;
    double slip_sign = 0.0;
    const vec2 h_ext = {H_EXT * real_cos(ball.phase), H_EXT * real_sin(ball.phase)};
    const double N = params.gamma_g - params.gamma / (2.0 + MU_R) *
        (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - XI_JA) * (ball.q * h_ext));
    if (N < 0.0 || !std::isfinite(N))
        return {-1, -1.0};
    const double M_magnet = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(ball.q, h_ext);
    const double theta_dot = ETA / (1 + ETA) * (1 - params.alpha) * M_magnet -
        params.alpha / (1 + ETA) * params.delta * ball.theta_dot * real_fabs(ball.theta_dot);
    if (real_fabs(theta_dot - M_magnet) * ETA * params.alpha > MU * N) {
        sliding = true;
        slip_sign = -signum(theta_dot - M_magnet);
    } else {
        ball.theta_dot = -ball.xdot / params.alpha;
    }

    // Phase locking is defined by the same order parameter as the phase scan:
    // |<phase_rate>| < 0.05, where phase_rate = 1 - xdot - theta_dot.
    // Smooth it over 1000 time units, then require an uninterrupted in-band
    // run of 2500 time units before accepting the first window as persistent.
    struct mean_segment { double duration, phase_rate; };
    std::deque<mean_segment> phase_history;
    constexpr double smooth_duration = 1000.0;
    double smooth_span = 0.0, smooth_integral = 0.0;
    double lock_entry_time = -1.0;
    int locked = 0;
    for (std::int64_t i = 0; i < max_confirm; ++i) {
        for (int j = 0; j < critical_slow_confirm_evolve_steps; ++j) {
            if (fwd1step(ball, dt, params, sliding, slip_sign) != 0)
                return {-1, -1.0};
        }
        // The CUDA classifier has no non-finite guard. Reject numerical failures
        // here instead of silently treating NaNs as an undecided trajectory.
        if (!std::isfinite(ball.t) || !std::isfinite(ball.x) || !std::isfinite(ball.xdot) ||
            !std::isfinite(ball.theta) || !std::isfinite(ball.theta_dot) ||
            !std::isfinite(ball.phase) || !std::isfinite(ball.q.x) || !std::isfinite(ball.q.y))
            return {-1, -1.0};
        const double phase_rate = 1.0 - ball.xdot - ball.theta_dot;
        if (!std::isfinite(phase_rate))
            return {-1, -1.0};
        phase_history.push_back({frame_time, phase_rate});
        smooth_span += frame_time;
        smooth_integral += frame_time * phase_rate;
        while (smooth_span > smooth_duration && !phase_history.empty()) {
            auto &front = phase_history.front();
            const double removed = std::min(front.duration, smooth_span - smooth_duration);
            smooth_integral -= removed * front.phase_rate;
            smooth_span -= removed;
            front.duration -= removed;
            if (front.duration <= 0.0)
                phase_history.pop_front();
        }
        const double end_time = static_cast<double>(i + 1) * frame_time;
        if (end_time >= smooth_duration) {
            const double smooth_phase_rate = smooth_integral / smooth_span;
            if (std::abs(smooth_phase_rate) < critical_slow_phase_rate_tolerance) {
                if (lock_entry_time < 0.0)
                    lock_entry_time = end_time - smooth_duration / 2.0;
                locked = std::min(locked + 1, critical_slow_lock_confirm_window);
            } else {
                lock_entry_time = -1.0;
                locked = 0;
            }
        }
        if (locked == critical_slow_lock_confirm_window && lock_entry_time >= 0.0)
            return {1, lock_entry_time};
    }
    return {0, -1.0};
}

void calc_slow_cpu(std::int32_t *answer_state, double *transit_time, const double *scan_values,
                   const std::size_t k, const double fixed_scale, const critical_slow_scan_mode mode,
                   const double dt, const std::int64_t max_confirm, const solver_params reference,
                   const int num_threads)
{
    if (num_threads <= 0)
        throw std::invalid_argument("num_threads must be positive");
    if (!std::isfinite(fixed_scale) || fixed_scale <= 0.0)
        throw std::invalid_argument("fixed_scale must be finite and positive");
    if (mode != critical_slow_scan_mode::somega && mode != critical_slow_scan_mode::sr)
        throw std::invalid_argument("invalid critical-slowing scan mode");
    if (k != 0 && (!answer_state || !transit_time || !scan_values))
        throw std::invalid_argument("nonempty arrays must not be null");
    // Validate the complete batch before touching any output or launching workers.
    for (std::size_t i = 0; i < k; ++i) {
        const double sr = mode == critical_slow_scan_mode::sr ? scan_values[i] : fixed_scale;
        const double somega = mode == critical_slow_scan_mode::somega ? scan_values[i] : fixed_scale;
        validate_inputs(dt, max_confirm, scaled_params(sr, somega, reference));
    }
    if (k == 0)
        return;

    std::mutex queue_mutex;
    std::size_t next = 0;
    std::exception_ptr failure;
    auto worker = [&]() {
        try {
            for (;;) {
                std::size_t index;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (failure || next == k)
                        return;
                    index = next++;
                }
                // Each worker owns a whole trajectory and its unique output slot.
                const double sr = mode == critical_slow_scan_mode::sr ? scan_values[index] : fixed_scale;
                const double somega = mode == critical_slow_scan_mode::somega ? scan_values[index] : fixed_scale;
                const auto result = evolve_critical_slow(
                    dt, max_confirm, scaled_params(sr, somega, reference));
                answer_state[index] = result.answer_state;
                transit_time[index] = result.transit_time;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!failure)
                failure = std::current_exception();
        }
    };
    const auto count = std::min(k, static_cast<std::size_t>(num_threads));
    std::vector<std::thread> workers;
    workers.reserve(count);
    try {
        for (std::size_t i = 0; i < count; ++i)
            workers.emplace_back(worker);
    } catch (...) {
        // Even a partial thread launch must join before stack/array lifetimes end.
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!failure)
                failure = std::current_exception();
        }
    }
    for (auto &thread : workers)
        thread.join();
    if (failure)
        std::rethrow_exception(failure);
}
