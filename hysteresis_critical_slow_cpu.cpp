#include "hysteresis_critical_slow_cpu.hpp"

#include <stdexcept>
#include <array>
#include <deque>
#include <algorithm>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace {
void validate_inputs(const double v0, const double dt, const std::int64_t max_confirm,
                     const solver_params &params)
{
    if (!std::isfinite(v0) || !std::isfinite(dt) || dt <= 0.0 || max_confirm < 0)
        throw std::invalid_argument("v0 and dt must be finite, dt > 0, max_confirm >= 0");
    if (!std::isfinite(params.alpha) || params.alpha == 0.0 || params.alpha == 1.0 ||
        !std::isfinite(params.delta) || !std::isfinite(params.gamma_g) ||
        !std::isfinite(params.gamma) || !std::isfinite(params.gamma_m))
        throw std::invalid_argument("parameters must be finite and alpha must differ from 0 and 1");

    const double frame_time = critical_slow_confirm_evolve_steps * dt;
    if (!std::isfinite(frame_time) ||
        !std::isfinite(static_cast<double>(max_confirm) * frame_time))
        throw std::invalid_argument("requested duration exceeds FP64 range");
}
} // namespace

critical_slow_result evolve_critical_slow(const double v0, const double dt,
                                         const std::int64_t max_confirm, const solver_params &params)
{
    validate_inputs(v0, dt, max_confirm, params);
    const double frame_time = critical_slow_confirm_evolve_steps * dt;

    state ball = {0.0, 0.0, v0, 0.0, -v0 / params.alpha, 0.0, {0.0, 0.0}};
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

    // Recovery uses the criteria from analyze_critical_slowing_detailed.detect_outcome,
    // with a finer sampling grid and the same physical window at dt=0.001.
    // Keep endpoint timestamps and the earliest global minimum on ties.
    struct sample { double t, v, phase_rate, slip; };
    constexpr int window = critical_slow_confirm_window;
    std::array<sample, window> history{};
    const double theoretical_speed = -params.alpha / (1 - params.alpha);
    int locked = 0;
    // A trailing time-weighted mean; each frame represents constant mean speed.
    // Track the current in-band run online, equivalent to backtracking at confirmation.
    struct mean_segment { double duration, speed; };
    std::deque<mean_segment> lock_history;
    constexpr double smooth_duration = 1000.0;
    double smooth_span = 0.0, smooth_integral = 0.0;
    double lock_entry_time = -1.0;
    double minimum_speed = std::numeric_limits<double>::infinity();
    double minimum_time = -1.0;
    std::int64_t minimum_index = -1;
    for (std::int64_t i = 0; i < max_confirm; ++i) {
        const double previous_x = ball.x;
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
        const double avg_spd = (ball.x - previous_x) / frame_time;
        if (!std::isfinite(avg_spd))
            return {-1, -1.0};
        lock_history.push_back({frame_time, avg_spd});
        smooth_span += frame_time;
        smooth_integral += frame_time * avg_spd;
        while (smooth_span > smooth_duration && !lock_history.empty()) {
            auto &front = lock_history.front();
            const double removed = std::min(front.duration, smooth_span - smooth_duration);
            smooth_integral -= removed * front.speed;
            smooth_span -= removed;
            front.duration -= removed;
            if (front.duration <= 0.0)
                lock_history.pop_front();
        }
        const double end_time = static_cast<double>(i + 1) * frame_time;
        if (end_time >= smooth_duration) {
            const double smooth_speed = smooth_integral / smooth_span;
            if (std::abs(smooth_speed - theoretical_speed) < 1.0e-3) {
                if (lock_entry_time < 0.0)
                    lock_entry_time = end_time - smooth_duration / 2.0;
            } else {
                lock_entry_time = -1.0;
            }
        }
        // Keep the original raw-frame confirmation. Only the reported event
        // time changes. A full smoothing window must also be available/in band.
        if (avg_spd > theoretical_speed - 1.0e-3 &&
            avg_spd < theoretical_speed + 1.0e-3)
            locked = std::min(locked + 1, critical_slow_lock_confirm_window);
        else
            locked = 0;
        if (locked == critical_slow_lock_confirm_window && lock_entry_time >= 0.0)
            return {1, lock_entry_time};
        const double phase_rate = 1.0 - ball.xdot - ball.theta_dot;
        const double slip = ball.xdot + params.alpha * ball.theta_dot;
        if (!std::isfinite(phase_rate) || !std::isfinite(slip))
            return {-1, -1.0};
        history[i % window] = {ball.t, avg_spd, phase_rate, slip};
        if (avg_spd < minimum_speed) {
            minimum_speed = avg_spd;
            minimum_time = ball.t;
            minimum_index = i;
        }
        if (i < window - 1)
            continue;
        const std::int64_t start = i - window + 1;
        double mean_t = 0.0, mean_v = 0.0, mean_phase = 0.0;
        double max_slip = 0.0;
        double early = 0.0, late = 0.0;
        for (int j = 0; j < window; ++j) {
            const auto &point = history[(start + j) % window];
            mean_t += point.t;
            mean_v += point.v;
            mean_phase += point.phase_rate;
            max_slip = std::max(max_slip, std::abs(point.slip));
            (j < window / 2 ? early : late) += point.v;
        }
        mean_t /= window;
        mean_v /= window;
        mean_phase /= window;
        double numerator = 0.0, denominator = 0.0;
        for (int j = 0; j < window; ++j) {
            const auto &point = history[(start + j) % window];
            const double centered_t = point.t - mean_t;
            numerator += centered_t * (point.v - mean_v);
            denominator += centered_t * centered_t;
        }
        const double slope = denominator == 0.0 ? 0.0 : numerator / denominator;
        const double half_rise = late / (window - window / 2) - early / (window / 2);
        if (minimum_index < start + window / 2 && slope >= 2.0e-7 &&
            half_rise >= 1.0e-3 && avg_spd - minimum_speed >= 2.0e-3 &&
            mean_phase >= 2.0e-2 && max_slip <= 1.0e-6)
            return {2, minimum_time};
    }
    return {0, -1.0};
}

void calc_slow_cpu(std::int32_t *answer_state, double *transit_time, const double *v0,
                   const std::size_t k, const double dt, const std::int64_t max_confirm,
                   const solver_params params, const int num_threads)
{
    if (num_threads <= 0)
        throw std::invalid_argument("num_threads must be positive");
    validate_inputs(0.0, dt, max_confirm, params);
    if (k != 0 && (!answer_state || !transit_time || !v0))
        throw std::invalid_argument("nonempty arrays must not be null");
    // Validate the complete batch before touching any output or launching workers.
    for (std::size_t i = 0; i < k; ++i)
        if (!std::isfinite(v0[i]))
            throw std::invalid_argument("all v0 values must be finite");
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
                const auto result = evolve_critical_slow(v0[index], dt, max_confirm, params);
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
