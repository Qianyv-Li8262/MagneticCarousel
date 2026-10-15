// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "scan_initial_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void update_mean_variance(double value, std::int64_t count, double &mean, double &m2)
{
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    m2 += delta * (value - mean);
}

int scan_one(double v0, double s0, const initial_scan_settings &settings,
             const solver_params &params, double *out)
{
    if (!std::isfinite(v0) || !std::isfinite(s0))
        return 4;
    const double omega0 = (s0 - v0) / static_cast<double>(params.alpha);
    if (!std::isfinite(omega0))
        return 4;

    state ball = {REAL_LITERAL(0.0), REAL_LITERAL(0.0), v0, REAL_LITERAL(0.0), omega0,
                  REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)}};
    bool sliding = false;
    real_t slip_sign = REAL_LITERAL(0.0);
    try {
        // Reuse the CPU trajectory API's exact initialization and contact-mode choice.
        ball = evolve_with_discrete_mode(ball, settings.dt, 0, params, sliding, slip_sign, true);
    } catch (const std::exception &) {
        return 1;
    }

    const auto window_start = settings.steps - settings.window_steps;
    const auto split_step = window_start + settings.window_steps / 2;
    std::int64_t count = 0, first_count = 0, second_count = 0, sliding_count = 0;
    double mean_v = 0, m2_v = 0, mean_omega = 0, mean_phase = 0, m2_phase = 0;
    double mean_abs_slip = 0, mean_abs_q = 0;
    double first_mean_v = 0, second_mean_v = 0;
    double first_mean_phase = 0, second_mean_phase = 0;
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    double global_min_v = static_cast<double>(ball.xdot);
    std::int64_t global_min_step = 0;
    bool velocity_recovered = false;

    for (std::int64_t step = 0; step < settings.steps; ++step) {
        const int flag = fwd1step(ball, settings.dt, params, sliding, slip_sign);
        if (flag != 0)
            return flag;
        if (!real_isfinite(ball.t) || !real_isfinite(ball.x) || !real_isfinite(ball.xdot) ||
            !real_isfinite(ball.theta) || !real_isfinite(ball.theta_dot) ||
            !real_isfinite(ball.phase) || !real_isfinite(ball.q.x) ||
            !real_isfinite(ball.q.y))
            return 3;
        const auto completed = step + 1;
        const double v = static_cast<double>(ball.xdot);
        if (v < global_min_v) {
            global_min_v = v;
            global_min_step = completed;
            velocity_recovered = false;
        } else if (v > global_min_v) {
            velocity_recovered = true;
        }
        if (completed < window_start || (completed - window_start) % settings.sample_stride != 0)
            continue;

        const double omega = static_cast<double>(ball.theta_dot);
        const double phase = 1.0 - v - omega;
        const double slip = v + static_cast<double>(params.alpha) * omega;
        const double qx = static_cast<double>(ball.q.x);
        const double qy = static_cast<double>(ball.q.y);
        const double abs_q = std::hypot(qx, qy);
        if (!std::isfinite(v) || !std::isfinite(omega) || !std::isfinite(phase) ||
            !std::isfinite(slip) || !std::isfinite(abs_q))
            return 3;

        ++count;
        update_mean_variance(v, count, mean_v, m2_v);
        update_mean_variance(phase, count, mean_phase, m2_phase);
        mean_omega += (omega - mean_omega) / static_cast<double>(count);
        mean_abs_slip += (std::abs(slip) - mean_abs_slip) / static_cast<double>(count);
        mean_abs_q += (abs_q - mean_abs_q) / static_cast<double>(count);
        sliding_count += sliding ? 1 : 0;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        if (completed < split_step) {
            ++first_count;
            first_mean_v += (v - first_mean_v) / static_cast<double>(first_count);
            first_mean_phase += (phase - first_mean_phase) / static_cast<double>(first_count);
        } else {
            ++second_count;
            second_mean_v += (v - second_mean_v) / static_cast<double>(second_count);
            second_mean_phase += (phase - second_mean_phase) / static_cast<double>(second_count);
        }
    }

    if (count < 2 || first_count == 0 || second_count == 0)
        return 4;
    out[0] = mean_v;
    out[1] = std::sqrt(m2_v / static_cast<double>(count - 1));
    out[2] = mean_omega;
    out[3] = mean_phase;
    out[4] = std::sqrt(m2_phase / static_cast<double>(count - 1));
    out[5] = mean_abs_slip;
    out[6] = static_cast<double>(sliding_count) / static_cast<double>(count);
    out[7] = mean_abs_q;
    out[8] = static_cast<double>(ball.xdot);
    out[9] = static_cast<double>(ball.theta_dot);
    out[10] = 1.0 - out[8] - out[9];
    out[11] = std::hypot(static_cast<double>(ball.q.x), static_cast<double>(ball.q.y));
    out[12] = second_mean_v - first_mean_v;
    out[13] = second_mean_phase - first_mean_phase;
    out[14] = min_v;
    out[15] = max_v;
    out[16] = static_cast<double>(count);
    out[17] = static_cast<double>(global_min_step);
    out[18] = velocity_recovered ? 1.0 : 0.0;
    return 0;
}

} // namespace

void scan_initial_grid_cpu(const double *v0_values, std::size_t v0_count,
                           const double *s0_values, std::size_t s0_count,
                           const initial_scan_settings &settings,
                           std::int32_t *status, double *metrics)
{
    if (v0_count == 0 || s0_count == 0 || v0_count > std::numeric_limits<std::size_t>::max() / s0_count)
        throw std::invalid_argument("initial scan requires nonempty grids without size overflow");
    if (!std::isfinite(settings.sr) || settings.sr <= 0 ||
        !std::isfinite(settings.somega) || settings.somega <= 0 ||
        !std::isfinite(settings.dt) || settings.dt <= 0 ||
        settings.steps < 2 || settings.window_steps < 2 ||
        settings.window_steps > settings.steps || settings.sample_stride < 1 ||
        settings.num_threads < 0)
        throw std::invalid_argument("invalid initial scan settings");

    const double inverse_sr = 1.0 / settings.sr;
    const double inverse_somega_squared = 1.0 / (settings.somega * settings.somega);
    const solver_params params = {
        settings.reference.alpha * settings.sr,
        settings.reference.delta * inverse_sr,
        settings.reference.gamma_g * inverse_somega_squared,
        settings.reference.gamma * inverse_somega_squared,
        settings.reference.gamma_m * inverse_sr * inverse_sr * inverse_somega_squared,
    };
    const std::size_t total = v0_count * s0_count;
    const unsigned automatic = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t workers = std::min(total, static_cast<std::size_t>(
        settings.num_threads > 0 ? settings.num_threads : automatic));
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    auto worker = [&] {
        while (true) {
            const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= total)
                break;
            double *out = metrics + initial_scan_metric_count * index;
            std::fill_n(out, initial_scan_metric_count, std::numeric_limits<double>::quiet_NaN());
            status[index] = scan_one(v0_values[index % v0_count], s0_values[index / v0_count],
                                     settings, params, out);
        }
    };
    for (std::size_t i = 0; i < workers; ++i)
        pool.emplace_back(worker);
    for (auto &thread : pool)
        thread.join();
}
