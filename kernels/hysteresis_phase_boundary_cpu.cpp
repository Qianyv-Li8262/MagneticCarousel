#include "hysteresis_phase_boundary_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace {

struct phase_measurement {
    double mean;
    int status;
};

solver_params scaled_params(double sr, double somega, const solver_params &reference)
{
    const double inverse_sr = 1.0 / sr;
    const double inverse_somega_squared = 1.0 / (somega * somega);
    return {
        static_cast<real_t>(static_cast<double>(reference.alpha) * sr),
        static_cast<real_t>(static_cast<double>(reference.delta) * inverse_sr),
        static_cast<real_t>(static_cast<double>(reference.gamma_g) * inverse_somega_squared),
        static_cast<real_t>(static_cast<double>(reference.gamma) * inverse_somega_squared),
        static_cast<real_t>(static_cast<double>(reference.gamma_m) * inverse_sr * inverse_sr * inverse_somega_squared),
    };
}

phase_measurement measure_phase_rate(double sr, double somega,
                                     const phase_boundary_settings &settings,
                                     const solver_params &reference)
{
    const solver_params params = scaled_params(sr, somega, reference);
    state ball = {REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0),
                  REAL_LITERAL(0.0), REAL_LITERAL(0.0), {REAL_LITERAL(0.0), REAL_LITERAL(0.0)}};
    bool sliding = false;
    real_t slip_sign = REAL_LITERAL(0.0);
    // Match phase_initialize() in the GPU analysis kernel.  The all-zero
    // state can immediately enter sliding even though its initial slip is 0.
    const field_evaluation field_lab = evaluate_field(static_cast<real_t>(ball.phase));
    const vec2 h_ext = field_lab.field;
    const vec2 h_ext_phase_derivative = field_lab.phase_derivative;
    const vec2 h_ext_height_derivative = field_lab.height_derivative;
    const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
    const real_t E = (REAL_LITERAL(1.0) - C) / ((MU_R + REAL_LITERAL(2.0)) * (REAL_LITERAL(1.0) - XI_JA));
    const real_t normal_force = params.gamma_g -
        params.gamma * (A * (h_ext * h_ext_height_derivative) - E * (ball.q * h_ext_height_derivative));
    if (!real_isfinite(normal_force) || normal_force < REAL_LITERAL(0.0))
        return {std::numeric_limits<double>::quiet_NaN(), -1};
    const real_t magnetic_torque = -params.gamma_m * E * cross(ball.q, h_ext);
    const real_t magnetic_force = ETA * params.alpha * params.alpha * params.gamma_m *
        (E * (ball.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
    const real_t theta_acceleration = ETA / (REAL_LITERAL(1.0) + ETA) * magnetic_torque -
        magnetic_force / ((REAL_LITERAL(1.0) + ETA) * params.alpha);
    if (real_fabs(theta_acceleration - magnetic_torque) * ETA * params.alpha > MU * normal_force) {
        sliding = true;
        slip_sign = -signum(theta_acceleration - magnetic_torque);
        if (slip_sign == REAL_LITERAL(0.0))
            return {std::numeric_limits<double>::quiet_NaN(), -2};
    } else {
        project_pure_rolling(ball, params);
    }
    const std::int64_t window_start = std::max<std::int64_t>(0, settings.total_steps - settings.window_steps);
    std::int64_t sample_count = 0;
    double mean = 0.0;
    for (std::int64_t step = 0; step < settings.total_steps; ++step) {
        const int flag = fwd1step(ball, static_cast<real_t>(settings.dt), params, sliding, slip_sign);
        if (flag != 0)
            return {std::numeric_limits<double>::quiet_NaN(), -flag};
        const std::int64_t completed = step + 1;
        if (completed < window_start || (completed - window_start) % settings.sample_stride != 0)
            continue;
        const double phase_rate = 1.0 - static_cast<double>(ball.xdot) - static_cast<double>(ball.theta_dot);
        if (!std::isfinite(phase_rate))
            return {std::numeric_limits<double>::quiet_NaN(), -3};
        ++sample_count;
        mean += (phase_rate - mean) / static_cast<double>(sample_count);
    }
    return sample_count >= 2 ? phase_measurement{mean, 0}
                             : phase_measurement{std::numeric_limits<double>::quiet_NaN(), -4};
}

phase_boundary_point bisect_one(double sr, const phase_boundary_settings &settings,
                                const solver_params &reference)
{
    phase_boundary_point result{sr, settings.somega_lower, settings.somega_upper,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(), 0, 0};
    phase_measurement lower_measure = measure_phase_rate(sr, result.lower, settings, reference);
    phase_measurement upper_measure = measure_phase_rate(sr, result.upper, settings, reference);
    if (lower_measure.status != 0 || upper_measure.status != 0) {
        result.status = lower_measure.status != 0 ? lower_measure.status : upper_measure.status;
        return result;
    }
    auto locked = [&](double mean) { return std::abs(mean) < settings.phase_rate_threshold; };
    bool lower_locked = locked(lower_measure.mean);
    bool upper_locked = locked(upper_measure.mean);
    if (lower_locked == upper_locked) {
        result.lower_mean_phase_rate = lower_measure.mean;
        result.upper_mean_phase_rate = upper_measure.mean;
        result.status = 1;
        return result;
    }
    while (result.upper - result.lower > settings.tolerance && result.iterations < settings.max_iterations) {
        const double midpoint = result.lower + (result.upper - result.lower) / 2.0;
        const phase_measurement middle_measure = measure_phase_rate(sr, midpoint, settings, reference);
        if (middle_measure.status != 0) {
            result.status = middle_measure.status;
            return result;
        }
        ++result.iterations;
        if (locked(middle_measure.mean) == lower_locked) {
            result.lower = midpoint;
            lower_measure = middle_measure;
            lower_locked = locked(middle_measure.mean);
        } else {
            result.upper = midpoint;
            upper_measure = middle_measure;
            upper_locked = locked(middle_measure.mean);
        }
    }
    result.lower_mean_phase_rate = lower_measure.mean;
    result.upper_mean_phase_rate = upper_measure.mean;
    return result;
}

} // namespace

void bisect_phase_boundaries_cpu(phase_boundary_point *output, std::size_t count,
                                 double sr_lower, double sr_upper,
                                 const phase_boundary_settings &settings,
                                 const solver_params &reference, int num_threads)
{
    if (count == 0)
        return;
    const int requested = num_threads > 0 ? num_threads : static_cast<int>(std::thread::hardware_concurrency());
    const int worker_count = std::max(1, std::min<int>(requested, static_cast<int>(count)));
    std::atomic<std::size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= count)
                return;
            const double fraction = count == 1 ? 0.0 : static_cast<double>(index) / static_cast<double>(count - 1);
            const double sr = sr_lower + (sr_upper - sr_lower) * fraction;
            output[index] = bisect_one(sr, settings, reference);
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int i = 0; i < worker_count; ++i)
        workers.emplace_back(worker);
    for (auto &thread : workers)
        thread.join();
}
