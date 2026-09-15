#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "1dmodel-hysteresis.hpp"
#if defined(USE_FP64) || defined(USE_DD)
#include "hysteresis_critical_slow_cpu.hpp"
#endif
#ifdef USE_FP64
#include "hysteresis_phase_boundary_cpu.hpp"
#endif

namespace py = pybind11;

#ifndef MODULE_NAME
#define MODULE_NAME hysteresis
#endif

#ifdef USE_DD
using binding_real_t = double;

real_t real_from_python(const py::handle &value)
{
    if (py::isinstance<py::str>(value))
        return dd_real::from_string(py::cast<std::string>(value).c_str());
    return real_t(py::cast<double>(value));
}
#else
using binding_real_t = real_t;
#endif

#ifdef USE_DD
py::tuple calc_slow_cpu_dd(const py::sequence &scan_values, const py::object &fixed_scale,
                           const critical_slow_scan_mode mode, const py::object &dt,
                           const std::int64_t max_confirm, const solver_params reference,
                           const int num_threads)
{
    const auto k = static_cast<std::size_t>(py::len(scan_values));
    std::vector<real_t> parsed;
    parsed.reserve(k);
    for (const py::handle value : scan_values)
        parsed.push_back(real_from_python(value));
    const real_t parsed_fixed = real_from_python(fixed_scale);
    const real_t parsed_dt = real_from_python(dt);
    py::array_t<std::int32_t> answer(k);
    py::array_t<double> times(k);
    {
        py::gil_scoped_release release;
        calc_slow_cpu(answer.mutable_data(), times.mutable_data(), parsed.data(), k,
                      parsed_fixed, mode, parsed_dt, max_confirm, reference, num_threads);
    }
    return py::make_tuple(answer, times);
}
#endif

#ifdef USE_FP64
void calc_slow_cpu_numpy(py::array_t<std::int32_t, py::array::c_style> answer,
                         py::array_t<double, py::array::c_style> times,
                         py::array_t<double, py::array::c_style> scan_values,
                         double fixed_scale, critical_slow_scan_mode mode, double dt,
                         std::int64_t max_confirm, solver_params reference, int num_threads)
{
    if (answer.ndim() != 1 || times.ndim() != 1 || scan_values.ndim() != 1 ||
        answer.size() != scan_values.size() || times.size() != scan_values.size())
        throw py::value_error("answer_state, transit_time and scan_values must be 1D arrays of equal length");
    if (!answer.writeable() || !times.writeable())
        throw py::value_error("output arrays must be writable");
    const auto overlaps = [](const py::array &a, const py::array &b) {
        const auto start_a = reinterpret_cast<std::uintptr_t>(a.data());
        const auto start_b = reinterpret_cast<std::uintptr_t>(b.data());
        return a.nbytes() && b.nbytes() &&
               (start_a <= start_b ? start_b - start_a < static_cast<std::uintptr_t>(a.nbytes())
                                   : start_a - start_b < static_cast<std::uintptr_t>(b.nbytes()));
    };
    if (overlaps(answer, times) || overlaps(answer, scan_values) || overlaps(times, scan_values))
        throw py::value_error("input and output arrays must not overlap");
    auto *answer_ptr = answer.mutable_data();
    auto *time_ptr = times.mutable_data();
    const auto *scan_ptr = scan_values.data();
    if (reinterpret_cast<std::uintptr_t>(answer_ptr) % alignof(std::int32_t) ||
        reinterpret_cast<std::uintptr_t>(time_ptr) % alignof(double) ||
        reinterpret_cast<std::uintptr_t>(scan_ptr) % alignof(double))
        throw py::value_error("arrays must be naturally aligned");
    const auto k = static_cast<std::size_t>(scan_values.size());
    py::gil_scoped_release release;
    calc_slow_cpu(answer_ptr, time_ptr, scan_ptr, k, fixed_scale, mode, dt, max_confirm,
                  reference, num_threads);
}

py::dict bisect_phase_boundaries_numpy(double sr_lower, double sr_upper, std::size_t plot_points,
                                       double somega_lower, double somega_upper, double tolerance,
                                       double threshold, double dt, double total_time,
                                       double observation_window, double sample_time,
                                       int max_iterations, solver_params reference, int num_threads)
{
    const auto finite_positive = [](double value) { return std::isfinite(value) && value > 0.0; };
    if (!finite_positive(sr_lower) || !finite_positive(sr_upper) || sr_lower > sr_upper)
        throw py::value_error("sr bounds must be positive, finite and increasing");
    if (plot_points == 0)
        throw py::value_error("plot_points must be positive");
    if (!finite_positive(somega_lower) || !finite_positive(somega_upper) || somega_lower >= somega_upper)
        throw py::value_error("somega bounds must be positive, finite and increasing");
    if (!finite_positive(tolerance) || !finite_positive(threshold) || !finite_positive(dt) ||
        !finite_positive(total_time) || !finite_positive(observation_window) || !finite_positive(sample_time))
        throw py::value_error("tolerance, threshold and time settings must be positive and finite");
    if (observation_window > total_time || max_iterations < 1 || num_threads < 0)
        throw py::value_error("invalid observation window, iteration count or thread count");
    const auto aligned_steps = [dt](double duration, const char *name) {
        const double ratio = duration / dt;
        const auto steps = static_cast<std::int64_t>(std::llround(ratio));
        const double error = std::abs(duration - static_cast<double>(steps) * dt);
        const double allowance = std::max(1e-12, 64.0 * std::numeric_limits<double>::epsilon() * duration);
        if (steps < 1 || error > allowance)
            throw py::value_error(std::string(name) + " must be an integer multiple of dt");
        return steps;
    };
    phase_boundary_settings settings{
        somega_lower, somega_upper, tolerance, threshold, dt,
        aligned_steps(total_time, "total_time"),
        aligned_steps(observation_window, "observation_window"),
        aligned_steps(sample_time, "sample_time"), max_iterations,
    };
    if (settings.window_steps < 2 * settings.sample_stride)
        throw py::value_error("observation_window must contain at least two sample intervals");

    std::vector<phase_boundary_point> results(plot_points);
    {
        py::gil_scoped_release release;
        bisect_phase_boundaries_cpu(results.data(), results.size(), sr_lower, sr_upper,
                                    settings, reference, num_threads);
    }
    py::array_t<double> sr(plot_points), lower(plot_points), upper(plot_points), midpoint(plot_points);
    py::array_t<double> lower_mean(plot_points), upper_mean(plot_points);
    py::array_t<std::int32_t> iterations(plot_points), status(plot_points);
    for (std::size_t i = 0; i < plot_points; ++i) {
        sr.mutable_data()[i] = results[i].sr;
        lower.mutable_data()[i] = results[i].lower;
        upper.mutable_data()[i] = results[i].upper;
        midpoint.mutable_data()[i] = (results[i].lower + results[i].upper) / 2.0;
        lower_mean.mutable_data()[i] = results[i].lower_mean_phase_rate;
        upper_mean.mutable_data()[i] = results[i].upper_mean_phase_rate;
        iterations.mutable_data()[i] = results[i].iterations;
        status.mutable_data()[i] = results[i].status;
    }
    py::dict output;
    output["sr"] = std::move(sr);
    output["lower"] = std::move(lower);
    output["upper"] = std::move(upper);
    output["midpoint"] = std::move(midpoint);
    output["lower_mean_phase_rate"] = std::move(lower_mean);
    output["upper_mean_phase_rate"] = std::move(upper_mean);
    output["iterations"] = std::move(iterations);
    output["status"] = std::move(status);
    return output;
}
#endif

state state_from_numpy(const py::array_t<binding_real_t, py::array::c_style | py::array::forcecast> &initial)
{
    if (initial.ndim() != 1 || initial.shape(0) != 8)
        throw py::value_error("initial must be a one-dimensional array of length 8");

    const auto in = initial.unchecked<1>();
    return {in(0), in(1), in(2), in(3), in(4), in(5), {in(6), in(7)}};
}

py::array_t<binding_real_t> state_to_numpy(const state &value)
{
    py::array_t<binding_real_t> result(8);
    auto out = result.mutable_unchecked<1>();
    out(0) = static_cast<binding_real_t>(value.t);
    out(1) = static_cast<binding_real_t>(value.x);
    out(2) = static_cast<binding_real_t>(value.xdot);
    out(3) = static_cast<binding_real_t>(value.theta);
    out(4) = static_cast<binding_real_t>(value.theta_dot);
    out(5) = static_cast<binding_real_t>(value.phase);
    out(6) = static_cast<binding_real_t>(value.q.x);
    out(7) = static_cast<binding_real_t>(value.q.y);
    return result;
}

py::array_t<binding_real_t> evolve_numpy(
    const py::array_t<binding_real_t, py::array::c_style | py::array::forcecast> &initial,
    const binding_real_t dt, const int steps,
    const solver_params &params)
{
    return state_to_numpy(evolve(state_from_numpy(initial), real_t(dt), steps, params));
}

py::tuple evolve_with_mode_numpy(
    const py::array_t<binding_real_t, py::array::c_style | py::array::forcecast> &initial,
    const binding_real_t dt, const int steps,
    const solver_params &params)
{
    bool sliding;
    const state final_state = evolve_with_mode(state_from_numpy(initial), real_t(dt), steps, params, sliding);
    return py::make_tuple(state_to_numpy(final_state), sliding);
}

PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "CPU evolution for the one-dimensional hysteresis model";
    m.attr("real_dtype") = py::dtype::of<binding_real_t>();
#ifdef USE_DD
    m.attr("real_backend") = "double-double";
    m.attr("real_precision_bits") = 106;
    m.attr("real_decimal_digits") = 31;
    m.def("real_parts", [](const py::object &value) {
        const real_t parsed = real_from_python(value);
        return py::make_tuple(parsed.hi, parsed.lo);
    }, py::arg("value"), "Return the (hi, lo) double-double representation of a float or decimal string.");
#else
    m.attr("real_backend") = "double";
    m.attr("real_precision_bits") = std::numeric_limits<real_t>::digits;
#endif
#if defined(USE_FP64) || defined(USE_DD)
    m.attr("critical_slow_confirm_evolve_steps") = critical_slow_confirm_evolve_steps;
    m.attr("critical_slow_lock_confirm_window") = critical_slow_lock_confirm_window;
    m.attr("critical_slow_phase_rate_tolerance") = critical_slow_phase_rate_tolerance;
    py::enum_<critical_slow_scan_mode>(m, "CriticalSlowScanMode")
        .value("SOMEGA", critical_slow_scan_mode::somega)
        .value("SR", critical_slow_scan_mode::sr)
        .export_values();
#ifdef USE_FP64
    m.def("calc_slow_cpu", &calc_slow_cpu_numpy,
          py::arg("answer_state").noconvert(), py::arg("transit_time").noconvert(),
          py::arg("scan_values").noconvert(), py::arg("fixed_scale"), py::arg("mode"),
          py::arg("dt"), py::arg("max_confirm"), py::arg("reference_params"),
          py::arg("num_threads"),
          "Synchronous FP64 CPU kernel; fills int32 answer_state and float64 transit_time in place. "
          "scan_values must be positive float64 sr or somega values; the other scale is fixed. "
          "Every trajectory starts from the all-zero state. Arrays must be contiguous, aligned, "
          "non-overlapping 1D arrays. "
          "Do not access outputs or mutate/resize any arrays concurrently with this call. "
          "C++ workers dynamically claim whole trajectories using a mutex; results retain input order.");
    m.def("bisect_phase_boundaries", &bisect_phase_boundaries_numpy,
          py::arg("sr_lower"), py::arg("sr_upper"), py::arg("plot_points"),
          py::arg("somega_lower") = 0.1, py::arg("somega_upper") = 3.0,
          py::arg("tolerance") = 1e-5, py::arg("threshold") = 0.05,
          py::arg("dt") = 1e-3, py::arg("total_time") = 50000.0,
          py::arg("observation_window") = 4000.0, py::arg("sample_time") = 0.2,
          py::arg("max_iterations") = 40, py::arg("reference_params"),
          py::arg("num_threads") = 0,
          "Parallel C++ bisection of the s_omega phase boundary for evenly spaced s_r points. "
          "Classification uses the GPU phase scan's terminal-window mean of 1-v-omega. "
          "num_threads=0 uses hardware concurrency. Returns a dict of NumPy arrays; status 0 is success, "
          "1 means the supplied s_omega endpoints have the same classification, and negative values are evolution errors.");
#else
    m.def("calc_slow_cpu_dd", &calc_slow_cpu_dd,
          py::arg("scan_values"), py::arg("fixed_scale"), py::arg("mode"), py::arg("dt"),
          py::arg("max_confirm"), py::arg("reference_params"), py::arg("num_threads"),
          "Run double-double critical-slow trajectories. Decimal strings preserve all input digits; "
          "returns (int32 classifications, float64 event times).");
#endif
#ifdef USE_DD
    m.def("evolve_critical_slow", [](const py::object &dt, std::int64_t max_confirm, solver_params params) {
        critical_slow_result result;
        const real_t parsed_dt = real_from_python(dt);
        {
            py::gil_scoped_release release;
            result = evolve_critical_slow(parsed_dt, max_confirm, params);
        }
        return py::make_tuple(result.answer_state, result.transit_time);
    }, py::arg("dt"), py::arg("max_confirm"), py::arg("params"));
#else
    m.def("evolve_critical_slow", [](double dt, std::int64_t max_confirm, solver_params params) {
        critical_slow_result result;
        {
            py::gil_scoped_release release;
            result = evolve_critical_slow(dt, max_confirm, params);
        }
        return py::make_tuple(result.answer_state, result.transit_time);
    }, py::arg("dt"), py::arg("max_confirm"), py::arg("params"),
       "FP64 single all-zero trajectory. Return (classification, event time). Phase locking means the trailing 1000-time-unit mean of phase_rate=1-v-omega has absolute value below 0.05 for 2500 time units; the event time is the midpoint of the first qualifying smoothing window. "
       "Each frame contains 50000 steps and locking requires 50 consecutive qualifying frames. "
       "Classification: -1 error, 0 no confirmed phase lock, 1 phase locked.");
#endif
#endif
#ifdef USE_DD
    py::class_<solver_params>(m, "SolverParams")
        .def(py::init([](const py::object &alpha, const py::object &delta, const py::object &gamma_g,
                         const py::object &gamma, const py::object &gamma_m) {
                 return solver_params{real_from_python(alpha), real_from_python(delta), real_from_python(gamma_g),
                                      real_from_python(gamma), real_from_python(gamma_m)};
             }),
             py::arg("alpha"), py::arg("delta"), py::arg("gamma_g"), py::arg("gamma"), py::arg("gamma_m"))
        .def_property("alpha", [](const solver_params &p) { return static_cast<double>(p.alpha); },
                      [](solver_params &p, const py::object &v) { p.alpha = real_from_python(v); })
        .def_property("delta", [](const solver_params &p) { return static_cast<double>(p.delta); },
                      [](solver_params &p, const py::object &v) { p.delta = real_from_python(v); })
        .def_property("gamma_g", [](const solver_params &p) { return static_cast<double>(p.gamma_g); },
                      [](solver_params &p, const py::object &v) { p.gamma_g = real_from_python(v); })
        .def_property("gamma", [](const solver_params &p) { return static_cast<double>(p.gamma); },
                      [](solver_params &p, const py::object &v) { p.gamma = real_from_python(v); })
        .def_property("gamma_m", [](const solver_params &p) { return static_cast<double>(p.gamma_m); },
                      [](solver_params &p, const py::object &v) { p.gamma_m = real_from_python(v); });
#else
    py::class_<solver_params>(m, "SolverParams")
        .def(py::init([](const real_t alpha, const real_t delta, const real_t gamma_g, const real_t gamma,
                         const real_t gamma_m) { return solver_params{alpha, delta, gamma_g, gamma, gamma_m}; }),
             py::arg("alpha"), py::arg("delta"), py::arg("gamma_g"), py::arg("gamma"), py::arg("gamma_m"))
        .def_readwrite("alpha", &solver_params::alpha)
        .def_readwrite("delta", &solver_params::delta)
        .def_readwrite("gamma_g", &solver_params::gamma_g)
        .def_readwrite("gamma", &solver_params::gamma)
        .def_readwrite("gamma_m", &solver_params::gamma_m);
#endif

    m.def("evolve", &evolve_numpy, py::arg("initial"), py::arg("dt"), py::arg("steps"), py::arg("params"),
          "Evolve [t, x, xdot, theta, theta_dot, phase, qx, qy] and return the final state.");
    m.def("evolve_with_mode", &evolve_with_mode_numpy, py::arg("initial"), py::arg("dt"), py::arg("steps"),
          py::arg("params"), "Evolve the state and return (state, sliding).");
}
