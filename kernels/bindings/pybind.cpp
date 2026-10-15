// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <array>

#include "hysteresis_config_field.hpp"
#include "hysteresis_types.hpp"
#include "scan_initial_cpu.hpp"

namespace py = pybind11;

#ifndef MODULE_NAME
#define MODULE_NAME hysteresis
#endif

State evolve_trajectory(const State &initial, real_t dt, int steps, const SolverParams &params,
                        bool &sliding, Vec2 &sliding_dir, bool initialize, std::int64_t start_step);
std::array<real_t, 3> magnetic_mechanical_power(const State &state, const SolverParams &params);
State magnetize_held_ball(const State &initial, real_t dt, int steps, const SolverParams &params);

// [disk angle, pos x/y, speed x/y, quaternion w/x/y/z,
//  omega x/y/z, hysteresis q x/y/z]. Time is tracked by Python.
constexpr py::ssize_t state_size = 15;

State state_from_numpy(const py::array_t<real_t, py::array::c_style | py::array::forcecast> &input)
{
    if (input.ndim() != 1 || input.shape(0) != state_size)
        throw py::value_error("state must be a one-dimensional array of length 15");
    const auto a = input.unchecked<1>();
    for (py::ssize_t i = 0; i < state_size; ++i)
        if (!std::isfinite(static_cast<double>(a(i))))
            throw py::value_error("state must contain finite values");
    return {a(0), {a(1), a(2)}, {a(3), a(4)},
            {a(5), {a(6), a(7), a(8)}}, {a(9), a(10), a(11)},
            {a(12), a(13), a(14)}};
}

py::array_t<real_t> state_to_numpy(const State &s)
{
    py::array_t<real_t> result(state_size);
    auto a = result.mutable_unchecked<1>();
    a(0) = static_cast<real_t>(s.down_disk_angle);
    a(1) = static_cast<real_t>(s.pos.x);
    a(2) = static_cast<real_t>(s.pos.y);
    a(3) = static_cast<real_t>(s.speed.x);
    a(4) = static_cast<real_t>(s.speed.y);
    a(5) = static_cast<real_t>(s.attitude.real);
    a(6) = static_cast<real_t>(s.attitude.imagine.x);
    a(7) = static_cast<real_t>(s.attitude.imagine.y);
    a(8) = static_cast<real_t>(s.attitude.imagine.z);
    a(9) = static_cast<real_t>(s.omega.x);
    a(10) = static_cast<real_t>(s.omega.y);
    a(11) = static_cast<real_t>(s.omega.z);
    a(12) = static_cast<real_t>(s.q.x);
    a(13) = static_cast<real_t>(s.q.y);
    a(14) = static_cast<real_t>(s.q.z);
    return result;
}

py::tuple evolve_numpy(const py::array_t<real_t, py::array::c_style | py::array::forcecast> &initial,
                       real_t dt, int steps, const SolverParams &params,
                       const py::object &mode, const py::object &direction, std::int64_t start_step)
{
    const bool initialize = mode.is_none() && direction.is_none();
    if (mode.is_none() != direction.is_none())
        throw py::value_error("sliding and sliding_dir must both be supplied or both be None");
    State state = state_from_numpy(initial);
    bool sliding = false;
    Vec2 sliding_dir{REAL_LITERAL(0.0), REAL_LITERAL(0.0)};
    if (!initialize) {
        sliding = py::cast<bool>(mode);
        auto dir = py::cast<py::array_t<real_t, py::array::c_style | py::array::forcecast>>(direction);
        if (dir.ndim() != 1 || dir.shape(0) != 2)
            throw py::value_error("sliding_dir must be a one-dimensional array of length 2");
        sliding_dir = {dir.at(0), dir.at(1)};
    }
    {
        py::gil_scoped_release release;
        state = evolve_trajectory(state, dt, steps, params, sliding, sliding_dir, initialize, start_step);
    }
    py::array_t<real_t> direction_out(2);
    direction_out.mutable_at(0) = sliding_dir.x;
    direction_out.mutable_at(1) = sliding_dir.y;
    return py::make_tuple(state_to_numpy(state), sliding, std::move(direction_out));
}

py::tuple scan_initial_numpy(const py::array_t<real_t, py::array::c_style> &v0,
                             const py::array_t<real_t, py::array::c_style> &s0,
                             real_t dt, int steps, int window_steps, int sample_stride,
                             real_t sr, real_t somega, const SolverParams &reference,
                             int num_threads)
{
    if (v0.ndim() != 1 || s0.ndim() != 1 || v0.size() != s0.size() ||
        v0.size() < 1 || v0.size() > 46340)
        throw py::value_error("v0 and s0 must be contiguous 1D arrays of equal length (1..46340)");
    if (num_threads < 0 || steps < 1 || window_steps < 2 || window_steps > steps ||
        sample_stride < 1 || !real_isfinite(dt) || dt <= REAL_LITERAL(0.0) ||
        !real_isfinite(sr) || sr <= REAL_LITERAL(0.0) ||
        !real_isfinite(somega) || somega <= REAL_LITERAL(0.0))
        throw py::value_error("invalid scan time, scale, or thread setting");
    const py::ssize_t k = v0.size();
    py::array_t<std::int32_t> status({k, k});
    py::array_t<real_t> metrics({k, k, py::ssize_t(20)});
    py::array_t<real_t> states({k, k, state_size});
    auto *status_ptr = status.mutable_data();
    auto *metrics_ptr = metrics.mutable_data();
    auto *states_ptr = states.mutable_data();
    const std::size_t cells = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
    for (std::size_t i = 0; i < cells; ++i) status_ptr[i] = -2;
    for (std::size_t i = 0; i < cells * 20; ++i)
        metrics_ptr[i] = std::numeric_limits<real_t>::quiet_NaN();
    for (std::size_t i = 0; i < cells * 15; ++i) states_ptr[i] = REAL_LITERAL(0.0);
    {
        py::gil_scoped_release release;
        scan_initial_grid_cpu(v0.data(), s0.data(), static_cast<std::size_t>(k),
                              dt, steps, window_steps, sample_stride, sr, somega,
                              reference, num_threads, status_ptr, metrics_ptr, states_ptr);
    }
    return py::make_tuple(std::move(status), std::move(metrics), std::move(states));
}

PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "CPU trajectories and initial-state scan for the two-dimensional hysteresis model";
    m.attr("real_dtype") = py::dtype::of<real_t>();
    m.attr("state_size") = state_size;
    m.attr("magnet_number") = magnet_number;
    m.attr("field_mode") = kFieldEvalMode == FieldEvalMode::kExact ? "exact" : "fourier-spline";
    py::class_<SolverParams>(m, "SolverParams")
        .def(py::init<real_t, real_t, real_t, real_t, real_t>(),
             py::arg("alpha"), py::arg("delta"), py::arg("gamma_g"),
             py::arg("gamma"), py::arg("gamma_m"))
        .def_readwrite("alpha", &SolverParams::alpha)
        .def_readwrite("delta", &SolverParams::delta)
        .def_readwrite("gamma_g", &SolverParams::gamma_g)
        .def_readwrite("gamma", &SolverParams::gamma)
        .def_readwrite("gamma_m", &SolverParams::gamma_m)
        .def_readwrite("reverse_push_acceleration", &SolverParams::reverse_push_acceleration,
                       "Following reverse tangential COM acceleration; zero disables the external push.");
    m.def("evolve_trajectory", &evolve_numpy, py::arg("initial"), py::arg("dt"),
          py::arg("steps"), py::arg("params"), py::arg("sliding") = py::none(),
          py::arg("sliding_dir") = py::none(), py::arg("start_step") = 0,
          "Advance one trajectory chunk and return (state, sliding, sliding_dir). "
          "Pass the returned mode and direction into the next chunk.");
    m.def("magnetic_power", [](const py::array_t<real_t, py::array::c_style | py::array::forcecast> &input,
                               const SolverParams &params) {
        const auto values = magnetic_mechanical_power(state_from_numpy(input), params);
        return py::make_tuple(values[0], values[1], values[2]);
    }, py::arg("state"), py::arg("params"),
       "Return magnetic (translational, rotational, total) mechanical power per unit mass in dimensionless 2D units.");
    m.def("magnetize_held_ball", [](const py::array_t<real_t, py::array::c_style | py::array::forcecast> &input,
                                  real_t dt, int steps, const SolverParams &params) {
        const State initial = state_from_numpy(input);
        State result;
        {
            py::gil_scoped_release release;
            result = magnetize_held_ball(initial, dt, steps, params);
        }
        return state_to_numpy(result);
    }, py::arg("initial"), py::arg("dt"), py::arg("steps"), py::arg("params"),
       "Hold position and attitude fixed, zero velocities, and evolve only disk phase and JA memory.");
    m.def("scan_initial_grid", &scan_initial_numpy,
          py::arg("v0").noconvert(), py::arg("s0").noconvert(),
          py::arg("dt"), py::arg("steps"), py::arg("window_steps"), py::arg("sample_stride"),
          py::arg("sr"), py::arg("somega"), py::arg("reference_params"),
          py::arg("num_threads") = 0,
          "Scan an initial (v0,S0) grid with C++ worker threads; return status, metrics, states. "
          "v0 and s0 must be contiguous arrays with the module's real_dtype. "
          "num_threads=0 uses hardware concurrency.");
}
