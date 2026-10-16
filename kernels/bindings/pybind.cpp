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
#include "hysteresis_config.hpp"
#include "state_io.hpp"
#include "scan_initial_cpu.hpp"

namespace py = pybind11;

#ifndef MODULE_NAME
#define MODULE_NAME hysteresis
#endif

State<kEddyCurrentModeCount> evolve_trajectory(const State<kEddyCurrentModeCount> &initial, real_t dt, int steps, const SolverParams &params,
                        bool &sliding, Vec2 &sliding_dir, bool initialize, std::int64_t start_step);
std::array<real_t, 3> magnetic_mechanical_power(const State<kEddyCurrentModeCount> &state, const SolverParams &params);
State<kEddyCurrentModeCount> magnetize_held_ball(const State<kEddyCurrentModeCount> &initial, real_t dt, int steps, const SolverParams &params);

// [disk angle, pos x/y, speed x/y, quaternion w/x/y/z,
//  omega x/y/z, hysteresis q x/y/z]. Time is tracked by Python.
constexpr py::ssize_t state_size = kSerializedStateSize;

State<kEddyCurrentModeCount> state_from_numpy(const py::array_t<real_t, py::array::c_style | py::array::forcecast> &input)
{
    if (input.ndim() != 1 || (input.shape(0) != state_size && input.shape(0) != 15))
        throw py::value_error("state must have length " + std::to_string(state_size) + " (or legacy length 15)");
    const auto a = input.unchecked<1>();
    for (py::ssize_t i = 0; i < input.shape(0); ++i)
        if (!std::isfinite(static_cast<double>(a(i))))
            throw py::value_error("state must contain finite values");
    if (input.shape(0) == 15)
        return read_state_values<false>(input.data());
    return read_state_values<kEnableEddyCurrents>(input.data());
}

py::array_t<real_t> state_to_numpy(const State<kEddyCurrentModeCount> &s)
{
    py::array_t<real_t> result(state_size);
    write_state_values<kEnableEddyCurrents>(s, result.mutable_data());
    return result;
}

py::tuple evolve_numpy(const py::array_t<real_t, py::array::c_style | py::array::forcecast> &initial,
                       real_t dt, int steps, const SolverParams &params,
                       const py::object &mode, const py::object &direction, std::int64_t start_step)
{
    const bool initialize = mode.is_none() && direction.is_none();
    if (mode.is_none() != direction.is_none())
        throw py::value_error("sliding and sliding_dir must both be supplied or both be None");
    State<kEddyCurrentModeCount> state = state_from_numpy(initial);
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
                             real_t sr, real_t somega,
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
    for (std::size_t i = 0; i < cells * state_size; ++i) states_ptr[i] = REAL_LITERAL(0.0);
    {
        py::gil_scoped_release release;
        scan_initial_grid_cpu(v0.data(), s0.data(), static_cast<std::size_t>(k),
                              dt, steps, window_steps, sample_stride, sr, somega,
                              num_threads, status_ptr, metrics_ptr, states_ptr);
    }
    return py::make_tuple(std::move(status), std::move(metrics), std::move(states));
}

PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "CPU trajectories and initial-state scan for the two-dimensional hysteresis model";
    m.attr("real_dtype") = py::dtype::of<real_t>();
    m.attr("state_size") = state_size;
    m.attr("eddy_currents_enabled") = kEnableEddyCurrents;
    m.attr("eddy_mode_count") = 4 * kEddyCurrentModeCount;
    m.attr("magnet_number") = magnet_number;
    m.attr("field_mode") = kFieldEvalMode == FieldEvalMode::kExact ? "exact" : "fourier-spline";
    py::class_<SolverParams>(m, "SolverParams")
        .def(py::init([](real_t sr, real_t somega) {
            if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) ||
                !real_isfinite(somega) || somega <= REAL_LITERAL(0.0))
                throw py::value_error("sr and somega must be finite and positive");
            const auto p = make_solver_params(magnet_number, sr, somega, Eta);
            if (!real_isfinite(p.alpha) || p.alpha <= REAL_LITERAL(0.0) ||
                !real_isfinite(p.delta) || !real_isfinite(p.gamma_g) ||
                !real_isfinite(p.gamma) || !real_isfinite(p.gamma_m) ||
                !real_isfinite(p.eddy_current_scale) || p.eddy_current_scale <= REAL_LITERAL(0.0))
                throw py::value_error("sr/somega exceed the solver's numeric range");
            return p;
        }), py::arg("sr") = REAL_LITERAL(1.0), py::arg("somega") = REAL_LITERAL(1.0))
        .def_readonly("alpha", &SolverParams::alpha)
        .def_readonly("delta", &SolverParams::delta)
        .def_readonly("gamma_g", &SolverParams::gamma_g)
        .def_readonly("gamma", &SolverParams::gamma)
        .def_readonly("gamma_m", &SolverParams::gamma_m)
        .def_readonly("eddy_current_scale", &SolverParams::eddy_current_scale)
        .def_readwrite("reverse_push_acceleration", &SolverParams::reverse_push_acceleration,
                       "Following reverse tangential COM acceleration; zero disables the external push.");
    m.attr("reference_ball_radius_m") = hysteresis_physics::Rref;
    m.attr("reference_disk_omega_rad_s") = hysteresis_physics::omega_ref;
    m.def("physical_units", [](double sr, double somega) {
        if (!std::isfinite(sr) || sr <= 0 || !std::isfinite(somega) || somega <= 0)
            throw py::value_error("sr and somega must be finite and positive");
        const double radius = hysteresis_physics::Rref * sr;
        const double mass = hysteresis_physics::rho_b * (4.0 * hysteresis_physics::pi / 3.0) * radius * radius * radius;
        const double omega = hysteresis_physics::omega_ref * somega;
        const double length = hysteresis_physics::L / (magnet_number / 2.0);
        if (!std::isfinite(mass) || mass <= 0 || !std::isfinite(omega) || omega <= 0)
            throw py::value_error("physical unit conversion exceeds numeric range");
        py::dict units;
        units["ball_radius_m"] = radius;
        units["ball_mass_kg"] = mass;
        units["disk_omega_rad_s"] = omega;
        units["length_unit_m"] = length;
        return units;
    }, py::arg("sr") = 1.0, py::arg("somega") = 1.0);
    m.def("push_acceleration", [](double force_mn, double sr, double somega) {
        if (!std::isfinite(force_mn) || force_mn < 0 ||
            !std::isfinite(sr) || sr <= 0 || !std::isfinite(somega) || somega <= 0)
            throw py::value_error("invalid push force or runtime scales");
        const double r = hysteresis_physics::Rref * sr;
        const double mass = hysteresis_physics::rho_b * 4.0 * hysteresis_physics::pi / 3.0 * r * r * r;
        const double omega = hysteresis_physics::omega_ref * somega;
        const double length = hysteresis_physics::L / (magnet_number / 2.0);
        const real_t acceleration = static_cast<real_t>(force_mn * 1e-3 / mass / length / (omega * omega));
        if (!real_isfinite(acceleration)) throw py::value_error("push acceleration exceeds numeric range");
        return acceleration;
    }, py::arg("force_mn"), py::arg("sr"), py::arg("somega"));
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
        const State<kEddyCurrentModeCount> initial = state_from_numpy(input);
        State<kEddyCurrentModeCount> result;
        {
            py::gil_scoped_release release;
            result = magnetize_held_ball(initial, dt, steps, params);
        }
        return state_to_numpy(result);
    }, py::arg("initial"), py::arg("dt"), py::arg("steps"), py::arg("params"),
       "Hold position and attitude fixed, zero velocities, and evolve disk phase, JA and optional eddy memory.");
    m.def("scan_initial_grid", &scan_initial_numpy,
          py::arg("v0").noconvert(), py::arg("s0").noconvert(),
          py::arg("dt"), py::arg("steps"), py::arg("window_steps"), py::arg("sample_stride"),
          py::arg("sr"), py::arg("somega"),
          py::arg("num_threads") = 0,
          "Scan an initial (v0,S0) grid with C++ worker threads; return status, metrics, states. "
          "v0 and s0 must be contiguous arrays with the module's real_dtype. "
          "num_threads=0 uses hardware concurrency.");
}
