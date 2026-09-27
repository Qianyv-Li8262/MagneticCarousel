// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

// The model header defines short parameter macros, so pybind11 must be parsed first.
// clang-format off
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "1dmodel-hysteresis-dump.hpp"
// clang-format on

#include <cstdint>

namespace py = pybind11;
namespace dump = hysteresis_dump_core;

state dump_state_from_numpy(const py::array_t<real_t, py::array::c_style | py::array::forcecast> &initial)
{
    if (initial.ndim() != 1 || initial.shape(0) != 8)
        throw py::value_error("initial must be a one-dimensional array of length 8");
    const auto in = initial.unchecked<1>();
    return {in(0), in(1), in(2), in(3), in(4), in(5), {in(6), in(7)}};
}

py::array_t<real_t> dump_state_to_numpy(const state &value)
{
    py::array_t<real_t> result(8);
    auto out = result.mutable_unchecked<1>();
    out(0) = value.t;
    out(1) = value.x;
    out(2) = value.xdot;
    out(3) = value.theta;
    out(4) = value.theta_dot;
    out(5) = value.phase;
    out(6) = value.q.x;
    out(7) = value.q.y;
    return result;
}

solver_params dump_params_from_python(const py::object &value)
{
    return {py::cast<real_t>(value.attr("alpha")), py::cast<real_t>(value.attr("delta")),
            py::cast<real_t>(value.attr("gamma_g")), py::cast<real_t>(value.attr("gamma")),
            py::cast<real_t>(value.attr("gamma_m"))};
}

template <typename T, typename Record, typename Getter>
py::array_t<T> dump_array(const std::vector<Record> &records, Getter getter)
{
    py::array_t<T> result(records.size());
    auto out = result.template mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(records.size()); ++i)
        out(i) = static_cast<T>(getter(records[static_cast<std::size_t>(i)]));
    return result;
}

py::dict dump_steps_to_python(const std::vector<dump::step_record> &records)
{
    py::dict out;
#define STEP_FIELD(type, name) out[#name] = dump_array<type>(records, [](const auto &r) { return r.name; })
    STEP_FIELD(std::int64_t, step);
    STEP_FIELD(real_t, time);
    STEP_FIELD(real_t, integrator_t);
    STEP_FIELD(real_t, x);
    STEP_FIELD(real_t, xdot);
    STEP_FIELD(real_t, theta);
    STEP_FIELD(real_t, theta_dot);
    STEP_FIELD(real_t, phase);
    STEP_FIELD(real_t, qx);
    STEP_FIELD(real_t, qy);
    STEP_FIELD(real_t, qnorm);
    STEP_FIELD(bool, sliding);
    STEP_FIELD(real_t, slip_sign);
    STEP_FIELD(std::int64_t, rhs_count);
    STEP_FIELD(std::int64_t, zero_branch_count);
    STEP_FIELD(std::int64_t, x_branch_count);
    STEP_FIELD(std::int64_t, branch_switch_count);
    STEP_FIELD(real_t, min_relu_test);
    STEP_FIELD(real_t, max_relu_test);
    STEP_FIELD(real_t, closest_relu_test);
    STEP_FIELD(std::int8_t, last_branch);
    STEP_FIELD(std::int64_t, sliding_to_sticking_count);
    STEP_FIELD(std::int64_t, sliding_reverse_count);
    STEP_FIELD(std::int64_t, sticking_to_sliding_count);
#undef STEP_FIELD
    return out;
}

py::dict dump_overview_to_python(const std::vector<dump::overview_record> &records)
{
    py::dict out;
#define OVERVIEW_FIELD(type, name) out[#name] = dump_array<type>(records, [](const auto &r) { return r.name; })
    OVERVIEW_FIELD(std::int64_t, step);
    OVERVIEW_FIELD(real_t, time);
    OVERVIEW_FIELD(real_t, integrator_t);
    OVERVIEW_FIELD(real_t, x);
    OVERVIEW_FIELD(real_t, xdot);
    OVERVIEW_FIELD(real_t, theta);
    OVERVIEW_FIELD(real_t, theta_dot);
    OVERVIEW_FIELD(real_t, phase);
    OVERVIEW_FIELD(real_t, qx);
    OVERVIEW_FIELD(real_t, qy);
    OVERVIEW_FIELD(real_t, qnorm);
    OVERVIEW_FIELD(bool, sliding);
    OVERVIEW_FIELD(real_t, slip_sign);
#undef OVERVIEW_FIELD
    return out;
}

py::dict dump_branch_changes_to_python(const std::vector<dump::branch_change_record> &records)
{
    py::dict out;
#define CHANGE_FIELD(type, name) out[#name] = dump_array<type>(records, [](const auto &r) { return r.name; })
    CHANGE_FIELD(std::int64_t, step);
    CHANGE_FIELD(real_t, step_time);
    CHANGE_FIELD(real_t, t_stage);
    CHANGE_FIELD(real_t, qx);
    CHANGE_FIELD(real_t, qy);
    CHANGE_FIELD(real_t, relu_test);
    CHANGE_FIELD(std::int8_t, previous_branch);
    CHANGE_FIELD(std::int8_t, new_branch);
    CHANGE_FIELD(std::int8_t, role);
    CHANGE_FIELD(std::int8_t, stage);
    CHANGE_FIELD(bool, sliding);
#undef CHANGE_FIELD
    return out;
}

py::dict dump_rhs_to_python(const std::vector<dump::rhs_record> &records)
{
    py::dict out;
#define RHS_FIELD(type, name) out[#name] = dump_array<type>(records, [](const auto &r) { return r.name; })
    RHS_FIELD(std::int64_t, step);
    RHS_FIELD(real_t, step_time);
    RHS_FIELD(real_t, t_stage);
    RHS_FIELD(real_t, qx);
    RHS_FIELD(real_t, qy);
    RHS_FIELD(real_t, qdotx);
    RHS_FIELD(real_t, qdoty);
    RHS_FIELD(real_t, h0_dot_x);
    RHS_FIELD(real_t, h0_dot_y);
    RHS_FIELD(real_t, he_dot_test_x);
    RHS_FIELD(real_t, he_dot_test_y);
    RHS_FIELD(real_t, relu_test);
    RHS_FIELD(std::int8_t, branch);
    RHS_FIELD(std::int8_t, role);
    RHS_FIELD(std::int8_t, stage);
    RHS_FIELD(bool, sliding);
    RHS_FIELD(bool, accepted);
#undef RHS_FIELD
    return out;
}

py::dict dump_trace_to_python(const dump::trace_result &trace)
{
    py::dict out;
    out["enabled"] = trace.enabled;
    out["truncated"] = trace.truncated;
    out["overview"] = dump_overview_to_python(trace.overview);
    out["steps"] = dump_steps_to_python(trace.steps);
    out["branch_changes"] = dump_branch_changes_to_python(trace.branch_changes);
    out["rhs"] = dump_rhs_to_python(trace.rhs);
    return out;
}

py::tuple dump_evolve_numpy(const py::array_t<real_t, py::array::c_style | py::array::forcecast> &initial,
                            const real_t dt, const int steps, const py::object &params_object, const bool dump_enabled,
                            const std::int64_t dump_every, const std::int64_t overview_every,
                            const std::int64_t start_step, const std::int64_t end_step, const bool log_branch_changes,
                            const bool log_rhs_stages, const bool log_rejected_trials, const std::int64_t max_records)
{
    const state initial_state = dump_state_from_numpy(initial);
    const solver_params params = dump_params_from_python(params_object);

    if (!dump_enabled) {
        bool sliding;
        state final_state;
        {
            py::gil_scoped_release release;
            final_state = evolve_with_mode(initial_state, dt, steps, params, sliding);
        }
        dump::trace_result empty_trace;
        return py::make_tuple(dump_state_to_numpy(final_state), sliding, dump_trace_to_python(empty_trace));
    }

    const dump::trace_options options{true,       dump_every,         overview_every, start_step,
                                      end_step,   log_branch_changes, log_rhs_stages, log_rejected_trials,
                                      max_records};
    dump::evolution_result result;
    {
        py::gil_scoped_release release;
        result = dump::evolve(initial_state, dt, steps, params, options);
    }
    return py::make_tuple(dump_state_to_numpy(result.final_state), result.sliding, dump_trace_to_python(result.trace));
}

PYBIND11_MODULE(hysteresis_dump, m)
{
    m.doc() = "CPU-only traced evolution for the one-dimensional hysteresis model";
    m.attr("real_dtype") = py::dtype::of<real_t>();
    m.attr("BRANCH_ZERO") = static_cast<std::int8_t>(dump::relu_branch::zero);
    m.attr("BRANCH_X") = static_cast<std::int8_t>(dump::relu_branch::x);
    m.attr("ROLE_FULL_TRIAL") = static_cast<std::int8_t>(dump::rk_role::full_trial);
    m.attr("ROLE_PRE_EVENT") = static_cast<std::int8_t>(dump::rk_role::pre_event);
    m.attr("ROLE_POST_EVENT") = static_cast<std::int8_t>(dump::rk_role::post_event);
    m.attr("STAGE_START") = static_cast<std::int8_t>(dump::rk_stage::start);
    m.attr("STAGE_MIDPOINT") = static_cast<std::int8_t>(dump::rk_stage::midpoint);

    py::class_<solver_params>(m, "SolverParams", py::module_local())
        .def(py::init([](const real_t alpha, const real_t delta, const real_t gamma_g, const real_t gamma,
                         const real_t gamma_m) { return solver_params{alpha, delta, gamma_g, gamma, gamma_m}; }),
             py::arg("alpha"), py::arg("delta"), py::arg("gamma_g"), py::arg("gamma"), py::arg("gamma_m"))
        .def_readwrite("alpha", &solver_params::alpha)
        .def_readwrite("delta", &solver_params::delta)
        .def_readwrite("gamma_g", &solver_params::gamma_g)
        .def_readwrite("gamma", &solver_params::gamma)
        .def_readwrite("gamma_m", &solver_params::gamma_m);

    m.def("evolve", &dump_evolve_numpy, py::arg("initial"), py::arg("dt"), py::arg("steps"), py::arg("params"),
          py::arg("dump") = false, py::arg("dump_every") = 1000, py::arg("overview_every") = 10000,
          py::arg("start_step") = 0, py::arg("end_step") = -1, py::arg("log_branch_changes") = true,
          py::arg("log_rhs_stages") = false, py::arg("log_rejected_trials") = false, py::arg("max_records") = 2000000,
          "Return (final_state, sliding, trace). dump=False calls the original numerical core. "
          "dump=True returns a whole-run overview sampled every overview_every steps and a detailed window sampled "
          "from start_step every dump_every steps, including both window boundaries and block ReLU statistics. "
          "Accepted/rejected RK stages are optional. ReLU branch 0 is the zero branch; 1 is the x branch.");
}
