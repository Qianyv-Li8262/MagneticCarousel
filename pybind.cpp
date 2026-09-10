#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "1dmodel-hysteresis.hpp"

namespace py = pybind11;

state evolve_with_mode(const state &initial, float dt, int steps, const solver_params &params, bool &sliding);

state state_from_numpy(const py::array_t<float, py::array::c_style | py::array::forcecast> &initial)
{
    if (initial.ndim() != 1 || initial.shape(0) != 8)
        throw py::value_error("initial must be a one-dimensional array of length 8");

    const auto in = initial.unchecked<1>();
    return {in(0), in(1), in(2), in(3), in(4), in(5), {in(6), in(7)}};
}

py::array_t<float> state_to_numpy(const state &value)
{
    py::array_t<float> result(8);
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

py::array_t<float> evolve_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &initial, const float dt, const int steps,
    const solver_params &params)
{
    return state_to_numpy(evolve(state_from_numpy(initial), dt, steps, params));
}

py::tuple evolve_with_mode_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &initial, const float dt, const int steps,
    const solver_params &params)
{
    bool sliding;
    const state final_state = evolve_with_mode(state_from_numpy(initial), dt, steps, params, sliding);
    return py::make_tuple(state_to_numpy(final_state), sliding);
}

PYBIND11_MODULE(hysteresis, m)
{
    m.doc() = "CPU evolution for the one-dimensional hysteresis model";
    py::class_<solver_params>(m, "SolverParams")
        .def(py::init([](const float alpha, const float delta, const float gamma_g, const float gamma,
                         const float gamma_m) { return solver_params{alpha, delta, gamma_g, gamma, gamma_m}; }),
             py::arg("alpha"), py::arg("delta"), py::arg("gamma_g"), py::arg("gamma"), py::arg("gamma_m"))
        .def_readwrite("alpha", &solver_params::alpha)
        .def_readwrite("delta", &solver_params::delta)
        .def_readwrite("gamma_g", &solver_params::gamma_g)
        .def_readwrite("gamma", &solver_params::gamma)
        .def_readwrite("gamma_m", &solver_params::gamma_m);

    m.def("evolve", &evolve_numpy, py::arg("initial"), py::arg("dt"), py::arg("steps"), py::arg("params"),
          "Evolve [t, x, xdot, theta, theta_dot, phase, qx, qy] and return the final float32 state.");
    m.def("evolve_with_mode", &evolve_with_mode_numpy, py::arg("initial"), py::arg("dt"), py::arg("steps"),
          py::arg("params"), "Evolve the float32 state and return (state, sliding).");
}
