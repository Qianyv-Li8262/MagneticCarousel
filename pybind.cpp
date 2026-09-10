#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "1dmodel-hysteresis.hpp"

namespace py = pybind11;

py::array_t<float> evolve_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &initial, const float dt, const int steps,
    const solver_params &params)
{
    if (initial.ndim() != 1 || initial.shape(0) != 8)
        throw py::value_error("initial must be a one-dimensional array of length 8");

    const auto in = initial.unchecked<1>();
    const state initial_state = {in(0), in(1), in(2), in(3), in(4), in(5), {in(6), in(7)}};
    const state final_state = evolve(initial_state, dt, steps, params);
    py::array_t<float> result(8);
    auto out = result.mutable_unchecked<1>();
    out(0) = final_state.t;
    out(1) = final_state.x;
    out(2) = final_state.xdot;
    out(3) = final_state.theta;
    out(4) = final_state.theta_dot;
    out(5) = final_state.phase;
    out(6) = final_state.q.x;
    out(7) = final_state.q.y;
    return result;
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
}
