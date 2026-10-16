// SPDX-License-Identifier: AGPL-3.0-only
#include "hysteresis_config.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char **)
{
#ifdef USE_FP64
    constexpr double tolerance = 1e-13;
#else
    constexpr double tolerance = 2e-6;
#endif
    const auto close = [](double actual, double expected) {
        return std::abs(actual - expected) <= tolerance * std::abs(expected);
    };
    // Independent legacy Python reference (8 magnets, 12 mm ball, 100 rpm).
    const auto reference = make_solver_params(8, REAL_LITERAL(1.0), REAL_LITERAL(1.0), Eta);
    assert(close(static_cast<double>(reference.alpha), 0.342857142857143));
    assert(close(static_cast<double>(reference.delta), 8.02199442675159e-5));
    assert(close(static_cast<double>(reference.gamma_g), 5.11179845930400));
    assert(close(static_cast<double>(reference.gamma), 17302.6849037817));
    assert(close(static_cast<double>(reference.gamma_m), 367982.448040497));
    assert(reference.eddy_current_scale == REAL_LITERAL(1.0));
    assert(reference.reverse_push_acceleration == REAL_LITERAL(0.0));

    std::cout << std::setprecision(17);
    for (int count : {4, 8, 16}) {
        for (double sr : {0.5, 1.0, 1.3}) {
            for (double somega : {0.1, 1.0, 2.0, 5.0}) {
                const auto p = make_solver_params(count, static_cast<real_t>(sr), static_cast<real_t>(somega), Eta);
                assert(close(static_cast<double>(p.gamma), static_cast<double>(Eta * p.alpha * p.alpha * p.gamma_m)));
                assert(close(static_cast<double>(p.eddy_current_scale), somega * sr * sr));
                assert(p.reverse_push_acceleration == REAL_LITERAL(0.0));
                // Optional CSV output for comparison with the unmodified Python factory.
                if (argc > 1) {
                    std::cout << count << ',' << sr << ',' << somega;
                    for (real_t value : std::array<real_t, 6>{p.alpha, p.delta, p.gamma_g, p.gamma,
                                                            p.gamma_m, p.eddy_current_scale})
                        std::cout << ',' << static_cast<double>(value);
                    std::cout << '\n';
                }
            }
        }
    }
}
