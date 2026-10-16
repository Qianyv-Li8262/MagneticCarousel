// SPDX-License-Identifier: AGPL-3.0-only
#include "eddy_coefficients.hpp"
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>

int main()
{
    namespace c = eddy_coefficients;
    assert(c::mode_count==4*kEddyCurrentModeCount);
    assert(reinterpret_cast<std::uintptr_t>(c::aj)%32==0);
    assert(reinterpret_cast<std::uintptr_t>(c::inv_tau_ref_star)%32==0);
    const double mu=Mu_r;
    const double td=hysteresis_physics::mu0*mu*hysteresis_physics::sigma*
                    hysteresis_physics::Rref*hysteresis_physics::Rref;
    double slope=0,maximum=0;
    for (int j=0;j<c::mode_count;++j) {
        assert(std::isfinite(c::aj[j]) && c::aj[j]>=0);
        assert(std::isfinite(c::inv_tau_ref_star[j]) && c::inv_tau_ref_star[j]>0);
        slope+=c::aj[j]/(c::reference_omega_rad_s*c::inv_tau_ref_star[j]);
    }
    const double expected_slope=.9*hysteresis_physics::mu0*hysteresis_physics::sigma*
        hysteresis_physics::Rref*hysteresis_physics::Rref*std::pow(mu/(mu+2),2);
    assert(std::abs(slope/expected_slope-1)<.001);
    // Independent FP64 implementation of the analytic sphere response; this
    // uses std::complex trigonometry rather than the generator's real formulas.
    for (int i=0;i<1025;++i) {
        const double f=c::fit_min_hz*std::pow(c::fit_max_hz/c::fit_min_hz,i/1024.);
        const double omega=2*hysteresis_physics::pi*f;
        const std::complex<double> x=std::sqrt(std::complex<double>(0,-omega*td));
        const auto j0=std::sin(x)/x;
        const auto j1=(std::sin(x)-x*std::cos(x))/(x*x);
        const auto exact=1.5*((2*mu+1)*j1-x*j0)/((mu-1)*j1+x*j0)-3*(mu-1)/(mu+2);
        std::complex<double> predicted{};
        for (int j=0;j<c::mode_count;++j) {
            const double tau=1/(c::reference_omega_rad_s*c::inv_tau_ref_star[j]);
            const auto s_tau=std::complex<double>(0,omega*tau);
            predicted+=c::aj[j]*(-s_tau)/(1.+s_tau);
            // Runtime scale and timestep enter only the exponent argument.
            const double scale=2*.5*.5;
            const double dt=.0005;
            assert(std::abs(dt*c::inv_tau_ref_star[j]/scale - dt/(scale*c::reference_omega_rad_s*tau))<1e-12);
        }
        maximum=std::max(maximum,std::abs(predicted-exact)/std::abs(exact));
    }
    assert(maximum<.001);
    std::cout << "Independent 1025-frequency maximum relative error: " << maximum << '\n';
}
