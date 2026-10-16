// SPDX-License-Identifier: AGPL-3.0-only
#include "hysteresis_config.hpp"
#include "eddy_codegen_config.hpp"
#include "eddy_codegen_math.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
using namespace eddy_codegen;
namespace settings = eddy_codegen_config;

Real frequency_at(int i, int count)
{
    const Real lo(settings::frequency_min_hz), hi(settings::frequency_max_hz);
    return lo*exp(log(hi/lo)*Real(i)/Real(count-1));
}

void write_if_changed(const std::filesystem::path &path, const std::string &contents)
{
    if (std::filesystem::exists(path)) {
        std::ifstream input(path, std::ios::binary);
        const std::string old((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (old==contents) return;
    }
    std::filesystem::create_directories(path.parent_path());
    const auto temporary=std::filesystem::path(path.string()+".tmp");
    {
        std::ofstream output(temporary, std::ios::binary);
        output.exceptions(std::ios::failbit|std::ios::badbit);
        output << contents;
    }
    std::filesystem::rename(temporary,path);
}

void emit_array(std::ostream &out, const char *name, const Vector &values)
{
    out << "IYPT16_EDDY_COEFFICIENT_ARRAY " << name << "[mode_count] = {\n";
    for (const auto &v:values) out << "    " << v << ",\n";
    out << "};\n\n";
}
} // namespace

int main(int argc, char **argv)
try {
    if (argc!=3) throw std::runtime_error("usage: generate_eddy_coefficients OUTPUT_HEADER OUTPUT_REPORT");
    const int count=4*kEddyCurrentModeCount;
    const int slow=settings::retained_slow_modes;
    if (count<=slow+1 || slow<1 || settings::fit_samples<2 || settings::validation_samples<2 ||
        settings::frequency_min_hz<=0 || settings::frequency_max_hz<=settings::frequency_min_hz ||
        settings::fastest_tau_ratio<=0 || settings::fastest_tau_ratio>=1 || settings::maximum_relative_error<=0)
        throw std::runtime_error("invalid eddy codegen configuration");
    // Reuse production constants; multiprecision arithmetic does not invent
    // precision beyond the original physical input definitions.
    const Real mu(Mu_r),sigma(hysteresis_physics::sigma),radius(hysteresis_physics::Rref);
    const Real omega_ref(hysteresis_physics::omega_ref);
    const Real diffusion=Real(hysteresis_physics::mu0)*mu*sigma*radius*radius;
    const Real two_pi=2*boost::math::constants::pi<Real>();
    Vector roots,physical_tau,physical_weight;
    for (int n=1;n<=slow;++n) {
        const Real q=physical_root(n,mu);
        roots.push_back(q);
        physical_tau.push_back(diffusion/(q*q));
        physical_weight.push_back(9*mu/(q*q+(mu-1)*(mu+2)));
    }
    Vector tau=physical_tau;
    const Real first_fast=physical_tau.back()/2;
    const Real last_fast=physical_tau.front()*Real(settings::fastest_tau_ratio);
    if (last_fast>=first_fast) throw std::runtime_error("eddy fast-mode interval is empty");
    for (int i=0;i<count-slow;++i)
        tau.push_back(first_fast*exp(log(last_fast/first_fast)*Real(i)/Real(count-slow-1)));

    Matrix a(2*settings::fit_samples,Vector(count));
    Vector b(2*settings::fit_samples);
    for (int i=0;i<settings::fit_samples;++i) {
        const Real omega=two_pi*frequency_at(i,settings::fit_samples);
        const Complex target=response(omega,mu,diffusion);
        const Real scale=1/norm(target);
        b[i]=target.real()*scale;
        b[i+settings::fit_samples]=target.imag()*scale;
        for (int j=0;j<count;++j) {
            const Real r=omega*tau[j];
            a[i][j]=-r*r/(1+r*r)*scale;
            a[i+settings::fit_samples][j]=-r/(1+r*r)*scale;
        }
    }
    const Vector weights=nnls(a,b);
    Vector inverse_tau_star,rounded_tau,rounded_weights;
    int active=0;
    for (int j=0;j<count;++j) {
        if (weights[j]<0 || tau[j]<=0) throw std::runtime_error("nonphysical compressed eddy mode");
        if (weights[j]>0) ++active;
        inverse_tau_star.push_back(1/(omega_ref*tau[j]));
        rounded_weights.push_back(Real(static_cast<double>(weights[j])));
        rounded_tau.push_back(1/(omega_ref*Real(static_cast<double>(inverse_tau_star[j]))));
    }
    Real maximum_error=0,low_frequency_slope=0;
    for (int i=0;i<settings::validation_samples;++i) {
        const Real omega=two_pi*frequency_at(i,settings::validation_samples);
        const Complex exact=response(omega,mu,diffusion);
        const Real error=norm(compressed_response(omega,rounded_tau,rounded_weights)-exact)/norm(exact);
        if (error>maximum_error) maximum_error=error;
    }
    for (int j=0;j<count;++j) low_frequency_slope+=rounded_weights[j]*rounded_tau[j];
    const Real expected_slope=Real("0.9")*Real(hysteresis_physics::mu0)*sigma*radius*radius*mu*mu/((mu+2)*(mu+2));
    const Real slope_error=abs(low_frequency_slope/expected_slope-1);
    if (maximum_error>Real(settings::maximum_relative_error) || slope_error>Real(settings::maximum_relative_error))
        throw std::runtime_error("compressed eddy model failed frequency/slope validation; increase mode count or adjust fit range");

    std::ostringstream header;
    header << std::scientific << std::setprecision(40);
    header << "// Generated by generate_eddy_coefficients; do not edit.\n#pragma once\n\n"
              "namespace eddy_coefficients {\ninline constexpr int mode_count = " << count << ";\n"
              "inline constexpr int avx_group_count = " << kEddyCurrentModeCount << ";\n"
              "inline constexpr double mu_r = " << mu << ";\n"
              "inline constexpr double sigma_si = " << sigma << ";\n"
              "inline constexpr double reference_radius_m = " << radius << ";\n"
              "inline constexpr double reference_omega_rad_s = " << omega_ref << ";\n"
              "inline constexpr double fit_min_hz = " << Real(settings::frequency_min_hz) << ";\n"
              "inline constexpr double fit_max_hz = " << Real(settings::frequency_max_hz) << ";\n"
              "inline constexpr double validated_max_relative_error = " << maximum_error << ";\n\n"
              "#ifdef __CUDACC__\n#define IYPT16_EDDY_COEFFICIENT_ARRAY static __device__ __constant__ const double\n"
              "#else\n#define IYPT16_EDDY_COEFFICIENT_ARRAY alignas(32) inline constexpr double\n#endif\n\n"
              "// delta_m_body = sum_j aj[j] * y_body[j]; aj is dimensionless.\n";
    emit_array(header,"aj",weights);
    header << "// Reference inverse tau*: runtime inverse tau* = value / eddy_current_scale.\n"
              "// r_j = dt_star * inv_tau_ref_star[j] / eddy_current_scale.\n";
    emit_array(header,"inv_tau_ref_star",inverse_tau_star);
    header << "#undef IYPT16_EDDY_COEFFICIENT_ARRAY\n} // namespace eddy_coefficients\n";

    std::ostringstream report;
    report << std::scientific << std::setprecision(30)
           << "{\n  \"precision_decimal_digits\": 100,\n  \"scalar_modes\": " << count
           << ",\n  \"active_modes\": " << active << ",\n  \"avx_groups\": " << kEddyCurrentModeCount
           << ",\n  \"mu_r\": " << mu << ",\n  \"sigma_si\": " << sigma
           << ",\n  \"reference_radius_m\": " << radius << ",\n  \"reference_omega_rad_s\": " << omega_ref
           << ",\n  \"fit_min_hz\": " << Real(settings::frequency_min_hz)
           << ",\n  \"fit_max_hz\": " << Real(settings::frequency_max_hz)
           << ",\n  \"fit_samples\": " << settings::fit_samples
           << ",\n  \"validation_samples\": " << settings::validation_samples
           << ",\n  \"max_relative_error_after_fp64_rounding\": " << maximum_error
           << ",\n  \"low_frequency_slope_relative_error\": " << slope_error
           << ",\n  \"first_physical_root\": " << roots.front()
           << ",\n  \"first_physical_weight\": " << physical_weight.front()
           << ",\n  \"first_physical_tau_seconds\": " << physical_tau.front() << "\n}\n";
    // Validate before replacing either existing artifact.
    write_if_changed(argv[1],header.str());
    write_if_changed(argv[2],report.str());
    std::cout << "Eddy codegen: " << count << " modes (" << active << " active), max relative error="
              << maximum_error << ", slope error=" << slope_error << '\n';
    return 0;
} catch (const std::exception &error) {
    std::cerr << "Eddy codegen failed: " << error.what() << '\n';
    return 1;
}
