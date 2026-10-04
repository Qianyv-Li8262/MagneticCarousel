// SPDX-License-Identifier: AGPL-3.0-only
#include "field_lut_common.hpp"
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

namespace {
using namespace field_lut;
struct radial_function
{
    mp_real f;
    mp_real df;
    mp_real d2f;
};

mp_real hypergeometric_2f1(const mp_real &a, const mp_real &b, const mp_real &c, const mp_real &x)
{
    return boost::math::hypergeometric_pFq({a, b}, {c}, x);
}

mp_real mode_pochhammer_over_factorial(const int harmonic)
{
    const mp_real three_halves("1.5");
    mp_real result = 1;
    for (int j = 0; j < harmonic; ++j)
        result *= (three_halves + j) / (j + 1);
    return result;
}

radial_function evaluate_radial_function(const int harmonic, const mp_real &t,
                                         const mp_real &pochhammer_over_factorial)
{
    const mp_real m = harmonic;
    const mp_real three_halves("1.5");
    const mp_real p = m + three_halves;
    const mp_real a = p;
    const mp_real b = three_halves;
    const mp_real c = m + 1;
    const mp_real x = t * t;

    const mp_real F0 = hypergeometric_2f1(a, b, c, x);
    const mp_real F1 = hypergeometric_2f1(a + 1, b + 1, c + 1, x);
    const mp_real F2 = hypergeometric_2f1(a + 2, b + 2, c + 2, x);
    const mp_real ab_over_c = a * b / c;
    const mp_real shifted_ab_over_c = (a + 1) * (b + 1) / (c + 1);

    const mp_real t_to_p = pow(t, p);
    const mp_real f = pochhammer_over_factorial * t_to_p * F0;
    const mp_real df = pochhammer_over_factorial *
                       (p * pow(t, p - 1) * F0 +
                        2 * ab_over_c * pow(t, p + 1) * F1);
    const mp_real d2f = pochhammer_over_factorial *
                        (p * (p - 1) * pow(t, p - 2) * F0 +
                         2 * ab_over_c * (2 * p + 1) * t_to_p * F1 +
                         4 * ab_over_c * shifted_ab_over_c * pow(t, p + 2) * F2);
    return {f, df, d2f};
}

radial_sample evaluate_mode(const int harmonic, const mp_real &rho, const mp_real &z, const mp_real &R,
                            const int magnets, const mp_real &pochhammer_over_factorial)
{
    const mp_real chi = (z * z + rho * rho + R * R) / (2 * rho * R);
    const mp_real discriminant = sqrt(chi * chi - 1);
    const mp_real t = 1 / (chi + discriminant);

    const mp_real dchi_drho = (rho * rho - z * z - R * R) / (2 * rho * rho * R);
    const mp_real dchi_dz = z / (rho * R);
    const mp_real d2chi_drho_dz = -z / (rho * rho * R);
    const mp_real dt_drho = -t * dchi_drho / discriminant;
    const mp_real dt_dz = -t * dchi_dz / discriminant;
    const mp_real d2t_drho_dz = dchi_drho * dchi_dz /
                                    (discriminant * discriminant * discriminant) -
                                t * d2chi_drho_dz / discriminant;

    const radial_function radial = evaluate_radial_function(harmonic, t, pochhammer_over_factorial);
    const mp_real rho_to_three_halves = rho * sqrt(rho);
    const mp_real R_to_three_halves = R * sqrt(R);
    const mp_real scale = 2 * magnets / (R_to_three_halves * rho_to_three_halves);

    const mp_real A = scale * z * radial.f;
    const mp_real dA_drho = scale * z *
                            (radial.df * dt_drho - 3 * radial.f / (2 * rho));
    const mp_real Z = scale * (radial.f + z * radial.df * dt_dz);
    const mp_real dZ_drho = scale *
                            (radial.df * dt_drho - 3 * radial.f / (2 * rho) +
                             z * (radial.d2f * dt_drho * dt_dz + radial.df * d2t_drho_dz -
                                  3 * radial.df * dt_dz / (2 * rho)));
    return {A, dA_drho, Z, dZ_drho};
}

}

int main(int argc, char **argv)
{
    return field_lut::generate_lut(argc, argv, [](int harmonic, const field_lut::mp_real &) {
        const auto factor = mode_pochhammer_over_factorial(harmonic);
        return [harmonic, factor](const field_lut::mp_real &rho, const field_lut::mp_real &z,
                                 const field_lut::mp_real &R, int magnets) {
            return evaluate_mode(harmonic, rho, z, R, magnets, factor);
        };
    });
}
