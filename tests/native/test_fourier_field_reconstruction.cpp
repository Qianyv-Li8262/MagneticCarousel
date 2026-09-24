#include "2dmodel-hysteresis.hpp"

#include <array>
#include <cmath>
#include <cstdlib>

namespace
{

bool close(const real_t actual, const real_t expected)
{
    return std::abs(actual - expected) <= REAL_LITERAL(2e-4) * (REAL_LITERAL(1.0) + std::abs(expected));
}

} // namespace

bool check_reconstruction(const real_t rho, FieldCache &cache)
{
    constexpr real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
    constexpr real_t rho_min = kLUTLowerBoundRatio * radius;
    constexpr real_t drho =
        (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * radius / static_cast<real_t>(kSplineNodes - 1);

    std::array<LUTNode, kFieldFourierModeCount * kSplineNodes> lut{};
    for (int node = 0; node < kSplineNodes; ++node) {
        const real_t rho = rho_min + static_cast<real_t>(node) * drho;
        // A(rho) = 0.7 - 0.2 rho + 0.03 rho^2 + 0.004 rho^3
        // Z(rho) = -0.4 + 0.11 rho - 0.006 rho^2
        lut[node] = {
            REAL_LITERAL(0.7) - REAL_LITERAL(0.2) * rho + REAL_LITERAL(0.03) * rho * rho +
                REAL_LITERAL(0.004) * rho * rho * rho,
            REAL_LITERAL(0.06) + REAL_LITERAL(0.024) * rho,
            REAL_LITERAL(-0.4) + REAL_LITERAL(0.11) * rho - REAL_LITERAL(0.006) * rho * rho,
            REAL_LITERAL(-0.012),
        };
    }

    const real_t phi = REAL_LITERAL(0.61);
    const real_t disk_angle = REAL_LITERAL(-0.23);
    const Vec2 pos = make_Vec2(rho * real_cos(phi), rho * real_sin(phi));
    const Vec3 q = make_Vec3(REAL_LITERAL(0.17), REAL_LITERAL(-0.31), REAL_LITERAL(0.26));
    const SolverParams params = {REAL_LITERAL(0.7), REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0),
                                 REAL_LITERAL(0.8)};
    const FieldForceResult actual = eval_mag_field_and_force(params, pos, q, disk_angle, &cache, lut.data());

    constexpr real_t m = static_cast<real_t>(magnet_number / 2);
    const real_t relative_angle = m * (phi - disk_angle);
    const real_t c = real_cos(relative_angle);
    const real_t s = real_sin(relative_angle);
    const real_t cp = real_cos(phi);
    const real_t sp = real_sin(phi);
    const real_t inv_rho = REAL_LITERAL(1.0) / rho;
    const real_t inv_rho2 = inv_rho * inv_rho;

    const real_t A = REAL_LITERAL(0.7) - REAL_LITERAL(0.2) * rho + REAL_LITERAL(0.03) * rho * rho +
                     REAL_LITERAL(0.004) * rho * rho * rho;
    const real_t dA = REAL_LITERAL(-0.2) + REAL_LITERAL(0.06) * rho + REAL_LITERAL(0.012) * rho * rho;
    const real_t d2A = REAL_LITERAL(0.06) + REAL_LITERAL(0.024) * rho;
    const real_t Z = REAL_LITERAL(-0.4) + REAL_LITERAL(0.11) * rho - REAL_LITERAL(0.006) * rho * rho;
    const real_t dZ = REAL_LITERAL(0.11) - REAL_LITERAL(0.012) * rho;

    const real_t h_rho = -dA * c;
    const real_t h_phi = m * A * inv_rho * s;
    const real_t h_z = -Z * c;
    const Vec3 h = Beta * make_Vec3(cp * h_rho - sp * h_phi, sp * h_rho + cp * h_phi, h_z);

    const real_t Grr = -d2A * c;
    const real_t Grp = m * (dA * inv_rho - A * inv_rho2) * s;
    const real_t Grz = -dZ * c;
    const real_t Gpp = (m * m * A * inv_rho2 - dA * inv_rho) * c;
    const real_t Gpz = m * Z * inv_rho * s;
    const real_t Gxx = Beta * (cp * cp * Grr - REAL_LITERAL(2.0) * cp * sp * Grp + sp * sp * Gpp);
    const real_t Gxy = Beta * (cp * sp * (Grr - Gpp) + (cp * cp - sp * sp) * Grp);
    const real_t Gxz = Beta * (cp * Grz - sp * Gpz);
    const real_t Gyy = Beta * (sp * sp * Grr + REAL_LITERAL(2.0) * cp * sp * Grp + cp * cp * Gpp);
    const real_t Gyz = Beta * (sp * Grz + cp * Gpz);

    const real_t E = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    const real_t D = (Mu_r - REAL_LITERAL(1.0)) / (Mu_r + REAL_LITERAL(2.0));
    const real_t force_h_coefficient = D * params.gamma_m * params.alpha * params.alpha * Eta;
    const real_t force_q_coefficient = -E * params.gamma_m * params.alpha * params.alpha * Eta;
    const Vec3 v = force_h_coefficient * h + force_q_coefficient * q;
    const Vec3 force = make_Vec3(Gxx * v.x + Gxy * v.y + Gxz * v.z, Gxy * v.x + Gyy * v.y + Gyz * v.z,
                                 Gxz * v.x + Gyz * v.y + (-Gxx - Gyy) * v.z);

    return close(actual.h.x, h.x) && close(actual.h.y, h.y) && close(actual.h.z, h.z) &&
                   close(actual.Force.x, force.x) && close(actual.Force.y, force.y) && close(actual.Force.z, force.z);
}

int main()
{
    // Cache-enabled builds require a valid cache, just like the trajectory solver.
    FieldCache cache{};
    // Cold cache, a different point in the same interval, another interval, then back.
    // Every call uses the same synthetic LUT values and an independent analytic reference.
    for (const real_t rho : {REAL_LITERAL(3.7), REAL_LITERAL(3.701), REAL_LITERAL(4.2), REAL_LITERAL(3.7)}) {
        if (!check_reconstruction(rho, cache))
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
