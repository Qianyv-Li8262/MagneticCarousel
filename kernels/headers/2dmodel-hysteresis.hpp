#pragma once
#include "hysteresis_config.hpp"
#include "hysteresis_config_field.hpp"
#include "hysteresis_types.hpp"

struct FieldForceResult {
    Vec3 B;
    Vec3 Force;
};

struct CubicCoeff {
    real_t a, b, c, d; // if cache mode is caching endpoint u should interpret them as f_left, f_right, M_left, M_right
};

struct FieldCache {
    real_t left;
    real_t right;
    CubicCoeff A_coeffs[kFieldFourierModeCount];
    CubicCoeff Z_coeffs[kFieldFourierModeCount];
};

DEVICE INLINE FieldForceResult eval_mag_field_and_force(const SolverParams &params, const Vec2 &pos, const Vec3 &q,
                                                        const real_t down_disk_angle, const FieldCache *cache = nullptr)
{
    Vec3 B = {0};
    real_t nablaB[5] = {0};
    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
        real_t s = REAL_LITERAL(1.0);
        constexpr int R_magnet = magnet_number / 2;
        for (int i = 0; i < magnet_number; i++) {
            Vec2 dk = pos - make_Vec2(R_magnet * real_cos(static_cast<real_t>(i) * magnet_phi0 + down_disk_angle),
                                      R_magnet * real_sin(static_cast<real_t>(i) * magnet_phi0 + down_disk_angle));
            real_t rhoksq = dk * dk;
            real_t qk = rhoksq + height * height;
            real_t invqk = REAL_LITERAL(1.0) / (qk);
            real_t sqrtinvqk = real_sqrt(invqk);
            B = B + s * invqk * invqk * sqrtinvqk *
                        make_Vec3(REAL_LITERAL(3.0) * height * dk.x, REAL_LITERAL(3.0) * height * dk.y,
                                  REAL_LITERAL(2.0) * height * height - rhoksq);
            real_t overall = REAL_LITERAL(3.0) * s * invqk * invqk * invqk * sqrtinvqk;
            nablaB[0] += overall * height * (qk - REAL_LITERAL(5.0) * dk.x * dk.x);
            nablaB[1] -= overall * REAL_LITERAL(5.0) * height * dk.x * dk.y;
            nablaB[2] += overall * dk.x * (rhoksq  -REAL_LITERAL(4.0) * height * height);
            nablaB[3] += overall * height * (qk - REAL_LITERAL(5.0) * dk.y * dk.y);
            nablaB[4] += overall * dk.y * (rhoksq  -REAL_LITERAL(4.0) * height * height);
            s *= -1;
        }
    }
    B=B*Beta;
    for (int i = 0; i < 5; i++) {
        nablaB[i] *= Beta;
    }
    const real_t E_ = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    const real_t D_ = (Mu_r - REAL_LITERAL(1.0)) / (Mu_r + REAL_LITERAL(2.0));
    const real_t A_ = -D_ * params.gamma_m * params.alpha * params.alpha * Eta;
    const real_t B_ = E_ * params.gamma_m * params.alpha * params.alpha * Eta;
}