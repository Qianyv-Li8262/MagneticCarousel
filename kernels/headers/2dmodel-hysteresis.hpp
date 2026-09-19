#pragma once
#include "hysteresis_config.hpp"
#include "hysteresis_config_field.hpp"
#include "hysteresis_types.hpp"

struct FieldForceResult {
    Vec3 h;
    Vec3 Force;
};

struct CubicCoeff {
    real_t a, b, c, d; // if cache mode is caching endpoint, u should interpret them as A_left, M_left, A_right, M_right
};

struct Spline012 {
    real_t value, d1, d2;
};

template <bool ReturnCoefficients> struct LUTInterpolation {
    Spline012 A, Z;
};

template <> struct LUTInterpolation<true> {
    Spline012 A, Z;
    CubicCoeff A_coeffs, Z_coeffs;
};

DEVICE INLINE CubicCoeff spline_coefficients(const real_t left, const real_t left_M, const real_t right,
                                             const real_t right_M, const real_t drho2)
{
    return {left, right - left - drho2 * (REAL_LITERAL(2.0) * left_M + right_M) / REAL_LITERAL(6.0),
            drho2 * left_M / REAL_LITERAL(2.0), drho2 * (right_M - left_M) / REAL_LITERAL(6.0)};
}

DEVICE INLINE Spline012 evaluate_spline(const CubicCoeff &c, const real_t u, const real_t inv_drho)
{
    const real_t d1_u = c.b + u * (REAL_LITERAL(2.0) * c.c + REAL_LITERAL(3.0) * c.d * u);
    return {c.a + u * (c.b + u * (c.c + u * c.d)), d1_u * inv_drho,
            (REAL_LITERAL(2.0) * c.c + REAL_LITERAL(6.0) * c.d * u) * inv_drho * inv_drho};
}

// Node layout: A, M_A, Z, M_Z. Derivatives are with respect to rho, not u.
template <bool ReturnCoefficients = false>
DEVICE INLINE LUTInterpolation<ReturnCoefficients>
interpolate_lut_interval(const real_t left_A, const real_t left_M_A, const real_t left_Z, const real_t left_M_Z,
                         const real_t right_A, const real_t right_M_A, const real_t right_Z, const real_t right_M_Z,
                         const real_t u)
{
    constexpr real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
    constexpr real_t drho =
        (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * radius / static_cast<real_t>(kSplineNodes - 1);
    constexpr real_t inv_drho = REAL_LITERAL(1.0) / drho;
    constexpr real_t drho2 = drho * drho;
    const CubicCoeff A = spline_coefficients(left_A, left_M_A, right_A, right_M_A, drho2);
    const CubicCoeff Z = spline_coefficients(left_Z, left_M_Z, right_Z, right_M_Z, drho2);

    LUTInterpolation<ReturnCoefficients> result{evaluate_spline(A, u, inv_drho), evaluate_spline(Z, u, inv_drho)};
    if constexpr (ReturnCoefficients) {
        result.A_coeffs = A;
        result.Z_coeffs = Z;
    }
    return result;
}

struct NodeCache {
    LUTNode left[kFieldFourierModeCount];
    LUTNode right[kFieldFourierModeCount];
};

struct CoeffCache {
    CubicCoeff A[kFieldFourierModeCount];
    CubicCoeff Z[kFieldFourierModeCount];
};

struct FieldCache {
    int interval_eff;
    union {
        NodeCache nodes;
        CoeffCache coeffs;
    };
};

struct LUTNode {
    real_t A;
    real_t M_A;
    real_t Z;
    real_t M_Z;
};
static_assert(sizeof(LUTNode) == 4 * sizeof(real_t));

DEVICE INLINE FieldForceResult eval_mag_field_and_force(const SolverParams &params, const Vec2 &pos, const Vec3 &q,
                                                        const real_t down_disk_angle, FieldCache *cache = nullptr,
                                                        const LUTNode *LUTptr = nullptr)
{
    Vec3 h = {0};
    real_t nabla_h[5] = {0};
    constexpr int R_magnet = magnet_number / 2;
    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
        real_t s = REAL_LITERAL(1.0);

        for (int i = 0; i < magnet_number; i++) {
            const Vec2 dk =
                pos - make_Vec2(R_magnet * real_cos(static_cast<real_t>(i) * magnet_phi0 + down_disk_angle),
                                R_magnet * real_sin(static_cast<real_t>(i) * magnet_phi0 + down_disk_angle));
            const real_t rhoksq = dk * dk;
            const real_t qk = rhoksq + height * height;
            const real_t invqk = REAL_LITERAL(1.0) / (qk);
            const real_t sqrtinvqk = real_sqrt(invqk);
            h = h + s * invqk * invqk * sqrtinvqk *
                        make_Vec3(REAL_LITERAL(3.0) * height * dk.x, REAL_LITERAL(3.0) * height * dk.y,
                                  REAL_LITERAL(2.0) * height * height - rhoksq);
            const real_t overall = REAL_LITERAL(3.0) * s * invqk * invqk * invqk * sqrtinvqk;
            nabla_h[0] += overall * height * (qk - REAL_LITERAL(5.0) * dk.x * dk.x);
            nabla_h[1] -= overall * REAL_LITERAL(5.0) * height * dk.x * dk.y;
            nabla_h[2] += overall * dk.x * (rhoksq - REAL_LITERAL(4.0) * height * height);
            nabla_h[3] += overall * height * (qk - REAL_LITERAL(5.0) * dk.y * dk.y);
            nabla_h[4] += overall * dk.y * (rhoksq - REAL_LITERAL(4.0) * height * height);
            s *= -1;
        }
    } else if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        constexpr real_t rho_min = kLUTLowerBoundRatio * R_magnet;
        constexpr real_t drho = (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * R_magnet / (kSplineNodes - 1);
        constexpr real_t inv_drho = REAL_LITERAL(1.0) / drho;
        constexpr real_t drho2 = drho * drho;
        const real_t rho = real_sqrt(pos * pos);
        const real_t phi = real_atan2(pos.y, pos.x);
        const real_t node_coordinate = (rho - rho_min) * inv_drho;
        const int left_node_index = static_cast<int>(real_floor(node_coordinate));
        const real_t u = node_coordinate - static_cast<real_t>(left_node_index);

        const LUTNode *left_node = nullptr;
        const LUTNode *right_node = nullptr;
        for (int i = 0; i < kFieldFourierModeCount; i++) {
            CubicCoeff local_A;
            CubicCoeff local_Z;
            CubicCoeff *coeffsA = &local_A;
            CubicCoeff *coeffsZ = &local_Z;
            if constexpr (kLUTCacheMode == LUTCacheMode::kCacheCoeffs) {
                coeffsA = &cache->coeffs.A[i];
                coeffsZ = &cache->coeffs.Z[i];
                if (cache->interval_eff == left_node_index)
                    goto CACHE_HIT_COEFFICIENTS;
                goto CACHE_MISS;
            } else if constexpr (kLUTCacheMode == LUTCacheMode::kCacheNode) {
                if (cache->interval_eff == left_node_index) {
                    left_node = &cache->nodes.left[i];
                    right_node = &cache->nodes.right[i];
                    goto CACHE_HIT_NODES;
                }
                goto CACHE_MISS;
            }
        CACHE_MISS:
            left_node = LUTptr + i * kSplineNodes + left_node_index;
            right_node = left_node + 1;
            if constexpr (kLUTCacheMode == LUTCacheMode::kCacheNode) {
                // 直接复制整个 LUTNode
                cache->nodes.left[i] = *left_node;
                cache->nodes.right[i] = *right_node;
            }
        CACHE_HIT_NODES:
            *coeffsA = spline_coefficients(left_node->A, left_node->M_A, right_node->A, right_node->M_A, drho2);
            *coeffsZ = spline_coefficients(left_node->Z, left_node->M_Z, right_node->Z, right_node->M_Z, drho2);
        CACHE_HIT_COEFFICIENTS:
            const Spline012 A = evaluate_spline(*coeffsA, u, inv_drho);
            const Spline012 Z = evaluate_spline(*coeffsZ, u, inv_drho);
            // TODO: Fourier reconstruction...
        }
        if constexpr (kLUTCacheMode == LUTCacheMode::kCacheCoeffs || kLUTCacheMode == LUTCacheMode::kCacheNode) {
            cache->interval_eff = left_node_index;
        }
    }
    h = h * Beta;
    for (int i = 0; i < 5; i++) {
        nabla_h[i] *= Beta;
    }
    const real_t E_ = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    const real_t D_ = (Mu_r - REAL_LITERAL(1.0)) / (Mu_r + REAL_LITERAL(2.0));
    const real_t A_ = -D_ * params.gamma_m * params.alpha * params.alpha * Eta;
    const real_t B_ = E_ * params.gamma_m * params.alpha * params.alpha * Eta;
    Vec3 v = A_ * h + B_ * q;
    Vec3 Force = make_Vec3((nabla_h[0] * v.x + nabla_h[1] * v.y + nabla_h[2] * v.z),
                           (nabla_h[1] * v.x + nabla_h[3] * v.y + nabla_h[4] * v.z),
                           (nabla_h[2] * v.x + nabla_h[4] * v.y + (-nabla_h[0] - nabla_h[3]) * v.z));
    return {h, Force};
}
