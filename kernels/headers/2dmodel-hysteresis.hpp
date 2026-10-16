// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "hysteresis_config.hpp"
#include "hysteresis_config_field.hpp"
#include "hysteresis_types.hpp"
#include "eddy_currents.hpp"
#include "state_io.hpp"

#ifndef __CUDACC__
#include <cstdio>
#endif

struct TracelessSymTensor {
    real_t component[5];
};
struct FieldResult {
    Vec3 h;
    TracelessSymTensor hGrad3d;
};
struct FieldForceResult {
    Vec3 h;
    Vec3 Force;
    TracelessSymTensor hGrad3d;
    Vec3 magnetic_angular_acceleration;
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

struct LUTNode {
    real_t A;
    real_t M_A;
    real_t Z;
    real_t M_Z;
};
static_assert(sizeof(LUTNode) == 4 * sizeof(real_t));

struct NodeCache {
    LUTNode left[kFieldFourierModeCount];
    LUTNode right[kFieldFourierModeCount];
};

struct CoeffCache {
    CubicCoeff A[kFieldFourierModeCount];
    CubicCoeff Z[kFieldFourierModeCount];
};

struct FieldCache {
    int interval_eff = -1;
    union {
        NodeCache nodes;
        CoeffCache coeffs;
    };
};

#if !defined(__CUDACC__) && defined(USE_FP64)
// Four lanes hold A_i, Z_i, A_(i+1), Z_(i+1). Nodes retain their
// shared CPU/CUDA layout; only registers are rearranged. The odd tail uses
// a zero node without reading past the final Fourier mode.
INLINE void evaluate_lut_splines_avx2(const LUTNode *lut, const int interval,
    const real_t u, const real_t drho2, const real_t inv_drho, FieldCache *cache,
    LUTInterpolation<false> *samples)
{
    const __m256d uv = _mm256_set1_pd(u);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d three = _mm256_set1_pd(3.0);
    const __m256d inv_dr = _mm256_set1_pd(inv_drho);
    const __m256d zero = _mm256_setzero_pd();
    const bool hit = kLUTCacheMode != LUTCacheMode::kNoCache && cache->interval_eff == interval;
    for (int mode = 0; mode < kFieldFourierModeCount; mode += 2) {
        const bool has_second = mode + 1 < kFieldFourierModeCount;
        __m256d a, b, c, d;
        if (kLUTCacheMode == LUTCacheMode::kCacheCoeffs && hit) {
            const CubicCoeff &a0 = cache->coeffs.A[mode], &z0 = cache->coeffs.Z[mode];
            const CubicCoeff empty{};
            const CubicCoeff &a1 = has_second ? cache->coeffs.A[mode + 1] : empty;
            const CubicCoeff &z1 = has_second ? cache->coeffs.Z[mode + 1] : empty;
            a = _mm256_setr_pd(a0.a, z0.a, a1.a, z1.a);
            b = _mm256_setr_pd(a0.b, z0.b, a1.b, z1.b);
            c = _mm256_setr_pd(a0.c, z0.c, a1.c, z1.c);
            d = _mm256_setr_pd(a0.d, z0.d, a1.d, z1.d);
        } else {
            __m256d left[2] = {zero, zero}, right[2] = {zero, zero};
            for (int j = 0; j < 2 && mode + j < kFieldFourierModeCount; ++j) {
                const int index = mode + j;
                const LUTNode *l = lut + index * kSplineNodes + interval;
                const LUTNode *r = l + 1;
                if constexpr (kLUTCacheMode == LUTCacheMode::kCacheNode) {
                    if (!hit) {
                        cache->nodes.left[index] = *l;
                        cache->nodes.right[index] = *r;
                    }
                    l = &cache->nodes.left[index];
                    r = &cache->nodes.right[index];
                }
                left[j] = _mm256_loadu_pd(&l->A);
                right[j] = _mm256_loadu_pd(&r->A);
            }
            // unpack gives [A_i,A_j,Z_i,Z_j]; permute interleaves A/Z.
            a = _mm256_permute4x64_pd(_mm256_unpacklo_pd(left[0], left[1]), 0xd8);
            const __m256d ml = _mm256_permute4x64_pd(_mm256_unpackhi_pd(left[0], left[1]), 0xd8);
            const __m256d r = _mm256_permute4x64_pd(_mm256_unpacklo_pd(right[0], right[1]), 0xd8);
            const __m256d mr = _mm256_permute4x64_pd(_mm256_unpackhi_pd(right[0], right[1]), 0xd8);
            b = _mm256_sub_pd(_mm256_sub_pd(r, a),
                _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(drho2),
                    _mm256_add_pd(_mm256_mul_pd(two, ml), mr)), _mm256_set1_pd(6.0)));
            c = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(drho2), ml), two);
            d = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(drho2), _mm256_sub_pd(mr, ml)),
                _mm256_set1_pd(6.0));
            if constexpr (kLUTCacheMode == LUTCacheMode::kCacheCoeffs) {
                alignas(32) double coeff[4][4];
                _mm256_store_pd(coeff[0], a); _mm256_store_pd(coeff[1], b);
                _mm256_store_pd(coeff[2], c); _mm256_store_pd(coeff[3], d);
                for (int j = 0; j < 2 && mode + j < kFieldFourierModeCount; ++j) {
                    cache->coeffs.A[mode+j] = {coeff[0][2*j], coeff[1][2*j], coeff[2][2*j], coeff[3][2*j]};
                    cache->coeffs.Z[mode+j] = {coeff[0][2*j+1], coeff[1][2*j+1], coeff[2][2*j+1], coeff[3][2*j+1]};
                }
            }
        }
        const __m256d value = _mm256_add_pd(a, _mm256_mul_pd(uv,
            _mm256_add_pd(b, _mm256_mul_pd(uv, _mm256_add_pd(c, _mm256_mul_pd(uv, d))))));
        const __m256d d1 = _mm256_mul_pd(_mm256_add_pd(b, _mm256_mul_pd(uv,
            _mm256_add_pd(_mm256_mul_pd(two, c), _mm256_mul_pd(_mm256_mul_pd(three, d), uv)))), inv_dr);
        const __m256d d2 = _mm256_mul_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(two, c),
            _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(6.0), d), uv)), inv_dr), inv_dr);
        alignas(32) double v[4], first[4], second[4];
        _mm256_store_pd(v, value); _mm256_store_pd(first, d1); _mm256_store_pd(second, d2);
        for (int j = 0; j < 2 && mode + j < kFieldFourierModeCount; ++j) {
            samples[mode+j] = {{v[2*j], first[2*j], second[2*j]},
                              {v[2*j+1], first[2*j+1], second[2*j+1]}};
        }
    }
}
#endif

// MARK: H&F calc



DEVICE NOINLINE FieldResult eval_mag_field(const Vec2 &pos, const real_t down_disk_angle, FieldCache *cache = nullptr,
                                           const LUTNode *LUTptr = nullptr)
// returns SCALED field and gradient
{
    Vec3 h = {0};
    real_t nabla_h[5] = {0};
    constexpr int R_magnet = magnet_number / 2;
    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) { // cache and LUTptr should be nullptr in this mode
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
        TracelessSymTensor Grad;
        h = h * Beta;
        for (int i = 0; i < 5; i++) {
            nabla_h[i] *= Beta;
        }
        for (int i = 0; i < 5; i++)
            Grad.component[i] = nabla_h[i];
        return {h, Grad};
    } else if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        constexpr real_t rho_min = kLUTLowerBoundRatio * R_magnet;
        constexpr real_t drho = (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * R_magnet / (kSplineNodes - 1);
        constexpr real_t inv_drho = REAL_LITERAL(1.0) / drho;
        constexpr real_t drho2 = drho * drho;
        const real_t rho = real_sqrt(pos * pos);
        const real_t phi = real_atan2(pos.y, pos.x);
        const real_t node_coordinate = (rho - rho_min) * inv_drho;
        // Keep both spline nodes inside the table, including at rho_max.
        // Invalid trial states are rejected by the integrator before field evaluation.
        const int left_node_index = !real_isfinite(node_coordinate) || node_coordinate <= REAL_LITERAL(0.0) ? 0
                                    : node_coordinate >= static_cast<real_t>(kSplineNodes - 1)
                                        ? kSplineNodes - 2
                                        : static_cast<int>(real_floor(node_coordinate));
        const real_t u = node_coordinate - static_cast<real_t>(left_node_index);

        real_t h_rho = REAL_LITERAL(0.0);
        real_t h_phi = REAL_LITERAL(0.0);
        real_t h_z = REAL_LITERAL(0.0);
        real_t gradient_rho_rho = REAL_LITERAL(0.0);
        real_t gradient_rho_phi = REAL_LITERAL(0.0);
        real_t gradient_rho_z = REAL_LITERAL(0.0);
        real_t gradient_phi_phi = REAL_LITERAL(0.0);
        real_t gradient_phi_z = REAL_LITERAL(0.0);
        const real_t inv_rho = REAL_LITERAL(1.0) / rho;
        const real_t inv_rho2 = inv_rho * inv_rho;

        // Odd Fourier harmonics theta, 3 theta, ... share a 2 theta rotation.
        const real_t theta = static_cast<real_t>(R_magnet) * (phi - down_disk_angle);
        real_t mode_cos = real_cos(theta);
        real_t mode_sin = real_sin(theta);
        const real_t step_cos = mode_cos * mode_cos - mode_sin * mode_sin;
        const real_t step_sin = REAL_LITERAL(2.0) * mode_sin * mode_cos;

#if !defined(__CUDACC__) && defined(USE_FP64)
        LUTInterpolation<false> samples[kFieldFourierModeCount];
        evaluate_lut_splines_avx2(LUTptr, left_node_index, u, drho2, inv_drho, cache, samples);
#else
        const LUTNode *left_node = nullptr;
        const LUTNode *right_node = nullptr;
#endif
        for (int i = 0; i < kFieldFourierModeCount; i++) {
#if defined(__CUDACC__) || !defined(USE_FP64)
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
#else
            const Spline012 A = samples[i].A;
            const Spline012 Z = samples[i].Z;
#endif

            const int harmonic_int = (2 * i + 1) * magnet_number / 2;
            const real_t harmonic = static_cast<real_t>(harmonic_int);

            // Phi_i = A_i(rho, z) cos(m_i (phi - disk_angle)); h = -grad Phi.
            h_rho -= A.d1 * mode_cos;
            h_phi += harmonic * A.value * inv_rho * mode_sin;
            h_z -= Z.value * mode_cos;

            // Independent components of grad(h) in the orthonormal
            // (e_rho, e_phi, e_z) basis.  Z = partial_z A and Z.d1 is
            // partial_rho partial_z A.
            gradient_rho_rho -= A.d2 * mode_cos;
            gradient_rho_phi += harmonic * (A.d1 * inv_rho - A.value * inv_rho2) * mode_sin;
            gradient_rho_z -= Z.d1 * mode_cos;
            gradient_phi_phi += (harmonic * harmonic * A.value * inv_rho2 - A.d1 * inv_rho) * mode_cos;
            gradient_phi_z += harmonic * Z.value * inv_rho * mode_sin;
            if (i + 1 < kFieldFourierModeCount) {
                const real_t next_cos = mode_cos * step_cos - mode_sin * step_sin;
                const real_t next_sin = mode_sin * step_cos + mode_cos * step_sin;
                mode_cos = next_cos;
                mode_sin = next_sin;
            }
        }
        if constexpr (kLUTCacheMode == LUTCacheMode::kCacheCoeffs || kLUTCacheMode == LUTCacheMode::kCacheNode) {
            cache->interval_eff = left_node_index;
        }

        // Rotate the vector and rank-2 tensor from the local cylindrical
        // basis into laboratory Cartesian coordinates.  This rotation uses
        // phi, not the relative Fourier phase phi - down_disk_angle.
        const real_t cos_phi = pos.x * inv_rho;
        const real_t sin_phi = pos.y * inv_rho;
        const real_t cos_phi2 = cos_phi * cos_phi;
        const real_t sin_phi2 = sin_phi * sin_phi;
        const real_t sin_cos_phi = sin_phi * cos_phi;
        h = make_Vec3(cos_phi * h_rho - sin_phi * h_phi, sin_phi * h_rho + cos_phi * h_phi, h_z);
        nabla_h[0] = cos_phi2 * gradient_rho_rho - REAL_LITERAL(2.0) * sin_cos_phi * gradient_rho_phi +
                     sin_phi2 * gradient_phi_phi;
        nabla_h[1] = sin_cos_phi * (gradient_rho_rho - gradient_phi_phi) + (cos_phi2 - sin_phi2) * gradient_rho_phi;
        nabla_h[2] = cos_phi * gradient_rho_z - sin_phi * gradient_phi_z;
        nabla_h[3] = sin_phi2 * gradient_rho_rho + REAL_LITERAL(2.0) * sin_cos_phi * gradient_rho_phi +
                     cos_phi2 * gradient_phi_phi;
        nabla_h[4] = sin_phi * gradient_rho_z + cos_phi * gradient_phi_z;
        TracelessSymTensor Grad;
        for (int i = 0; i < 5; i++)
            Grad.component[i] = nabla_h[i];
        h = h * Beta;
        for (int i = 0; i < 5; i++) {
            Grad.component[i] *= Beta;
        }
        return {h, Grad};
    }
}

DEVICE INLINE FieldForceResult eval_mag_field_and_force(const SolverParams &params, const Vec2 &pos, const Vec3 &q,
                                                        const real_t down_disk_angle, FieldCache *cache = nullptr,
                                                        const LUTNode *LUTptr = nullptr,
                                                        const FieldResult *H_cache = nullptr)
{
    const FieldResult field = (H_cache == nullptr) ? eval_mag_field(pos, down_disk_angle, cache, LUTptr) : *H_cache;
    const Vec3 &h = field.h;
    const TracelessSymTensor &nabla_h = field.hGrad3d;

    constexpr real_t E_ = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    constexpr real_t D_ = (Mu_r - REAL_LITERAL(1.0)) / (Mu_r + REAL_LITERAL(2.0));
    const real_t A_ = D_ * params.gamma_m * params.alpha * params.alpha * Eta;
    const real_t B_ = -E_ * params.gamma_m * params.alpha * params.alpha * Eta;
    Vec3 v = A_ * h + B_ * q;
    Vec3 Force = make_Vec3((nabla_h.component[0] * v.x + nabla_h.component[1] * v.y + nabla_h.component[2] * v.z),
                           (nabla_h.component[1] * v.x + nabla_h.component[3] * v.y + nabla_h.component[4] * v.z),
                           (nabla_h.component[2] * v.x + nabla_h.component[4] * v.y +
                            (-nabla_h.component[0] - nabla_h.component[3]) * v.z));
    return {h, Force, nabla_h, -params.gamma_m * E_ * cross(q, h)};
}

// Shared magnetic loads for dynamics, contact decisions and power diagnostics.
// Field caches remain external-field-only; the state-dependent loads are fresh.
DEVICE INLINE FieldForceResult eval_mag_field_and_force(const SolverParams &params,
    const State<kEddyCurrentModeCount> &state, FieldCache *cache = nullptr,
    const LUTNode *LUTptr = nullptr, const FieldResult *H_cache = nullptr)
{
    const Vec3 q_world = vec_cast<real_t>(rotate(state.attitude, state.q));
    auto result = eval_mag_field_and_force(params, state.pos, q_world, state.down_disk_angle,
                                           cache, LUTptr, H_cache);
    if constexpr (kEnableEddyCurrents) {
        const Vec3 delta_m = eval_eddy_current_moment_world(state);
        const real_t scale = params.gamma_m * params.alpha * params.alpha * Eta / REAL_LITERAL(3.0);
        const auto &g = result.hGrad3d.component;
        result.Force = result.Force + scale * make_Vec3(
            g[0] * delta_m.x + g[1] * delta_m.y + g[2] * delta_m.z,
            g[1] * delta_m.x + g[3] * delta_m.y + g[4] * delta_m.z,
            g[2] * delta_m.x + g[4] * delta_m.y - (g[0] + g[3]) * delta_m.z);
        result.magnetic_angular_acceleration = result.magnetic_angular_acceleration +
            (params.gamma_m / REAL_LITERAL(3.0)) * cross(delta_m, result.h);
    }
    return result;
}

// Use the same effective force in the RHS and all contact-mode decisions.
// The field evaluator continues to return the unmodified magnetic force.
DEVICE INLINE Vec3 apply_inclined_disk_force(const SolverParams &params, Vec3 force,
                                           const Vec2 &position = Vec2{})
{
    if constexpr (kEnableInclinedDisk) {
        const real_t downhill = kTiltAngleRad * (params.gamma_g - force.z);
        force.x += downhill * real_sin(kTiltAxisAzimuthRad);
        force.y -= downhill * real_cos(kTiltAxisAzimuthRad);
    }
    // A force through the ball centre follows the reverse tangent, without
    // changing the normal load or applying a direct torque.
    if (params.reverse_push_acceleration > REAL_LITERAL(0.0)) {
        const real_t radius = length(position);
        if (radius > REAL_LITERAL(0.0)) {
            force.x += params.reverse_push_acceleration * position.y / radius;
            force.y -= params.reverse_push_acceleration * position.x / radius;
        }
    }
    return force;
}

DEVICE INLINE auto eval_body_field_derivative(const State<kEddyCurrentModeCount> &in, const Vec3 &h, const TracelessSymTensor &grad)
{
    // The lower disk angular velocity is one in the dimensionless system.
    const auto relative_velocity = make_Vec2(in.speed.x + in.pos.y, in.speed.y - in.pos.x);
    const auto h_dot_world =
        make_Vec3(grad.component[0] * relative_velocity.x + grad.component[1] * relative_velocity.y - h.y,
                  grad.component[1] * relative_velocity.x + grad.component[3] * relative_velocity.y + h.x,
                  grad.component[2] * relative_velocity.x + grad.component[4] * relative_velocity.y);

    return rotate(conjugate(in.attitude), h_dot_world - cross(in.omega, h));
}


// Return total angular acceleration (magnetic torque plus passive rolling torque).
// Share it between the RHS and every rolling/sliding contact decision.
DEVICE INLINE Vec3 add_gaussian_rolling_resistance(const SolverParams &params, const Vec3 &magnetic,
                                                  const State<kEddyCurrentModeCount> &state, const Vec3 &force)
{
    if constexpr (!kEnableRollingResistance) {
        return magnetic;
    }
#ifdef IYPT16_DISABLE_GAUSSIAN_ROLLING
    return magnetic; // Test the original zero-added-resistance limit.
#else
    const Vec3 omega = vec_cast<real_t>(state.omega);
    const real_t w2 = omega.x * omega.x + omega.y * omega.y;
    const real_t rate2 = w2 / (kRollingTransitionOmega * kRollingTransitionOmega);
#ifdef __CUDACC__
    const real_t gaussian = exp(-rate2);
#else
    const real_t gaussian = static_cast<real_t>(std::exp(-static_cast<double>(rate2)));
#endif
    const real_t mu = kRollingMuHigh + (kRollingMuLow - kRollingMuHigh) * gaussian;
    const real_t normal = params.gamma_g - force.z;
    const real_t regularized_w = real_sqrt(w2 + kRollingOmegaEpsilon * kRollingOmegaEpsilon);
    const real_t damping = mu * normal / (Eta * params.alpha * regularized_w) + kRollingLambda;
    return magnetic - make_Vec3(damping * omega.x, damping * omega.y,0.0);
#endif
}

// MARK: RHS eval
template <bool ExportFM>
DEVICE NOINLINE bool eval_rhs(const State<kEddyCurrentModeCount> &in, State<kEddyCurrentModeCount> &out, Vec3 &hb_dot,
                              const SolverParams &params, const bool &sliding,
                              const Vec2 &sliding_dir, FieldCache *cache = nullptr, const LUTNode *LUTptr = nullptr,
                              const FieldResult *H_cache = nullptr, Vec3 *Fext = nullptr, Vec3 *Mext = nullptr)
{
    // the caller should cope with selection by giving nullptr when magField cache is not valid
    clear_eddy_current_modes(out);

    const FieldForceResult field_force =
        eval_mag_field_and_force(params, in, cache, LUTptr, H_cache);
    const Vec3 &h = field_force.h;
    const Vec3 F = apply_inclined_disk_force(params, field_force.Force, in.pos);
    constexpr real_t E_ = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    const Vec3 M = add_gaussian_rolling_resistance(params, field_force.magnetic_angular_acceleration, in, F);
    if constexpr (ExportFM) {
        *Fext = F;
        *Mext = M;
    }
    const real_t N = params.gamma_g - F.z;
    if (N < REAL_LITERAL(0.0))
        return false;
    if (sliding) {
        const Vec2 f = -sliding_dir * Mu * N;
        out.speed = f + make_Vec2(F.x, F.y) - (params.delta * in.speed) * length(in.speed);
        out.omega = M + make_Vec3(f.y, -f.x, REAL_LITERAL(0.0)) / (Eta * params.alpha);
    } else {
        constexpr real_t inv_one_plus_eta = REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Eta);
        const real_t inv_alpha_one_plus_eta = inv_one_plus_eta / params.alpha;
        const real_t speed_norm = length(in.speed);

        const real_t drag_factor = params.delta * speed_norm;

        const Vec3 totalF = make_Vec3(F.x - drag_factor * in.speed.x, F.y - drag_factor * in.speed.y, F.z);
        // Pure-rolling angular acceleration.
        out.omega = make_Vec3(Eta * inv_one_plus_eta * M.x - inv_alpha_one_plus_eta * totalF.y,
                              Eta * inv_one_plus_eta * M.y + inv_alpha_one_plus_eta * totalF.x, M.z);
        // Recover translational acceleration from rolling constraint.
        out.speed = make_Vec2(params.alpha * out.omega.y, -params.alpha * out.omega.x);
    }

    const auto h_ext_dot = eval_body_field_derivative(in, h, field_force.hGrad3d);
    hb_dot = vec_cast<real_t>(h_ext_dot);

    Vec3 h0_dot = h_ext_dot * (REAL_LITERAL(3.0) / (Mu_r + REAL_LITERAL(2.0)));
    // assume ReLU use the 0 branch
    real_t interme = Alpha_JA * (REAL_LITERAL(1.0) - C_JA * Xi_JA) / Xi_JA - E_;
    Vec3 q_dot = (REAL_LITERAL(1.0) / interme) * h0_dot;
    // IMPORTANT: self-consistent H_e_dot
    Vec3 he_dot = (Alpha_JA / (Xi_JA * interme)) * h0_dot;

    // if reject we should select ReLU x branch
    if (in.q * he_dot > REAL_LITERAL(0.0)) {
        constexpr real_t c = Xi_JA / Alpha_JA;
        const real_t d = -length(in.q) / K;
        constexpr real_t a = REAL_LITERAL(1.0) - C_JA * Xi_JA - E_ * c;
        const real_t b = -Alpha_JA * (REAL_LITERAL(1.0) - C_JA) / K * length(in.q) - E_ * d;
        const real_t a_final = c / a;
        const real_t b_final = d / a - b * (c + d) / a / (a + b);
        const Vec3 qnorm = in.q * (REAL_LITERAL(1.0) / length(in.q));
        q_dot = a_final * h0_dot + b_final * (qnorm * h0_dot) * qnorm;
    }
    out.q = q_dot;
    out.pos = in.speed;
    out.attitude = REAL_LITERAL(0.5) * make_Quaternion(REAL_LITERAL(0.0), in.omega) * in.attitude;
    out.down_disk_angle = REAL_LITERAL(1.0);
    return true;
}


// MARK: Misc

enum class AdvanceStepResult {
    kStepOk,
    kStepBallFly,
    kStepFinerRetry,
    kStepNumericalUnstable,
    kStepNaNOrInf,
    kStepSolverSelfContradict,
    kStepLUTOutOfDomain,
    kStepRefinementRecursionLimit,
    // Temporary diagnostic statuses: one per finer-retry return site.
    kRetrySlidingHalfTcZero,
    kRetrySlidingContactCannotStick,
    kRetrySlidingRollingMidRelease,
    kRetrySlidingRollingFinalRelease,
    kRetrySlidingMidZero,
    kRetrySlidingMidPredictsContact,
    kRetrySlidingFinalZero,
    kRetryRollingThresholdRecross,
    kRetryRollingEventInvalid,
    kRetryRollingReleaseMidZero,
    kRetryRollingReleaseMidContact,
    kRetryRollingReleaseFinalZero,
};

struct RetryDiagnostics {
    unsigned long long counts[12] = {};
    int last_reason = -1;
    int last_depth = -1;
    real_t last_dt = REAL_LITERAL(0.0);
    State<kEddyCurrentModeCount> last_state{};
    real_t release_g_start = REAL_LITERAL(0.0);
    real_t release_g_mid = REAL_LITERAL(0.0);
    real_t release_g_final = REAL_LITERAL(0.0);
    real_t release_A_event = REAL_LITERAL(0.0);
    real_t release_b_event = REAL_LITERAL(0.0);
    real_t release_tr = REAL_LITERAL(0.0);
};

// RK2 is kept inline because additional checks are performed between stages.

// Remove roundoff drift from the sticking constraint while preserving the
// contact impulse balance used at a sliding-to-sticking event.
DEVICE INLINE void project_pure_rolling(State<kEddyCurrentModeCount> &in, const SolverParams &params)
{
    const accumulator_t eta = static_cast<accumulator_t>(Eta);
    const accumulator_t alpha = static_cast<accumulator_t>(params.alpha);
    const accumulator_t denominator = static_cast<accumulator_t>(REAL_LITERAL(1.0)) + eta;
    const accumulator_t speed_x = static_cast<accumulator_t>(in.speed.x);
    const accumulator_t speed_y = static_cast<accumulator_t>(in.speed.y);
    const accumulator_t omega_x = static_cast<accumulator_t>(in.omega.x);
    const accumulator_t omega_y = static_cast<accumulator_t>(in.omega.y);
    const accumulator_t omega_z = static_cast<accumulator_t>(in.omega.z);

    const StateVec2 vf = make_Vec2(static_cast<accumulator_t>((speed_x + eta * alpha * omega_y) / denominator),
                                   static_cast<accumulator_t>((speed_y - eta * alpha * omega_x) / denominator));
    const StateVec3 omegaf = make_Vec3(static_cast<accumulator_t>((eta * omega_x - speed_y / alpha) / denominator),
                                       static_cast<accumulator_t>((eta * omega_y + speed_x / alpha) / denominator),
                                       static_cast<accumulator_t>(omega_z));
    in.speed = vf;
    in.omega = omegaf;
}

DEVICE INLINE bool normalize_attitude(State<kEddyCurrentModeCount> &in)
{
    const real_t squared_norm = static_cast<real_t>(norm_sq(in.attitude));
    if (!real_isfinite(squared_norm) || squared_norm <= REAL_LITERAL(0.0))
        return false;

    real_t scale;
    if constexpr (kAttitudeNormalizeMode == AttitudeNormalizeMode::kExact) {
        scale = REAL_LITERAL(1.0) / real_sqrt(squared_norm);
    } else {
        // One Newton step for 1 / sqrt(squared_norm), starting from 1.
        scale = REAL_LITERAL(0.5) * (REAL_LITERAL(3.0) - squared_norm);
    }
    if (!real_isfinite(scale) || scale <= REAL_LITERAL(0.0))
        return false;
    in.attitude = in.attitude * scale;
    return true;
}

DEVICE INLINE bool state_in_lut_domain(const State<kEddyCurrentModeCount> &state)
{
    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        constexpr real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
        constexpr real_t rho_min = kLUTLowerBoundRatio * radius;
        constexpr real_t rho_max = kLUTUpperBoundRatio * radius;
        const real_t rho2 = static_cast<real_t>(state.pos * state.pos);
        return real_isfinite(rho2) && rho2 >= rho_min * rho_min && rho2 <= rho_max * rho_max;
    }
    return true;
}

// A zero-slip endpoint can either stick or launch on the next step.
// Shared by the ordinary sliding and post-release paths; never normalize zero slip.
DEVICE INLINE AdvanceStepResult resolve_zero_slip_mode(const State<kEddyCurrentModeCount> &state, const SolverParams &params, bool &sliding,
                                                       Vec2 &direction, FieldCache *cache = nullptr,
                                                       const LUTNode *LUTptr = nullptr)
{
    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        if (!state_in_lut_domain(state))
            return AdvanceStepResult::kStepLUTOutOfDomain;
    }
    const FieldForceResult field =
        eval_mag_field_and_force(params, state, cache, LUTptr, nullptr);
    const Vec3 F = apply_inclined_disk_force(params, field.Force, state.pos);
    const Vec3 torque = add_gaussian_rolling_resistance(params, field.magnetic_angular_acceleration, state, F);
    const real_t N = params.gamma_g - F.z;
    if (!real_isfinite(N))
        return AdvanceStepResult::kStepNaNOrInf;
    if (N < REAL_LITERAL(0.0))
        return AdvanceStepResult::kStepBallFly;
    const Vec2 a = make_Vec2(F.x, F.y) - params.delta * length(state.speed) * state.speed +
                   params.alpha * make_Vec2(-torque.y, torque.x);
    const real_t A = length(a);
    const real_t b = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N;
    if (!real_isfinite(A) || !real_isfinite(b))
        return AdvanceStepResult::kStepNaNOrInf;
    sliding = A > b;
    direction = sliding ? a / A : make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    return AdvanceStepResult::kStepOk;
}

// MARK: Try 1 step
DEVICE NOINLINE AdvanceStepResult try_advance_1_step(const real_t dt, const SolverParams &params, State<kEddyCurrentModeCount> &in,
                                                     bool &sliding, Vec2 &sliding_dir, bool &isHCacheValid,
                                                     FieldCache *cache = nullptr, const LUTNode *LUTptr = nullptr,
                                                     FieldResult *H_cache = nullptr,
                                                     RetryDiagnostics *diagnostics = nullptr)
{
    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        if (!state_in_lut_domain(in))
            return AdvanceStepResult::kStepLUTOutOfDomain;
    }
    if (sliding) {
        Vec3 F_start;
        Vec3 M_start;
        State<kEddyCurrentModeCount> Statedot_start;
        Vec3 hb_dot_start;
        if (isHCacheValid) {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(in, Statedot_start, hb_dot_start, params, sliding, sliding_dir, nullptr, nullptr, H_cache,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(in, Statedot_start, hb_dot_start, params, sliding, sliding_dir, cache, LUTptr, H_cache, &F_start,
                                    &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        } else {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(in, Statedot_start, hb_dot_start, params, sliding, sliding_dir, nullptr, nullptr, nullptr,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(in, Statedot_start, hb_dot_start, params, sliding, sliding_dir, cache, LUTptr, nullptr, &F_start,
                                    &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        }
        const Vec2 G = make_Vec2(F_start.x, F_start.y) - params.delta * length(in.speed) * in.speed;
        const Vec2 a = G + params.alpha * make_Vec2(-M_start.y, M_start.x);
        real_t A = length(a);
        real_t b = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_start.z);
        if (!real_isfinite(A) || !real_isfinite(b))
            return AdvanceStepResult::kStepNaNOrInf;
        if (A < b) {
            real_t D_c = (b - A) * (b + A);
            Vec2 s0 = vec_cast<real_t>(in.speed + params.alpha * make_Vec2(-in.omega.y, in.omega.x));
            real_t Q = b * length(s0) + a * s0;
            if (!real_isfinite(Q) || !real_isfinite(D_c) || !real_isfinite(dt * D_c))
                return AdvanceStepResult::kStepNaNOrInf;
            if (Q <= dt * D_c) {
                real_t tc = Q / D_c;
                if (!real_isfinite(tc) || tc <= REAL_LITERAL(0.0) || tc > dt)
                    return AdvanceStepResult::kStepNumericalUnstable;
                State<kEddyCurrentModeCount> state_at_halftc = in + tc * REAL_LITERAL(0.5) * Statedot_start;
                advance_eddy_current_state(in, state_at_halftc, tc * REAL_LITERAL(0.5), hb_dot_start, params);
                State<kEddyCurrentModeCount> dot_estimate;
                Vec3 hb_dot_estimate;
                Vec2 s_mid =
                    vec_cast<real_t>(state_at_halftc.speed +
                                     params.alpha * make_Vec2(-state_at_halftc.omega.y, state_at_halftc.omega.x));
                const real_t s_mid_norm = length(s_mid);
                if (!real_isfinite(s_mid_norm))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (s_mid_norm == REAL_LITERAL(0.0))
                    return AdvanceStepResult::kStepNumericalUnstable;
                Vec2 sliddir_mid = s_mid / s_mid_norm;
                if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                    if (!state_in_lut_domain(state_at_halftc))
                        return AdvanceStepResult::kStepLUTOutOfDomain;
                }
                if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                    if (!eval_rhs<false>(state_at_halftc, dot_estimate, hb_dot_estimate, params, sliding, sliddir_mid, nullptr, nullptr,
                                         nullptr))
                        return AdvanceStepResult::kStepBallFly;
                } else {
                    if (!eval_rhs<false>(state_at_halftc, dot_estimate, hb_dot_estimate, params, sliding, sliddir_mid, cache, LUTptr,
                                         nullptr))
                        return AdvanceStepResult::kStepBallFly;
                }
                State<kEddyCurrentModeCount> state_at_tc = in + tc * dot_estimate;
                advance_eddy_current_state(in, state_at_tc, tc, hb_dot_estimate, params);
                project_pure_rolling(state_at_tc, params);
                if (!normalize_attitude(state_at_tc))
                    return AdvanceStepResult::kStepNumericalUnstable;
                if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                    if (!state_in_lut_domain(state_at_tc))
                        return AdvanceStepResult::kStepLUTOutOfDomain;
                }
                const FieldForceResult field_force = eval_mag_field_and_force(
                    params, state_at_tc, cache, LUTptr, nullptr);
                const Vec3 &h = field_force.h;
                const Vec3 F = apply_inclined_disk_force(params, field_force.Force, state_at_tc.pos);
                const Vec3 M = add_gaussian_rolling_resistance(params, field_force.magnetic_angular_acceleration, state_at_tc, F);
                const real_t N = params.gamma_g - F.z;
                if (!real_isfinite(N) || !real_isfinite(F.x) || !real_isfinite(F.y) || !real_isfinite(M.x) ||
                    !real_isfinite(M.y))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (N < REAL_LITERAL(0.0))
                    return AdvanceStepResult::kStepBallFly;
                const Vec2 G_ = make_Vec2(F.x, F.y) - params.delta * length(state_at_tc.speed) * state_at_tc.speed;
                const Vec2 a_ = G_ + params.alpha * make_Vec2(-M.y, M.x);
                real_t A_ = length(a_);
                real_t b_ = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N;
                if (!real_isfinite(A_) || !real_isfinite(b_))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (A_ > b_)
                    return AdvanceStepResult::kRetrySlidingContactCannotStick;

                const real_t remaining = dt - tc;
                State<kEddyCurrentModeCount> final_state = state_at_tc;
                FieldResult final_field = {h, field_force.hGrad3d};
                if (remaining > REAL_LITERAL(0.0)) {
                    const bool ksliding = false;
                    const Vec2 no_sliding_dir = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
                    State<kEddyCurrentModeCount> dot_rolling_start;
                    Vec3 hb_dot_rolling_start;
                    State<kEddyCurrentModeCount> dot_rolling_mid;
                    Vec3 hb_dot_rolling_mid;
                    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                        if (!eval_rhs<false>(state_at_tc, dot_rolling_start, hb_dot_rolling_start, params, ksliding, no_sliding_dir, nullptr,
                                             nullptr, &final_field))
                            return AdvanceStepResult::kStepBallFly;
                    } else {
                        if (!eval_rhs<false>(state_at_tc, dot_rolling_start, hb_dot_rolling_start, params, ksliding, no_sliding_dir, cache,
                                             LUTptr, &final_field))
                            return AdvanceStepResult::kStepBallFly;
                    }
                    State<kEddyCurrentModeCount> rolling_mid = state_at_tc + remaining * REAL_LITERAL(0.5) * dot_rolling_start;
                    advance_eddy_current_state(state_at_tc, rolling_mid, remaining * REAL_LITERAL(0.5), hb_dot_rolling_start, params);
                    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                        if (!state_in_lut_domain(rolling_mid))
                            return AdvanceStepResult::kStepLUTOutOfDomain;
                    }
                    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                        if (!eval_rhs<false>(rolling_mid, dot_rolling_mid, hb_dot_rolling_mid, params, ksliding, no_sliding_dir, nullptr,
                                             nullptr, nullptr))
                            return AdvanceStepResult::kStepBallFly;
                    } else {
                        if (!eval_rhs<false>(rolling_mid, dot_rolling_mid, hb_dot_rolling_mid, params, ksliding, no_sliding_dir, cache,
                                             LUTptr, nullptr))
                            return AdvanceStepResult::kStepBallFly;
                    }

                    final_state = state_at_tc + remaining * dot_rolling_mid;
                    advance_eddy_current_state(state_at_tc, final_state, remaining, hb_dot_rolling_mid, params);
                    project_pure_rolling(final_state, params);
                    if (!normalize_attitude(final_state))
                        return AdvanceStepResult::kStepNumericalUnstable;
                    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                        if (!state_in_lut_domain(final_state))
                            return AdvanceStepResult::kStepLUTOutOfDomain;
                    }
                    const FieldForceResult field_force_final = eval_mag_field_and_force(
                        params, final_state, cache, LUTptr, nullptr);
                    const Vec3 &h_final = field_force_final.h;
                    const Vec3 F_final = apply_inclined_disk_force(params, field_force_final.Force, final_state.pos);
                    const Vec3 M_final = add_gaussian_rolling_resistance(params, field_force_final.magnetic_angular_acceleration, final_state, F_final);
                    const real_t N_final = params.gamma_g - F_final.z;
                    if (!real_isfinite(N_final) || !real_isfinite(F_final.x) || !real_isfinite(F_final.y) ||
                        !real_isfinite(M_final.x) || !real_isfinite(M_final.y))
                        return AdvanceStepResult::kStepNaNOrInf;
                    if (N_final < REAL_LITERAL(0.0))
                        return AdvanceStepResult::kStepBallFly;
                    const Vec2 G_final =
                        make_Vec2(F_final.x, F_final.y) - params.delta * length(final_state.speed) * final_state.speed;
                    const Vec2 a_final = G_final + params.alpha * make_Vec2(-M_final.y, M_final.x);
                    const real_t A_final = length(a_final);
                    const real_t b_final = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N_final;
                    if (!real_isfinite(A_final) || !real_isfinite(b_final))
                        return AdvanceStepResult::kStepNaNOrInf;
                    if (A_final > b_final) {
                        final_field = {h_final, field_force_final.hGrad3d};
                        sliding = true;
                        sliding_dir = a_final / A_final;
                        goto FINALE;
                    }
                    final_field = {h_final, field_force_final.hGrad3d};
                }

                sliding = false;
                sliding_dir = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
            FINALE:
                if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                    if (!state_in_lut_domain(final_state))
                        return AdvanceStepResult::kStepLUTOutOfDomain;
                }
                if (H_cache != nullptr)
                    *H_cache = final_field;
                in = final_state;
                isHCacheValid = H_cache != nullptr;
                return AdvanceStepResult::kStepOk;
            } else {
                goto NO_PREDICTED_TRANSIT;
            }
        } else {
        NO_PREDICTED_TRANSIT:
            // Resolve the friction direction when slip is small. The capture
            // prediction above takes priority over this sliding-step guard.
            const Vec2 s_start = vec_cast<real_t>(in.speed + params.alpha * make_Vec2(-in.omega.y, in.omega.x));
            const real_t s_start_norm = length(s_start);
            if (!real_isfinite(s_start_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if constexpr (kFinerStepCheck) {
                if (s_start_norm > REAL_LITERAL(0.0)) {
                    // Use the actual initial slip derivative, including cancellation
                    // between the drive and friction. Its first bound keeps the RK2
                    // midpoint slip at least 0.75 * |s_start|. The second limits
                    // transverse friction stiffness, b / |s_start|.
                    const Vec2 s_dot_start =
                        vec_cast<real_t>(Statedot_start.speed +
                                         params.alpha * make_Vec2(-Statedot_start.omega.y, Statedot_start.omega.x));
                    if (dt * length(s_dot_start) > REAL_LITERAL(0.5) * s_start_norm || dt * b > s_start_norm)
                        return AdvanceStepResult::kStepFinerRetry;
                }
            }
            State<kEddyCurrentModeCount> rk2midpoint = dt * REAL_LITERAL(0.5) * Statedot_start + in;
            advance_eddy_current_state(in, rk2midpoint, dt * REAL_LITERAL(0.5), hb_dot_start, params);
            Vec2 s_mid = vec_cast<real_t>(rk2midpoint.speed +
                                          params.alpha * make_Vec2(-rk2midpoint.omega.y, rk2midpoint.omega.x));
            const real_t s_mid_norm = length(s_mid);
            if (!real_isfinite(s_mid_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            // Preserve the incoming direction at a zero-slip RK stage; the capture check below still applies.
            const Vec2 sliddir_mid = s_mid_norm > REAL_LITERAL(0.0) ? s_mid / s_mid_norm : sliding_dir;
            if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                if (!state_in_lut_domain(rk2midpoint))
                    return AdvanceStepResult::kStepLUTOutOfDomain;
            }
            Vec3 F_mid;
            Vec3 M_mid;
            State<kEddyCurrentModeCount> Statedot_mid;
            Vec3 hb_dot_mid;
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(rk2midpoint, Statedot_mid, hb_dot_mid, params, sliding, sliddir_mid, nullptr, nullptr, nullptr,
                                    &F_mid, &M_mid))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(rk2midpoint, Statedot_mid, hb_dot_mid, params, sliding, sliddir_mid, cache, LUTptr, nullptr,
                                    &F_mid, &M_mid))
                    return AdvanceStepResult::kStepBallFly;
            }
            const Vec2 Gmid =
                make_Vec2(F_mid.x, F_mid.y) - params.delta * length(rk2midpoint.speed) * rk2midpoint.speed;
            const Vec2 amid = Gmid + params.alpha * make_Vec2(-M_mid.y, M_mid.x);
            real_t Amid = length(amid);
            real_t bmid = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_mid.z);
            if (!real_isfinite(Amid) || !real_isfinite(bmid))
                return AdvanceStepResult::kStepNaNOrInf;
            // Friction changes slip length at first order, but its contribution
            // to the instantaneous change of slip direction is radial. Use the
            // midpoint drive to predict that direction without normalizing the
            // potentially tiny Euler midpoint slip.
            const Vec2 direction_predictor = s_start + dt * REAL_LITERAL(0.5) * amid;
            const real_t predictor_norm = length(direction_predictor);
            if (!real_isfinite(predictor_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if (predictor_norm == REAL_LITERAL(0.0))
                return AdvanceStepResult::kRetrySlidingMidZero;
            const Vec2 predicted_dir_mid = direction_predictor / predictor_norm;
            const Vec2 friction_delta = -Mu * (params.gamma_g - F_mid.z) * (predicted_dir_mid - sliddir_mid);
            Statedot_mid.speed = Statedot_mid.speed + friction_delta;
            Statedot_mid.omega = Statedot_mid.omega +
                                  make_Vec3(friction_delta.y, -friction_delta.x, REAL_LITERAL(0.0)) /
                                      (Eta * params.alpha);
            if (Amid < bmid) {
                real_t D_cmid = (bmid - Amid) * (bmid + Amid);
                real_t Qmid = bmid * length(s_mid) + amid * s_mid;
                if (!real_isfinite(Qmid) || !real_isfinite(D_cmid) || !real_isfinite(dt * REAL_LITERAL(0.5) * D_cmid))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (Qmid <= dt * REAL_LITERAL(0.5) * D_cmid) {
                    return AdvanceStepResult::kRetrySlidingMidPredictsContact;
                }
            }
            State<kEddyCurrentModeCount> final_state = in + dt * Statedot_mid;
            advance_eddy_current_state(in, final_state, dt, hb_dot_mid, params);
            if (!normalize_attitude(final_state))
                return AdvanceStepResult::kStepNumericalUnstable;
            Vec2 sf = vec_cast<real_t>(final_state.speed +
                                       params.alpha * make_Vec2(-final_state.omega.y, final_state.omega.x));
            const real_t sf_norm = length(sf);
            if (!real_isfinite(sf_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                if (!state_in_lut_domain(final_state))
                    return AdvanceStepResult::kStepLUTOutOfDomain;
            }
            bool final_sliding = true;
            Vec2 final_dir;
            if (sf_norm == REAL_LITERAL(0.0)) {
                const AdvanceStepResult mode_result =
                    resolve_zero_slip_mode(final_state, params, final_sliding, final_dir, cache, LUTptr);
                if (mode_result != AdvanceStepResult::kStepOk)
                    return mode_result;
            } else {
                final_dir = sf / sf_norm;
            }
            in = final_state;
            sliding = final_sliding;
            sliding_dir = final_dir;
            isHCacheValid = false;
            return AdvanceStepResult::kStepOk;
        }
    } else {
        State<kEddyCurrentModeCount> start = in;
        project_pure_rolling(start, params);
        Vec3 F_start;
        Vec3 M_start;
        State<kEddyCurrentModeCount> Statedot_start;
        Vec3 hb_dot_start;
        if (isHCacheValid) {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(start, Statedot_start, hb_dot_start, params, sliding, sliding_dir, nullptr, nullptr, H_cache,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(start, Statedot_start, hb_dot_start, params, sliding, sliding_dir, cache, LUTptr, H_cache,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        } else {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(start, Statedot_start, hb_dot_start, params, sliding, sliding_dir, nullptr, nullptr, nullptr,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(start, Statedot_start, hb_dot_start, params, sliding, sliding_dir, cache, LUTptr, nullptr,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        }
        const Vec2 G = make_Vec2(F_start.x, F_start.y) - params.delta * length(start.speed) * start.speed;
        const Vec2 a = G + params.alpha * make_Vec2(-M_start.y, M_start.x);
        real_t A = length(a);
        real_t b = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_start.z);
        if (!real_isfinite(A) || !real_isfinite(b))
            return AdvanceStepResult::kStepNaNOrInf;
        if (A > b)
            return AdvanceStepResult::kStepSolverSelfContradict;

        const real_t g_start = A - b; // kappa * (|F_req| - Mu * N).
        const Vec2 no_sliding_dir = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
        const bool krolling = false;
        const bool ksliding = true;
        State<kEddyCurrentModeCount> rk2midpoint = start + dt * REAL_LITERAL(0.5) * Statedot_start;
        advance_eddy_current_state(start, rk2midpoint, dt * REAL_LITERAL(0.5), hb_dot_start, params);
        if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
            if (!state_in_lut_domain(rk2midpoint))
                return AdvanceStepResult::kStepLUTOutOfDomain;
        }
        State<kEddyCurrentModeCount> Statedot_mid;
        Vec3 hb_dot_mid;
        Vec3 F_mid;
        Vec3 M_mid;
        if (!eval_rhs<true>(rk2midpoint, Statedot_mid, hb_dot_mid, params, krolling, no_sliding_dir, cache, LUTptr, nullptr, &F_mid,
                            &M_mid))
            return AdvanceStepResult::kStepBallFly;
        const Vec2 G_mid = make_Vec2(F_mid.x, F_mid.y) - params.delta * length(rk2midpoint.speed) * rk2midpoint.speed;
        const Vec2 a_mid = G_mid + params.alpha * make_Vec2(-M_mid.y, M_mid.x);
        const real_t A_mid = length(a_mid);
        const real_t b_mid = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_mid.z);
        if (!real_isfinite(A_mid) || !real_isfinite(b_mid))
            return AdvanceStepResult::kStepNaNOrInf;
        const real_t g_mid = A_mid - b_mid;

        State<kEddyCurrentModeCount> trial_final = start + dt * Statedot_mid;
        advance_eddy_current_state(start, trial_final, dt, hb_dot_mid, params);
        project_pure_rolling(trial_final, params);
        if (!normalize_attitude(trial_final))
            return AdvanceStepResult::kStepNumericalUnstable;
        if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
            if (!state_in_lut_domain(trial_final))
                return AdvanceStepResult::kStepLUTOutOfDomain;
        }
        const FieldForceResult field_force_final = eval_mag_field_and_force(
            params, trial_final, cache, LUTptr, nullptr);
        const Vec3 &h_final = field_force_final.h;
        const Vec3 F_final = apply_inclined_disk_force(params, field_force_final.Force, trial_final.pos);
        const Vec3 M_final = add_gaussian_rolling_resistance(params, field_force_final.magnetic_angular_acceleration, trial_final, F_final);
        const real_t N_final = params.gamma_g - F_final.z;
        if (!real_isfinite(N_final))
            return AdvanceStepResult::kStepNaNOrInf;
        if (N_final < REAL_LITERAL(0.0))
            return AdvanceStepResult::kStepBallFly;
        const Vec2 G_final =
            make_Vec2(F_final.x, F_final.y) - params.delta * length(trial_final.speed) * trial_final.speed;
        const Vec2 a_final = G_final + params.alpha * make_Vec2(-M_final.y, M_final.x);
        const real_t A_final = length(a_final);
        const real_t b_final = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N_final;
        if (!real_isfinite(A_final) || !real_isfinite(b_final))
            return AdvanceStepResult::kStepNaNOrInf;
        const real_t g_final = A_final - b_final;

        if constexpr (kDebugChkEight) {
            if (g_mid > REAL_LITERAL(0.0) && g_final <= REAL_LITERAL(0.0))
                printf("[DEBUG] Found Event 8: g_mid = %.17g, g_final = %.17g\n", static_cast<double>(g_mid),
                       static_cast<double>(g_final));
        }
        // Ignore a transient midpoint excursion when the endpoint can still roll.
        if (g_final <= REAL_LITERAL(0.0)) {
            if (H_cache != nullptr)
                *H_cache = {h_final, field_force_final.hGrad3d};
            in = trial_final;
            sliding_dir = no_sliding_dir;
            isHCacheValid = H_cache != nullptr;
            return AdvanceStepResult::kStepOk;
        }

        real_t tr;
        if (g_mid > REAL_LITERAL(0.0)) {
            tr = dt * REAL_LITERAL(0.5) * (-g_start) / (g_mid - g_start);
        } else {
            tr = dt * REAL_LITERAL(0.5) + dt * REAL_LITERAL(0.5) * (-g_mid) / (g_final - g_mid);
        }
        if (!real_isfinite(tr))
            return AdvanceStepResult::kStepNaNOrInf;
        if (tr < REAL_LITERAL(0.0) || tr > dt)
            return AdvanceStepResult::kStepNumericalUnstable;

        State<kEddyCurrentModeCount> state_at_tr = start;
        if (tr > REAL_LITERAL(0.0)) {
            State<kEddyCurrentModeCount> state_at_halftr = start + tr * REAL_LITERAL(0.5) * Statedot_start;
            advance_eddy_current_state(start, state_at_halftr, tr * REAL_LITERAL(0.5), hb_dot_start, params);
            if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                if (!state_in_lut_domain(state_at_halftr))
                    return AdvanceStepResult::kStepLUTOutOfDomain;
            }
            State<kEddyCurrentModeCount> dot_estimate;
            Vec3 hb_dot_estimate;
            if (!eval_rhs<false>(state_at_halftr, dot_estimate, hb_dot_estimate, params, krolling, no_sliding_dir, cache, LUTptr,
                                 nullptr))
                return AdvanceStepResult::kStepBallFly;
            state_at_tr = start + tr * dot_estimate;
            advance_eddy_current_state(start, state_at_tr, tr, hb_dot_estimate, params);
            project_pure_rolling(state_at_tr, params);
            if (!normalize_attitude(state_at_tr))
                return AdvanceStepResult::kStepNumericalUnstable;
        }

        if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
            if (!state_in_lut_domain(state_at_tr))
                return AdvanceStepResult::kStepLUTOutOfDomain;
        }
        const FieldForceResult field_force_event = eval_mag_field_and_force(
            params, state_at_tr, cache, LUTptr, nullptr);
        const Vec3 &h_event = field_force_event.h;
        const Vec3 F_event = apply_inclined_disk_force(params, field_force_event.Force, state_at_tr.pos);
        const FieldResult event_field = {h_event, field_force_event.hGrad3d};
        const Vec3 M_event = add_gaussian_rolling_resistance(params, field_force_event.magnetic_angular_acceleration, state_at_tr, F_event);
        const real_t N_event = params.gamma_g - F_event.z;
        if (!real_isfinite(N_event))
            return AdvanceStepResult::kStepNaNOrInf;
        if (N_event < REAL_LITERAL(0.0))
            return AdvanceStepResult::kStepBallFly;
        const Vec2 G_event =
            make_Vec2(F_event.x, F_event.y) - params.delta * length(state_at_tr.speed) * state_at_tr.speed;
        const Vec2 a_event = G_event + params.alpha * make_Vec2(-M_event.y, M_event.x);
        const real_t A_event = length(a_event);
        const real_t b_event = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N_event;
        if (!real_isfinite(A_event) || !real_isfinite(b_event))
            return AdvanceStepResult::kStepNaNOrInf;
        if (A_event == REAL_LITERAL(0.0))
            return AdvanceStepResult::kRetryRollingEventInvalid;
        // The event-to-end segment starts at zero slip. Average the two
        // available driving-force directions for its first RK2 stage.
        const Vec2 dir_sum = a_event / A_event + a_final / A_final;
        const real_t dir_sum_norm = length(dir_sum);
        if (!real_isfinite(dir_sum_norm))
            return AdvanceStepResult::kStepNaNOrInf;
        if (dir_sum_norm == REAL_LITERAL(0.0))
            return AdvanceStepResult::kRetryRollingEventInvalid;
        const Vec2 launch_dir = dir_sum / dir_sum_norm;
        const real_t remaining = dt - tr;
        State<kEddyCurrentModeCount> final_state = state_at_tr;
        Vec2 final_dir = launch_dir;
        bool final_sliding = true;
        if (remaining > REAL_LITERAL(0.0)) {
            State<kEddyCurrentModeCount> dot_sliding_start;
            Vec3 hb_dot_sliding_start;
            if (!eval_rhs<false>(state_at_tr, dot_sliding_start, hb_dot_sliding_start, params, ksliding, launch_dir, cache, LUTptr,
                                 &event_field))
                return AdvanceStepResult::kStepBallFly;
            State<kEddyCurrentModeCount> sliding_mid = state_at_tr + remaining * REAL_LITERAL(0.5) * dot_sliding_start;
            advance_eddy_current_state(state_at_tr, sliding_mid, remaining * REAL_LITERAL(0.5), hb_dot_sliding_start, params);
            // Release starts at zero slip: avoid cancellation between midpoint speed and spin.
            const Vec2 s_mid = (remaining * REAL_LITERAL(0.5) * (a_event - b_event * launch_dir));
            const real_t s_mid_norm = length(s_mid);
            if (!real_isfinite(s_mid_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                if (!state_in_lut_domain(sliding_mid))
                    return AdvanceStepResult::kStepLUTOutOfDomain;
            }
            const FieldForceResult field_force_sliding_mid = eval_mag_field_and_force(
                params, sliding_mid, cache, LUTptr, nullptr);
            const Vec3 &h_sliding_mid = field_force_sliding_mid.h;
            const Vec3 F_sliding_mid = apply_inclined_disk_force(params, field_force_sliding_mid.Force, sliding_mid.pos);
            const FieldResult mid_field = {h_sliding_mid, field_force_sliding_mid.hGrad3d};
            const Vec3 M_sliding_mid = add_gaussian_rolling_resistance(params, field_force_sliding_mid.magnetic_angular_acceleration, sliding_mid, F_sliding_mid);
            const real_t N_sliding_mid = params.gamma_g - F_sliding_mid.z;
            if (!real_isfinite(N_sliding_mid))
                return AdvanceStepResult::kStepNaNOrInf;
            if (N_sliding_mid < REAL_LITERAL(0.0))
                return AdvanceStepResult::kStepBallFly;
            const Vec2 G_sliding_mid = make_Vec2(F_sliding_mid.x, F_sliding_mid.y) -
                                       params.delta * length(sliding_mid.speed) * sliding_mid.speed;
            const Vec2 a_sliding_mid = G_sliding_mid + params.alpha * make_Vec2(-M_sliding_mid.y, M_sliding_mid.x);
            const real_t A_sliding_mid = length(a_sliding_mid);
            const real_t b_sliding_mid = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N_sliding_mid;
            if (!real_isfinite(A_sliding_mid) || !real_isfinite(b_sliding_mid))
                return AdvanceStepResult::kStepNaNOrInf;
            if (A_sliding_mid == REAL_LITERAL(0.0))
                return AdvanceStepResult::kRetryRollingReleaseMidZero;
            if (s_mid_norm == REAL_LITERAL(0.0)) {
                if constexpr (kDebugChkTen) {
                    if (A_sliding_mid <= b_sliding_mid)
                        printf("[DEBUG] Found Event 10: A_mid = %.17g, b_mid = %.17g\n",
                               static_cast<double>(A_sliding_mid), static_cast<double>(b_sliding_mid));
                }
            } else {
                if (A_sliding_mid < b_sliding_mid) {
                    const real_t D_cmid = (b_sliding_mid - A_sliding_mid) * (b_sliding_mid + A_sliding_mid);
                    const real_t Qmid = b_sliding_mid * s_mid_norm + a_sliding_mid * s_mid;
                    if (!real_isfinite(D_cmid) || !real_isfinite(Qmid) ||
                        !real_isfinite(remaining * REAL_LITERAL(0.5) * D_cmid))
                        return AdvanceStepResult::kStepNaNOrInf;
                    if (Qmid <= remaining * REAL_LITERAL(0.5) * D_cmid)
                        return AdvanceStepResult::kRetryRollingReleaseMidContact;
                }
            }
            State<kEddyCurrentModeCount> dot_sliding_mid;
            Vec3 hb_dot_sliding_mid;
            const Vec2 mid_sliding_dir = a_sliding_mid / A_sliding_mid;
            if (!eval_rhs<false>(sliding_mid, dot_sliding_mid, hb_dot_sliding_mid, params, ksliding, mid_sliding_dir, cache, LUTptr,
                                 &mid_field))
                return AdvanceStepResult::kStepBallFly;
            final_state = state_at_tr + remaining * dot_sliding_mid;
            advance_eddy_current_state(state_at_tr, final_state, remaining, hb_dot_sliding_mid, params);
            if (!normalize_attitude(final_state))
                return AdvanceStepResult::kStepNumericalUnstable;
            const Vec2 sf = vec_cast<real_t>(final_state.speed +
                                             params.alpha * make_Vec2(-final_state.omega.y, final_state.omega.x));
            const real_t sf_norm = length(sf);
            if (!real_isfinite(sf_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if (sf_norm == REAL_LITERAL(0.0)) {
                if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                    if (!state_in_lut_domain(final_state))
                        return AdvanceStepResult::kStepLUTOutOfDomain;
                }
                const AdvanceStepResult mode_result =
                    resolve_zero_slip_mode(final_state, params, final_sliding, final_dir, cache, LUTptr);
                if (mode_result != AdvanceStepResult::kStepOk)
                    return mode_result;
            } else {
                final_dir = sf / sf_norm;
            }
        }

        if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
            if (!state_in_lut_domain(final_state))
                return AdvanceStepResult::kStepLUTOutOfDomain;
        }
        if (remaining == REAL_LITERAL(0.0) && H_cache != nullptr)
            *H_cache = event_field;
        in = final_state;
        sliding = final_sliding;
        sliding_dir = final_dir;
        isHCacheValid = remaining == REAL_LITERAL(0.0) && H_cache != nullptr;
        return AdvanceStepResult::kStepOk;
    }
}


// MARK: fwd1step
// Advance one complete interval. A rejected interval is split into its left
// and right halves; the node number encodes the pending right halves.
DEVICE NOINLINE AdvanceStepResult fwd1step(const real_t dt, const SolverParams &params, State<kEddyCurrentModeCount> &in, bool &sliding,
                                           Vec2 &sliding_dir, bool &isHCacheValid, FieldCache *cache = nullptr,
                                           const LUTNode *LUTptr = nullptr, FieldResult *H_cache = nullptr,
                                           RetryDiagnostics *diagnostics = nullptr)
{
    constexpr int max_depth = kStepRefinementRecursionDepth;
    static_assert(max_depth >= 0 && max_depth <= 30, "refinement depth must fit in unsigned int");
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0))
        return AdvanceStepResult::kStepNumericalUnstable;
    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        if (!state_in_lut_domain(in))
            return AdvanceStepResult::kStepLUTOutOfDomain;
    }

    State<kEddyCurrentModeCount> work = in;
    bool work_sliding = sliding;
    Vec2 work_dir = sliding_dir;
    bool work_cache_valid = isHCacheValid && H_cache != nullptr;
    FieldResult work_field{};
    if (work_cache_valid)
        work_field = *H_cache;

    unsigned int node = 1;
    int depth = 0;
    real_t h = dt;
    for (;;) {
        const AdvanceStepResult result =
            try_advance_1_step(h, params, work, work_sliding, work_dir, work_cache_valid, cache, LUTptr,
                               H_cache != nullptr ? &work_field : nullptr, diagnostics);
        if (result == AdvanceStepResult::kStepFinerRetry ||
            (result >= AdvanceStepResult::kRetrySlidingHalfTcZero &&
             result <= AdvanceStepResult::kRetryRollingReleaseFinalZero)) {
            if (diagnostics != nullptr) {
                const int reason =
                    static_cast<int>(result) - static_cast<int>(AdvanceStepResult::kRetrySlidingHalfTcZero);
                if (reason >= 0 && reason < 12)
                    ++diagnostics->counts[reason];
                diagnostics->last_reason = reason;
                diagnostics->last_depth = depth;
                diagnostics->last_dt = h;
                diagnostics->last_state = work;
            }
            const real_t half = h * REAL_LITERAL(0.5);
            if (depth == max_depth || half == REAL_LITERAL(0.0) || half == h)
                return AdvanceStepResult::kStepRefinementRecursionLimit;
            node <<= 1;
            ++depth;
            h = half;
            continue;
        }
        if (result != AdvanceStepResult::kStepOk)
            return result;

        if (!eddy_current_state_finite(work))
            return AdvanceStepResult::kStepNaNOrInf;

        while (node > 1 && (node & 1u)) {
            node >>= 1;
            --depth;
            h *= REAL_LITERAL(2.0);
        }
        if (node == 1) {
            if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
                if (!state_in_lut_domain(work))
                    return AdvanceStepResult::kStepLUTOutOfDomain;
            }
            in = work;
            sliding = work_sliding;
            sliding_dir = work_dir;
            isHCacheValid = work_cache_valid;
            if (H_cache != nullptr && work_cache_valid)
                *H_cache = work_field;
            return AdvanceStepResult::kStepOk;
        }
        ++node;
    }
}
