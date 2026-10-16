// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "../generated/eddy_current_coefficients.generated.hpp"
#include "hysteresis_config.hpp"
#include "hysteresis_config_field.hpp"
#include "hysteresis_types.hpp"


#ifndef __CUDACC__
#include "numeric_extensions/avx2.hpp"

#include <cstdio>

#include <immintrin.h>


// Advance four modes in place, freezing dh_body/dt_star over this substep.
// inv_tau_star contains runtime rates: inv_tau_ref_star / eddy_current_scale.
// dt and hb_dot use dimensionless disk-angular time. Requires finite dt >= 0
// and finite positive rates. The three vectors must refer to distinct storage.
INLINE void advance_eddy_current_avx(__m256d &yx, __m256d &yy, __m256d &yz, const double dt, const Vec3 &hb_dot,
                                     const __m256d &inv_tau_star)
{
    if (dt == 0.0)
        return;

    const __m256d r = _mm256_mul_pd(_mm256_set1_pd(dt), inv_tau_star);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d minus_r = _mm256_sub_pd(zero, r);
    const __m256d decay = avx2::exp(minus_r);
    const __m256d is_zero = _mm256_cmp_pd(r, zero, _CMP_EQ_OQ);
    // Avoid 0/0 before selecting the limit g(0) = 1.
    const __m256d denominator = _mm256_blendv_pd(r, one, is_zero);
    const __m256d g =
        _mm256_blendv_pd(_mm256_div_pd(_mm256_sub_pd(zero, avx2::expm1(minus_r)), denominator), one, is_zero);
    const __m256d gain = _mm256_mul_pd(_mm256_set1_pd(dt), g);

    // y_new = exp(-r) * y_old - dt * (1 - exp(-r)) / r * hb_dot.
    yx = _mm256_sub_pd(_mm256_mul_pd(decay, yx), _mm256_mul_pd(gain, _mm256_set1_pd(static_cast<double>(hb_dot.x))));
    yy = _mm256_sub_pd(_mm256_mul_pd(decay, yy), _mm256_mul_pd(gain, _mm256_set1_pd(static_cast<double>(hb_dot.y))));
    yz = _mm256_sub_pd(_mm256_mul_pd(decay, yz), _mm256_mul_pd(gain, _mm256_set1_pd(static_cast<double>(hb_dot.z))));
}
#else
DEVICE INLINE void advance_eddy_current_cuda(real_t *yx, real_t *yy, real_t *yz, const real_t dt, const Vec3 &hb_dot,
                                             const real_t eddy_current_scale)
{
    if (dt == REAL_LITERAL(0.0))
        return;

    for (int j = 0; j < eddy_coefficients::mode_count; ++j) {
        const real_t rate = static_cast<real_t>(eddy_coefficients::inv_tau_ref_star[j]) / eddy_current_scale;
        const real_t r = dt * rate;
        const real_t decay = exp(-r);
        const real_t gain = dt * (r == REAL_LITERAL(0.0) ? REAL_LITERAL(1.0) : -expm1(-r) / r);
        yx[j] = decay * yx[j] - gain * hb_dot.x;
        yy[j] = decay * yy[j] - gain * hb_dot.y;
        yz[j] = decay * yz[j] - gain * hb_dot.z;
    }
}
#endif

#ifndef __CUDACC__
using EddyCurrentModes = __m256d[kEddyCurrentModeCount];
#else
using EddyCurrentModes = real_t[4 * kEddyCurrentModeCount];
#endif

// Compile-time dispatch for the CPU AVX and CUDA scalar State array layouts.
// Same input convention as advance_eddy_current_avx; scale must be positive.
DEVICE INLINE void advance_eddy_current(EddyCurrentModes &yx, EddyCurrentModes &yy, EddyCurrentModes &yz,
                                        const real_t dt, const Vec3 &hb_dot, const real_t eddy_current_scale)
{
    static_assert(eddy_coefficients::mode_count == 4 * kEddyCurrentModeCount,
                  "eddy coefficients are stale; rebuild generate_eddy_coefficients");
#ifndef __CUDACC__
    const __m256d scale = _mm256_set1_pd(static_cast<double>(eddy_current_scale));
#pragma unroll
    for (int group = 0; group < kEddyCurrentModeCount; ++group) {
        const __m256d rates = _mm256_div_pd(_mm256_load_pd(eddy_coefficients::inv_tau_ref_star + 4 * group), scale);
        advance_eddy_current_avx(yx[group], yy[group], yz[group], static_cast<double>(dt), hb_dot, rates);
    }
#else
    advance_eddy_current_cuda(yx, yy, yz, dt, hb_dot, eddy_current_scale);
#endif
}

#ifndef __CUDACC__
INLINE double eddy_horizontal_sum(const __m256d value)
{
    const __m128d pairs = _mm_add_pd(_mm256_castpd256_pd128(value),
                                    _mm256_extractf128_pd(value, 1));
    return _mm_cvtsd_f64(_mm_add_sd(pairs, _mm_unpackhi_pd(pairs, pairs)));
}

INLINE Vec3 eval_eddy_current_moment_avx(const EddyCurrentModes &yx, const EddyCurrentModes &yy,
                                        const EddyCurrentModes &yz)
{
    __m256d sx = _mm256_setzero_pd();
    __m256d sy = _mm256_setzero_pd();
    __m256d sz = _mm256_setzero_pd();
    for (int group = 0; group < kEddyCurrentModeCount; ++group) {
        const __m256d a = _mm256_load_pd(eddy_coefficients::aj + 4 * group);
        sx = _mm256_add_pd(sx, _mm256_mul_pd(a, yx[group]));
        sy = _mm256_add_pd(sy, _mm256_mul_pd(a, yy[group]));
        sz = _mm256_add_pd(sz, _mm256_mul_pd(a, yz[group]));
    }
    return {static_cast<real_t>(eddy_horizontal_sum(sx)),
            static_cast<real_t>(eddy_horizontal_sum(sy)),
            static_cast<real_t>(eddy_horizontal_sum(sz))};
}
#else
DEVICE INLINE Vec3 eval_eddy_current_moment_cuda(const real_t *yx, const real_t *yy, const real_t *yz)
{
    Vec3 delta_m{};
    for (int j = 0; j < eddy_coefficients::mode_count; ++j) {
        const real_t a = static_cast<real_t>(eddy_coefficients::aj[j]);
        delta_m.x += a * yx[j];
        delta_m.y += a * yy[j];
        delta_m.z += a * yz[j];
    }
    return delta_m;
}
#endif

// Body-frame dimensionless magnetization increment: delta_m = sum_j aj[j] * y[j].
// The physical dipole moment is V * M_ref * delta_m; no volume factor is applied here.
DEVICE INLINE Vec3 eval_eddy_current_moment(const EddyCurrentModes &yx, const EddyCurrentModes &yy,
                                           const EddyCurrentModes &yz)
{
    static_assert(eddy_coefficients::mode_count == 4 * kEddyCurrentModeCount,
                  "eddy coefficients are stale; rebuild generate_eddy_coefficients");
#ifndef __CUDACC__
    return eval_eddy_current_moment_avx(yx, yy, yz);
#else
    return eval_eddy_current_moment_cuda(yx, yy, yz);
#endif
}
