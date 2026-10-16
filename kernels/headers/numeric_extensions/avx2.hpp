// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "../hysteresis_config.hpp"

#ifndef __CUDACC__
#include <immintrin.h>
#include <limits>

namespace avx2 {

namespace detail {
constexpr double inverse_factorial(const int degree)
{
    double coefficient = 1.0;
    for (int k = 2; k <= degree; ++k)
        coefficient /= k;
    return coefficient;
}

// Compile-time Horner expansion: 1/K! + x/(K+1)! + ... + x^(Order-K)/Order!.
template <int Order, int K>
inline __m256d taylor_tail(const __m256d x)
{
    constexpr double coefficient = inverse_factorial(K);
    if constexpr (K == Order) {
        return _mm256_set1_pd(coefficient);
    } else {
        return _mm256_add_pd(_mm256_set1_pd(coefficient),
                            _mm256_mul_pd(x, taylor_tail<Order, K + 1>(x)));
    }
}

// Build normal 2^n values from four integer exponents using AVX2.
inline __m256d pow2_normal(const __m128i n)
{
    const __m256i biased = _mm256_add_epi64(_mm256_cvtepi32_epi64(n),
                                           _mm256_set1_epi64x(1023));
    return _mm256_castsi256_pd(_mm256_slli_epi64(biased, 52));
}

// Full-range software path. "Exact" distinguishes it from the configurable
// small-input truncation; it is not a correctly-rounded-result guarantee.
// No scalar lanes or transcendental library calls, and no FMA requirement.
template <bool MinusOne>
inline __m256d exp_exact_impl(const __m256d x)
{
    const __m256d one = _mm256_set1_pd(1.0);
    // Bound the integer conversion; restore exceptional inputs at the end.
    const __m256d bounded = _mm256_min_pd(_mm256_max_pd(x, _mm256_set1_pd(-746.0)),
                                          _mm256_set1_pd(710.0));
    const __m256d n = _mm256_round_pd(
        _mm256_mul_pd(bounded, _mm256_set1_pd(1.44269504088896340736)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Split ln(2) so n * ln2_hi is exact over the required exponent range.
    __m256d z = _mm256_sub_pd(bounded,
        _mm256_mul_pd(n, _mm256_set1_pd(0.693145751953125)));
    z = _mm256_sub_pd(z,
        _mm256_mul_pd(n, _mm256_set1_pd(1.42860682030941723212e-6)));
    // |z| <= ln(2)/2: fixed degree-13 Taylor evaluation of expm1(z).
    const __m256d u = _mm256_mul_pd(z, taylor_tail<13, 1>(z));
    const __m128i ni = _mm256_cvttpd_epi32(n);
    const __m128i n1 = _mm_srai_epi32(ni, 1);
    const __m128i n2 = _mm_sub_epi32(ni, n1);
    // Two normal powers cover overflow edges and subnormal results.
    __m256d value = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_add_pd(one, u), pow2_normal(n1)), pow2_normal(n2));
    if constexpr (MinusOne) {
        // Near zero return the small polynomial directly, avoiding exp(x)-1.
        value = _mm256_blendv_pd(_mm256_sub_pd(value, one), u,
                                _mm256_cmp_pd(n, _mm256_setzero_pd(), _CMP_EQ_OQ));
        value = _mm256_blendv_pd(value, x,
                                _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ));
    }
    value = _mm256_blendv_pd(value, _mm256_set1_pd(std::numeric_limits<double>::infinity()),
        _mm256_cmp_pd(x, _mm256_set1_pd(709.782712893383973096), _CMP_GT_OQ));
    value = _mm256_blendv_pd(value, _mm256_set1_pd(MinusOne ? -1.0 : 0.0),
        _mm256_cmp_pd(x, _mm256_set1_pd(-746.0), _CMP_LT_OQ));
    return _mm256_blendv_pd(value, x, _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
}
} // namespace detail

inline __m256d exp_exact(const __m256d x)
{
    return detail::exp_exact_impl<false>(x);
}

inline __m256d expm1_exact(const __m256d x)
{
    return detail::exp_exact_impl<true>(x);
}

// Small-|x| approximation only: no range reduction, clamping or exact fallback.
// Order counts the highest power of x, so order 1 returns x for expm1.
template <int Order>
inline __m256d expm1_expansion(const __m256d x)
{
    static_assert(Order >= 1, "AVX exponential expansion order must be positive");
    return _mm256_mul_pd(x, detail::taylor_tail<Order, 1>(x));
}

template <int Order>
inline __m256d exp_expansion(const __m256d x)
{
    return _mm256_add_pd(_mm256_set1_pd(1.0), expm1_expansion<Order>(x));
}

// Configuration is resolved at compile time; no runtime mode branch.
inline __m256d exp(const __m256d x)
{
    if constexpr (kAvxExpMode == AvxExpMode::kExact)
        return exp_exact(x);
    else
        return exp_expansion<kAvxExpExpansionOrder>(x);
}

inline __m256d expm1(const __m256d x)
{
    if constexpr (kAvxExpMode == AvxExpMode::kExact)
        return expm1_exact(x);
    else
        return expm1_expansion<kAvxExpExpansionOrder>(x);
}

} // namespace avx2
#endif
