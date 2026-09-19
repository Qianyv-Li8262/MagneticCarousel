#pragma once
#include "hysteresis_types.hpp"

#define IYPT16_STRINGIZE_IMPL(value) #value
#define IYPT16_STRINGIZE(value) STRINGIZE_IMPL(value)
#define IYPT16_REAL_LITERAL_IMPL(value) REAL_LITERAL(value)
#define IYPT16_REAL_LITERAL(value) IYPT16_REAL_LITERAL_IMPL(value)

#define IYPT16_FIELD_HEIGHT 0.3
#define IYPT16_LUT_LOWER_BOUND_RATIO 0.5
#define IYPT16_LUT_UPPER_BOUND_RATIO 1.5
// Manually configured model parameters.
enum class FieldEvalMode {
    kExact,
    kFourierSpline,
};

enum class LUTCacheMode {
    kNoCache,
    kCacheNode,
    kCacheCoeffs,
};

#ifndef FIELD_EVAL_MODE
#define FIELD_EVAL_MODE 0
#endif

#ifndef LUT_CACHE_MODE
#define LUT_CACHE_MODE 0
#endif

inline constexpr FieldEvalMode kFieldEvalMode =
#if FIELD_EVAL_MODE == 0
    FieldEvalMode::kExact;
#elif FIELD_EVAL_MODE == 1
    FieldEvalMode::kFourierSpline;
#else
#error "Invalid FIELD_EVAL_MODE"
#endif

inline constexpr LUTCacheMode kLUTCacheMode =
#if LUT_CACHE_MODE == 0
    LUTCacheMode::kNoCache;
#elif LUT_CACHE_MODE == 1
    LUTCacheMode::kCacheNode;
#elif LUT_CACHE_MODE == 2
    LUTCacheMode::kCacheCoeffs；
#else
#error "Invalid LUT_CACHE_MODE"
#endif

// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics n/2,3n/2, 5n/2 ... generated for the alternating dipole array.

inline constexpr real_t height =
    IYPT16_REAL_LITERAL(IYPT16_FIELD_HEIGHT); // normalized by k = 2 * pi / lambda, not height_phys / lambda
inline constexpr int magnet_number = 8;
inline constexpr real_t magnet_phi0 = real_two_pi() / magnet_number;
inline constexpr int kFieldFourierModeCount = 2;
inline constexpr int kSplineNodes = 128;
inline constexpr real_t kLUTLowerBoundRatio =
    IYPT16_REAL_LITERAL(IYPT16_LUT_LOWER_BOUND_RATIO); // LUT domain: distance in units of the magnet radius.
inline constexpr real_t kLUTUpperBoundRatio =
    IYPT16_REAL_LITERAL(IYPT16_LUT_UPPER_BOUND_RATIO); // The magnet radius is R = magnet_num / 2.
inline constexpr bool kUseBigEndian =
#ifdef USE_BIG_ENDIAN
    true;
#else
    false;
#endif
