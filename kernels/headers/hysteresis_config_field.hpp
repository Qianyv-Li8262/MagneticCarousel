#pragma once
#include "hysteresis_types.hpp"

#define IYPT16_STRINGIZE_IMPL(value) #value
#define IYPT16_STRINGIZE(value) IYPT16_STRINGIZE_IMPL(value)
#define IYPT16_REAL_LITERAL_IMPL(value) REAL_LITERAL(value)
#define IYPT16_REAL_LITERAL(value) IYPT16_REAL_LITERAL_IMPL(value)

#define IYPT16_FIELD_HEIGHT 1.610066235
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

CONFIG_CONST FieldEvalMode kFieldEvalMode =
#if FIELD_EVAL_MODE == 0
    FieldEvalMode::kExact;
#elif FIELD_EVAL_MODE == 1
    FieldEvalMode::kFourierSpline;
#else
#error "Invalid FIELD_EVAL_MODE"
#endif

CONFIG_CONST LUTCacheMode kLUTCacheMode =
#if LUT_CACHE_MODE == 0
    LUTCacheMode::kNoCache;
#elif LUT_CACHE_MODE == 1
    LUTCacheMode::kCacheNode;
#elif LUT_CACHE_MODE == 2
    LUTCacheMode::kCacheCoeffs;
#else
#error "Invalid LUT_CACHE_MODE"
#endif

// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics n/2,3n/2, 5n/2 ... generated for the alternating dipole array.

CONFIG_CONST real_t height =
    IYPT16_REAL_LITERAL(IYPT16_FIELD_HEIGHT); // normalized by k = 2 * pi / lambda, not height_phys / lambda
CONFIG_CONST int magnet_number = 8;
CONFIG_CONST real_t magnet_phi0 = real_two_pi() / magnet_number;
CONFIG_CONST int kFieldFourierModeCount = 7;
// Keep this manual. The LUT spans [0.5R, 1.5R] with R = magnet_number / 2.
// To preserve the 8-magnet (128-node) radial spacing when changing N,
// choose approximately 1 + round(127 * N / 8) nodes (e.g. N=16 -> 255).
CONFIG_CONST int kSplineNodes = 128;
CONFIG_CONST real_t kLUTLowerBoundRatio =
    IYPT16_REAL_LITERAL(IYPT16_LUT_LOWER_BOUND_RATIO); // LUT domain: distance in units of the magnet radius.
CONFIG_CONST real_t kLUTUpperBoundRatio =
    IYPT16_REAL_LITERAL(IYPT16_LUT_UPPER_BOUND_RATIO); // The magnet radius is R = magnet_num / 2.
CONFIG_CONST bool kUseBigEndian =
#ifdef USE_BIG_ENDIAN
    true;
#else
    false;
#endif
