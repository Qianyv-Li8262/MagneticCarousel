#pragma once
#include "hysteresis_types.hpp"
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

inline constexpr FieldEvalMode kFieldEvalMode =
#if FIELD_EVAL_MODE == 0
    FieldEvalMode::kExact;
#elif FIELD_EVAL_MODE == 1
    FieldEvalMode::kFourierSpline;
#else
    FieldEvalMode::kExact;
#endif

// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics n/2,3n/2, 5n/2 ... generated for the alternating dipole array.

inline constexpr real_t height = 0.3; // normalized by k = 2 * pi / lambda, not height_phys / lambda
inline constexpr int magnet_number = 8;
inline constexpr real_t magnet_phi0 = real_two_pi() / magnet_number;
inline constexpr int kFieldFourierModeCount = 2;