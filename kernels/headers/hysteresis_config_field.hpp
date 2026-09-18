#pragma once
// Manually configured model parameters.
enum class FieldEvalMode {
    Precise,
    Fourier_Spline,
};

#ifndef FIELD_EVAL_MODE
#define FIELD_EVAL_MODE 0
#endif

inline constexpr FieldEvalMode field_eval_mode =
#if FIELD_EVAL_MODE == 0
    FieldEvalMode::Precise;
#elif FIELD_EVAL_MODE == 1
    FieldEvalMode::Fourier_Spline;
#else
    FieldEvalMode::Precise;
#endif
// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics 1, 3, 5, ... generated for the alternating dipole array.
#define FIELD_Z0 0.3
inline constexpr int MAGNET_NUM=8;
#define FIELD_FOURIER_MODE_COUNT 4

// Reproduce the circular single-harmonic field used by the numerical kernel
// before ce866ba.  The old kernel used H_EXT as the already-normalized first
// harmonic and used a height coordinate for which dH/dz = -H.  Keep this
// switch explicit: the finite-height dipole coefficient is elliptical and
// therefore cannot be made exactly equivalent by changing H_EXT alone.
#define FIELD_LEGACY_FIRST_HARMONIC_COMPATIBILITY 0