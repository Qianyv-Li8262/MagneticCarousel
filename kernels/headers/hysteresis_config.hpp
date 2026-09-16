#pragma once

// Manually configured model parameters.

// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics 1, 3, 5, ... generated for the alternating dipole array.
#define FIELD_Z0 0.25
#define FIELD_FOURIER_MODE_COUNT 5

// Reproduce the circular single-harmonic field used by the numerical kernel
// before ce866ba.  The old kernel used H_EXT as the already-normalized first
// harmonic and used a height coordinate for which dH/dz = -H.  Keep this
// switch explicit: the finite-height dipole coefficient is elliptical and
// therefore cannot be made exactly equivalent by changing H_EXT alone.
#define FIELD_LEGACY_FIRST_HARMONIC_COMPATIBILITY 0

// Fundamental magnetic constants
#define MU_R REAL_LITERAL(2.2926667e3)
#define ALPHA_JA REAL_LITERAL(3.2e-3)
#define C REAL_LITERAL(0.38)
#define K_HYSTER REAL_LITERAL(1.0909091e-3)

#define XI_JA (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + ALPHA_JA * (MU_R - REAL_LITERAL(1.0)))))

// Magnetic field strength
#define H_EXT REAL_LITERAL(0.11504)

// Mechanical parameters
#define ETA REAL_LITERAL(0.4)
#define MU REAL_LITERAL(0.02)
