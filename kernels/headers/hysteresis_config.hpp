#pragma once

// Manually configured model parameters.

// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics 1, 3, 5, ... generated for the alternating dipole array.
#define FIELD_Z0 0.3
#define FIELD_FOURIER_MODE_COUNT 5

// Fundamental magnetic constants
#define MU_R REAL_LITERAL(2.2926667e3)
#define ALPHA_JA REAL_LITERAL(3.2e-3)
#define C REAL_LITERAL(0.38)
#define K_HYSTER REAL_LITERAL(1.0909091e-3)

#define XI_JA (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + ALPHA_JA * (MU_R - REAL_LITERAL(1.0)))))

// Magnetic field strength
#define H_EXT REAL_LITERAL(1.4661546e-1)

// Mechanical parameters
#define ETA REAL_LITERAL(0.4)
#define MU REAL_LITERAL(0.02)
