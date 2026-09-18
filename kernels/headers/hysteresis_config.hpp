#pragma once

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
