// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once

// Manually configured model parameters.

// Field geometry and Fourier truncation. The mode count refers to the odd
// harmonics 1, 3, 5, ... generated for the alternating dipole array.
#define FIELD_Z0 0.25625
#define FIELD_FOURIER_MODE_COUNT 5

// Reproduce the circular single-harmonic field used by the numerical kernel
// before ce866ba.  The old kernel used H_EXT as the already-normalized first
// harmonic and used a height coordinate for which dH/dz = -H.  Keep this
// switch explicit: the finite-height dipole coefficient is elliptical and
// therefore cannot be made exactly equivalent by changing H_EXT alone.
#define FIELD_LEGACY_FIRST_HARMONIC_COMPATIBILITY 0

// Fixed Q235 linearized Fig. 10 baseline, selected by the user on 2026-10-06.
// See research/q235/FIXED_PARAMETERS.md; M_ref remains 1.1e6 A/m.
// Keep these four material parameters fixed in subsequent exploration.
#define MU_R REAL_LITERAL(605.8974337495617)
#define ALPHA_JA REAL_LITERAL(0.0021323083849413885)
#define C REAL_LITERAL(0.22415154978863677)
#define K_HYSTER REAL_LITERAL(0.005375927782155882)

#define XI_JA (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + ALPHA_JA * (MU_R - REAL_LITERAL(1.0)))))

// Magnetic field strength
#define H_EXT REAL_LITERAL(0.10105801)

// Mechanical parameters
#define ETA REAL_LITERAL(0.4)
#define MU REAL_LITERAL(0.02)
