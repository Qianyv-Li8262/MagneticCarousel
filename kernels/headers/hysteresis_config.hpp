#pragma once
#include "hysteresis_types.hpp"
// Fundamental magnetic constants
CONFIG_CONST real_t Mu_r =REAL_LITERAL(2.2926667e3);
CONFIG_CONST real_t Alpha_JA= REAL_LITERAL(3.2e-3);
CONFIG_CONST real_t C_JA= REAL_LITERAL(0.38);
CONFIG_CONST real_t K =REAL_LITERAL(1.0909091e-3);

CONFIG_CONST real_t Xi_JA= (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Alpha_JA * (Mu_r - REAL_LITERAL(1.0)))));

// Magnetic dipole strength
CONFIG_CONST real_t Beta = REAL_LITERAL(0.079370775450601);
// Beta is m k^3 / (4 pi M_s).

// Mechanical parameters
CONFIG_CONST real_t Eta= REAL_LITERAL(0.4);
CONFIG_CONST real_t Mu = REAL_LITERAL(0.02);
// Mu is the friction coefficient.

CONFIG_CONST int kStepRefinementRecursionDepth = 30;

enum class AttitudeNormalizeMode {
    kExact,
    kNewtonOnce,
};

CONFIG_CONST AttitudeNormalizeMode kAttitudeNormalizeMode = AttitudeNormalizeMode::kNewtonOnce;

CONFIG_CONST bool kDebugChkEight = false;
CONFIG_CONST bool kDebugChkTen = false;
// Diagnostic switch for the near-zero-slip step-size guard. The sliding
// midpoint direction is predicted separately when this guard is disabled.
CONFIG_CONST bool kFinerStepCheck = false;
