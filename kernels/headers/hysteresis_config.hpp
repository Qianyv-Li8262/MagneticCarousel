// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "hysteresis_types.hpp"
// Fundamental magnetic constants
CONFIG_CONST real_t Mu_r =REAL_LITERAL(4138.55);
CONFIG_CONST real_t Alpha_JA= REAL_LITERAL(7.0921e-4);
CONFIG_CONST real_t C_JA= REAL_LITERAL(0.0198);
CONFIG_CONST real_t K =REAL_LITERAL(1.04963636364e-3);

CONFIG_CONST real_t Xi_JA= (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Alpha_JA * (Mu_r - REAL_LITERAL(1.0)))));

// Magnetic dipole strength
CONFIG_CONST real_t Beta = REAL_LITERAL(0.056999326368103);
// Beta is m k^3 / (4 pi M_s).

// Mechanical parameters
CONFIG_CONST real_t Eta= REAL_LITERAL(0.4);
CONFIG_CONST real_t Mu = REAL_LITERAL(0.02);
// Mu is the friction coefficient.

// Fixed tilt axis measured from +x; positive tilt follows the right-hand rule.
// Small-angle model: add theta * (gamma_g - F_mag.z) along the downhill direction.
// This assumes the normal magnetic attraction contributes to the downslope drive.
// Only the horizontal force changes; the normal load and magnetic torque do not.
CONFIG_CONST bool kEnableInclinedDisk = true;
CONFIG_CONST real_t kTiltAxisAzimuthRad = REAL_LITERAL(1.35); // 90 degrees
CONFIG_CONST real_t kTiltAngleRad = REAL_LITERAL(0.02); // 0.5 degrees

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
