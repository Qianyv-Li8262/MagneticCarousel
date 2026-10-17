// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "hysteresis_types.hpp"
// Active 56c6 reduced-overshoot fit, selected by the user on 2026-10-17.
// Conditional fit values, not measured material properties; M_ref stays 1.1e6 A/m.
// Original Q235 baseline: research/q235/FIXED_PARAMETERS.md.
CONFIG_CONST real_t Mu_r = REAL_LITERAL(144.7026012439737);
CONFIG_CONST real_t Alpha_JA = REAL_LITERAL(0.002081136707547583);
CONFIG_CONST real_t C_JA = REAL_LITERAL(0.4928880388429773);
CONFIG_CONST real_t K = REAL_LITERAL(0.0016276793884410028);

CONFIG_CONST real_t Xi_JA= (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Alpha_JA * (Mu_r - REAL_LITERAL(1.0)))));

// Restore the reference magnet moment on 2026-10-13 at the user's request.
// Beta=0.056999326368103 corresponds to 4.222663 A m^2 per cylinder.
// The earlier 3.04x moment candidate is retained only in historical audit outputs.
CONFIG_CONST real_t Beta = REAL_LITERAL(0.0569993);
// Beta is m k^3 / (4 pi M_s).

// Mechanical parameters
CONFIG_CONST real_t Eta= REAL_LITERAL(0.4);
// MEASURED RESTORE BASELINE (2026-10-15; preserve this comment unchanged):
// Sliding/contact Mu = 0.20.
// kEnableRollingResistance = true;
// kRollingMuLow            = 0.00490;
// kRollingMuHigh           = 0.00490;
// kRollingTransitionOmega  = 1.5;
// kRollingOmegaEpsilon     = 0.002;
// kRollingLambda           = 0.0;
// Source: user's Side-dialogue measurement, confirmed verbatim 2026-10-15.
// Keep this measured block unchanged if active parameters are tuned later;
// restore the active assignments below to these values to recover the data.
CONFIG_CONST real_t Mu = REAL_LITERAL(0.2);
// Mu is the friction coefficient.

// Passive rolling torque: -mu(w)*N*R*omega_xy/sqrt(w^2+epsilon^2) - c*omega_xy.
// mu(w)=mu_inf+(mu_0-mu_inf)*exp(-(w/wc)^2); w is ball spin, not orbital rate.
// Rates use disk-angular time; lambda=c/(I*Omega_disk). No added normal-spin drag.
// Pre-measurement defaults (historical model, not measured):
// low=0.00001; high=0.0001; transition=1.5; epsilon=0.002; lambda=0.004.
// Active 56c6 fit: vc=0.22218599283313184 m/s at R=6mm and 200rpm.
// omega_c_code=vc/(R*Omega_disk); the horizontal spin determines rolling loss.
// These fitted values differ from the measured baseline preserved above.
CONFIG_CONST bool kEnableRollingResistance = true;
CONFIG_CONST real_t kRollingMuLow = REAL_LITERAL(0.007556264150617018);
CONFIG_CONST real_t kRollingMuHigh = REAL_LITERAL(0.005764652328562314);
CONFIG_CONST real_t kRollingTransitionOmega = REAL_LITERAL(1.768099952258668);
CONFIG_CONST real_t kRollingOmegaEpsilon = REAL_LITERAL(0.002);
CONFIG_CONST real_t kRollingLambda = REAL_LITERAL(0.0);


// AVX groups, four scalar memory modes per group (3 groups = 12 modes).
CONFIG_CONST int kEddyCurrentModeCount = 3;

// Set to 1 and rebuild to include eddy-current dynamics and magnetic loads.
// The default-off build retains the legacy 15-value external state format.
#ifndef IYPT16_ENABLE_EDDY_CURRENTS
#define IYPT16_ENABLE_EDDY_CURRENTS 1
#endif
CONFIG_CONST bool kEnableEddyCurrents = IYPT16_ENABLE_EDDY_CURRENTS != 0;
inline constexpr int kSerializedStateSize = 15 + (IYPT16_ENABLE_EDDY_CURRENTS ? 12 * kEddyCurrentModeCount : 0);

enum class AvxExpMode {
    kExact,
    kExpansion,
};

// Exact uses vector range reduction and a fixed high-order polynomial.
// Expansion is a direct small-|x| Taylor approximation without range reduction;
// order 1 means exp(x) ~= 1 + x. Both paths avoid scalar exp/expm1 calls.
CONFIG_CONST AvxExpMode kAvxExpMode = AvxExpMode::kExact;
CONFIG_CONST int kAvxExpExpansionOrder = 1;
static_assert(kAvxExpExpansionOrder >= 1, "AVX exponential expansion order must be positive");

// Fixed tilt axis measured from +x; positive tilt follows the right-hand rule.
// Small-angle model: add theta * (gamma_g - F_mag.z) along the downhill direction.
// This assumes the normal magnetic attraction contributes to the downslope drive.
// Only the horizontal force changes; the normal load and magnetic torque do not.
// 2026-10-15: horizontal-disk period fit; no added normal-spin damping.
CONFIG_CONST bool kEnableInclinedDisk = false;
CONFIG_CONST real_t kTiltAxisAzimuthRad = REAL_LITERAL(1.5707963267948966); // downhill +x; laboratory azimuth unspecified
CONFIG_CONST real_t kTiltAngleRad = REAL_LITERAL(0.0021505243734457795); // Trial at allowed boundary: 4 mm / 930 mm reflected path

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
