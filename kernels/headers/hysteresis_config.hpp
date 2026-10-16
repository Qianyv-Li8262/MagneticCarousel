// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "hysteresis_types.hpp"
// Fixed Q235 linearized Fig. 10 baseline, selected by the user on 2026-10-06.
// See research/q235/FIXED_PARAMETERS.md; M_ref remains 1.1e6 A/m.
// Baseline values retained here; user reauthorized isolated material scans on 2026-10-13.
CONFIG_CONST real_t Mu_r = REAL_LITERAL(605.8974337495617);
CONFIG_CONST real_t Alpha_JA = REAL_LITERAL(0.0021323083849413885);
CONFIG_CONST real_t C_JA = REAL_LITERAL(0.22415154978863677);
CONFIG_CONST real_t K = REAL_LITERAL(0.005375927782155882);

CONFIG_CONST real_t Xi_JA= (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Alpha_JA * (Mu_r - REAL_LITERAL(1.0)))));

// Restore the reference magnet moment on 2026-10-13 at the user's request.
// Beta=0.056999326368103 corresponds to 4.222663 A m^2 per cylinder.
// The earlier 3.04x moment candidate is retained only in historical audit outputs.
CONFIG_CONST real_t Beta = REAL_LITERAL(0.0854985);
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
// Equal measured low/high remove the Gaussian speed dependence; lambda=0
// adds no linear rolling loss. The near-zero regularization is retained.
CONFIG_CONST bool kEnableRollingResistance = true;
CONFIG_CONST real_t kRollingMuLow = REAL_LITERAL(0.00490);
CONFIG_CONST real_t kRollingMuHigh = REAL_LITERAL(0.00490);
CONFIG_CONST real_t kRollingTransitionOmega = REAL_LITERAL(1.5);
CONFIG_CONST real_t kRollingOmegaEpsilon = REAL_LITERAL(0.002);
CONFIG_CONST real_t kRollingLambda = REAL_LITERAL(0.0);


// AVX groups, four scalar memory modes per group (3 groups = 12 modes).
CONFIG_CONST int kEddyCurrentModeCount = 3;

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
