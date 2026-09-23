#pragma once
#include "hysteresis_types.hpp"
// Fundamental magnetic constants
inline constexpr real_t Mu_r =REAL_LITERAL(2.2926667e3);
inline constexpr real_t Alpha_JA= REAL_LITERAL(3.2e-3);
inline constexpr real_t C_JA= REAL_LITERAL(0.38);
inline constexpr real_t K =REAL_LITERAL(1.0909091e-3);

inline constexpr real_t Xi_JA= (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Alpha_JA * (Mu_r - REAL_LITERAL(1.0)))));

// Magnetic dipole strength
inline constexpr real_t Beta =REAL_LITERAL(0.11504); // this is \frac{m k^3}{4\pi M_s} 

// Mechanical parameters
inline constexpr real_t Eta= REAL_LITERAL(0.4);
inline constexpr real_t Mu =REAL_LITERAL(0.02); // friction

inline constexpr int kStepRefinementRecursionLimit = 5;

enum class AttitudeNormalizeMode {
    kExact,
    kNewtonOnce,
};

inline constexpr AttitudeNormalizeMode kAttitudeNormalizeMode = AttitudeNormalizeMode::kNewtonOnce;
