#pragma once

#if defined(USE_FP64) && defined(USE_DD)
#error "USE_FP64 and USE_DD are mutually exclusive"
#endif

#if (defined(USE_FP64) || defined(USE_DD)) && defined(__CUDACC__)
#error "FP64 and double-double are only supported by the CPU backend; CUDA must use FP32"
#endif

// Macros for CUDA and CPU C++ backends.
#ifdef __CUDACC__
#define DEVICE __device__
#define GLOBAL __global__
#define HOST __host__
#define INLINE __forceinline__
#define NOINLINE __noinline__
#define REAL_ISFINITE_IMPL isfinite
#else
#define DEVICE
#define GLOBAL
#define HOST
#define INLINE inline
#define NOINLINE
#define REAL_ISFINITE_IMPL std::isfinite
#endif

#ifndef __CUDACC__
#include <cmath>
using std::cosf;
using std::fabsf;
using std::remainderf;
using std::sinf;
using std::sqrtf;
#endif

#if defined(USE_DD) && !defined(__CUDACC__)
#include "numeric_extensions/double_double.hpp"
using real_t = dd_real;
#elif defined(USE_FP64) && !defined(__CUDACC__)
using real_t = double;
#define REAL_COS_IMPL std::cos
#define REAL_FABS_IMPL std::fabs
#define REAL_REMAINDER_IMPL std::remainder
#define REAL_SIN_IMPL std::sin
#define REAL_SQRT_IMPL std::sqrt
#else
using real_t = float;
#define REAL_COS_IMPL cosf
#define REAL_FABS_IMPL fabsf
#define REAL_REMAINDER_IMPL remainderf
#define REAL_SIN_IMPL sinf
#define REAL_SQRT_IMPL sqrtf
#endif

#ifdef USE_KAHAN
#include "numeric_extensions/kahan.hpp"
using accumulator_t = kahan<real_t>;
#else
using accumulator_t = real_t;
#endif

#if defined(USE_DD) && !defined(__CUDACC__)
DEVICE HOST INLINE real_t real_cos(const real_t value) { return dd_cos(value); }
DEVICE HOST INLINE real_t real_fabs(const real_t value) { return dd_abs(value); }
DEVICE HOST INLINE real_t real_remainder(const real_t value, const real_t divisor) { return dd_remainder(value, divisor); }
DEVICE HOST INLINE real_t real_sin(const real_t value) { return dd_sin(value); }
DEVICE HOST INLINE real_t real_sqrt(const real_t value) { return dd_sqrt(value); }
DEVICE HOST INLINE bool real_isfinite(const real_t value) { return dd_isfinite(value); }
DEVICE HOST INLINE const real_t &real_two_pi() { return dd_two_pi(); }
#else
DEVICE HOST INLINE real_t real_cos(const real_t value)
{
    return REAL_COS_IMPL(value);
}
DEVICE HOST INLINE real_t real_fabs(const real_t value)
{
    return REAL_FABS_IMPL(value);
}
DEVICE HOST INLINE real_t real_remainder(const real_t value, const real_t divisor)
{
    return REAL_REMAINDER_IMPL(value, divisor);
}
DEVICE HOST INLINE real_t real_sin(const real_t value)
{
    return REAL_SIN_IMPL(value);
}
DEVICE HOST INLINE real_t real_sqrt(const real_t value)
{
    return REAL_SQRT_IMPL(value);
}
DEVICE HOST INLINE bool real_isfinite(const real_t value)
{
    return REAL_ISFINITE_IMPL(value);
}
DEVICE HOST INLINE real_t real_two_pi()
{
    return static_cast<real_t>(6.28318530717958647692);
}
#endif

#undef REAL_COS_IMPL
#undef REAL_FABS_IMPL
#undef REAL_REMAINDER_IMPL
#undef REAL_SIN_IMPL
#undef REAL_SQRT_IMPL

#if defined(USE_DD) && !defined(__CUDACC__)
#define REAL_LITERAL(value) real_t(static_cast<double>(value))
#elif defined(USE_FP64) && !defined(__CUDACC__)
#define REAL_LITERAL(value) value
#else
#define REAL_LITERAL(value) value##f
#endif

struct vec2 {
    real_t x;
    real_t y;
};

DEVICE HOST INLINE vec2 operator+(const vec2 &a, const vec2 &b)
{
    return {a.x + b.x, a.y + b.y};
}

DEVICE HOST INLINE vec2 operator-(const vec2 &a, const vec2 &b)
{
    return {a.x - b.x, a.y - b.y};
}

DEVICE HOST INLINE real_t operator*(const vec2 &a, const vec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

DEVICE HOST INLINE vec2 operator*(const vec2 &a, const real_t b)
{
    return {b * a.x, b * a.y};
}

DEVICE HOST INLINE vec2 operator/(const vec2 &a, const real_t b)
{
    return {a.x / b, a.y / b};
}

DEVICE HOST INLINE vec2 operator*(const real_t b, const vec2 &a)
{
    return a * b;
}

DEVICE HOST INLINE vec2 make_vec2(const real_t a, const real_t b)
{
    return {a, b};
}

DEVICE HOST INLINE real_t cross(const vec2 &a, const vec2 &b)
{
    return a.x * b.y - a.y * b.x;
}

DEVICE HOST INLINE real_t signum(const real_t value)
{
    return static_cast<real_t>((value > REAL_LITERAL(0.0)) - (value < REAL_LITERAL(0.0)));
}

DEVICE HOST INLINE vec2 rotate(const vec2 vec, const real_t angle)
{
    // 逆时针旋转矢量angle弧度
    return {real_cos(angle) * vec.x - real_sin(angle) * vec.y, real_sin(angle) * vec.x + real_cos(angle) * vec.y};
}

DEVICE HOST INLINE real_t length(const vec2 &a)
{
    return real_sqrt(a.x * a.x + a.y * a.y);
}

struct state_vec2 {
    accumulator_t x;
    accumulator_t y;
};

DEVICE HOST INLINE real_t operator*(const state_vec2 &a, const vec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

DEVICE HOST INLINE vec2 operator*(const state_vec2 &a, const real_t b)
{
    return {b * a.x, b * a.y};
}

DEVICE HOST INLINE real_t cross(const state_vec2 &a, const vec2 &b)
{
    return a.x * b.y - a.y * b.x;
}

DEVICE HOST INLINE real_t length(const state_vec2 &a)
{
    return real_sqrt(a.x * a.x + a.y * a.y);
}

struct state {
    accumulator_t t;
    accumulator_t x;
    accumulator_t xdot;
    accumulator_t theta;
    accumulator_t theta_dot;
    accumulator_t phase;
    state_vec2 q;
};

DEVICE HOST INLINE state operator+(const state &a, const state &b)
{
    return {a.t + b.t,
            a.x + b.x,
            a.xdot + b.xdot,
            a.theta + b.theta,
            a.theta_dot + b.theta_dot,
            a.phase + b.phase,
            {a.q.x + b.q.x, a.q.y + b.q.y}};
}

DEVICE HOST INLINE state operator*(const real_t a, const state &b)
{
    return {a * b.t,
            a * b.x,
            a * b.xdot,
            a * b.theta,
            a * b.theta_dot,
            a * b.phase,
            {a * b.q.x, a * b.q.y}};
}

struct solver_params {
    real_t alpha;
    real_t delta;
    real_t gamma_g;
    real_t gamma;
    real_t gamma_m;
};
