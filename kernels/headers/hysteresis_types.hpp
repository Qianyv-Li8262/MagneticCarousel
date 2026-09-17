#pragma once

#if defined(USE_FP64) && defined(USE_DD)
#error "USE_FP64 and USE_DD are mutually exclusive"
#endif

#if defined(USE_DD) && defined(__CUDACC__)
#error "Double-double is only supported by the CPU backend; CUDA must use FP32 or FP64"
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
#define REAL_FLOOR_IMPL std::floor
#elif defined(USE_FP64) && defined(__CUDACC__)
using real_t = double;
#define REAL_COS_IMPL cos
#define REAL_FABS_IMPL fabs
#define REAL_REMAINDER_IMPL remainder
#define REAL_SIN_IMPL sin
#define REAL_SQRT_IMPL sqrt
#define REAL_FLOOR_IMPL floor
#else
using real_t = float;
#define REAL_COS_IMPL cosf
#define REAL_FABS_IMPL fabsf
#define REAL_REMAINDER_IMPL remainderf
#define REAL_SIN_IMPL sinf
#define REAL_SQRT_IMPL sqrtf
#define REAL_FLOOR_IMPL floorf
#endif

#ifdef USE_KAHAN
#include "numeric_extensions/kahan.hpp"
using accumulator_t = kahan<real_t>;
#else
using accumulator_t = real_t;
#endif

#if defined(USE_DD) && !defined(__CUDACC__)
DEVICE HOST INLINE real_t real_cos(const real_t value)
{
    return dd_cos(value);
}
DEVICE HOST INLINE real_t real_fabs(const real_t value)
{
    return dd_abs(value);
}
DEVICE HOST INLINE real_t real_remainder(const real_t value, const real_t divisor)
{
    return dd_remainder(value, divisor);
}
DEVICE HOST INLINE real_t real_sin(const real_t value)
{
    return dd_sin(value);
}
DEVICE HOST INLINE real_t real_sqrt(const real_t value)
{
    return dd_sqrt(value);
}
DEVICE HOST INLINE bool real_isfinite(const real_t value)
{
    return dd_isfinite(value);
}
DEVICE HOST INLINE const real_t &real_two_pi()
{
    return dd_two_pi();
}
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
DEVICE HOST INLINE real_t real_floor(const real_t value)
{
    return REAL_FLOOR_IMPL(value);
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
#undef REAL_FLOOR_IMPL

#if defined(USE_DD) && !defined(__CUDACC__)
#define REAL_LITERAL(value) real_t(static_cast<double>(value))
#elif defined(USE_FP64)
#define REAL_LITERAL(value) value
#else
#define REAL_LITERAL(value) value##f
#endif

#pragma once

// ============================================================
// Vector types
// ============================================================

template <typename T> struct vec2_t {
    T x;
    T y;
};

template <typename T> struct vec3_t {
    T x;
    T y;
    T z;
};

// 常规动力学量
using vec2 = vec2_t<real_t>;
using vec3 = vec3_t<real_t>;

// 积分器 / 高精度状态量
using state_vec2 = vec2_t<accumulator_t>;
using state_vec3 = vec3_t<accumulator_t>;

// ============================================================
// Addition
// ============================================================

template <typename T, typename U> DEVICE HOST INLINE auto operator+(const vec2_t<T> &a, const vec2_t<U> &b)
{
    using R = decltype(a.x + b.x);

    return vec2_t<R>{a.x + b.x, a.y + b.y};
}

template <typename T, typename U> DEVICE HOST INLINE auto operator+(const vec3_t<T> &a, const vec3_t<U> &b)
{
    using R = decltype(a.x + b.x);

    return vec3_t<R>{a.x + b.x, a.y + b.y, a.z + b.z};
}

// ============================================================
// Subtraction
// ============================================================

template <typename T, typename U> DEVICE HOST INLINE auto operator-(const vec2_t<T> &a, const vec2_t<U> &b)
{
    using R = decltype(a.x - b.x);

    return vec2_t<R>{a.x - b.x, a.y - b.y};
}

template <typename T, typename U> DEVICE HOST INLINE auto operator-(const vec3_t<T> &a, const vec3_t<U> &b)
{
    using R = decltype(a.x - b.x);

    return vec3_t<R>{a.x - b.x, a.y - b.y, a.z - b.z};
}

// ============================================================
// Unary minus
// ============================================================

template <typename T> DEVICE HOST INLINE auto operator-(const vec2_t<T> &a)
{
    using R = decltype(-a.x);

    return vec2_t<R>{-a.x, -a.y};
}

template <typename T> DEVICE HOST INLINE auto operator-(const vec3_t<T> &a)
{
    using R = decltype(-a.x);

    return vec3_t<R>{-a.x, -a.y, -a.z};
}

// ============================================================
// Dot product
//
// vec * vec 点积
// ============================================================

template <typename T, typename U> DEVICE HOST INLINE auto operator*(const vec2_t<T> &a, const vec2_t<U> &b)
{
    return a.x * b.x + a.y * b.y;
}

template <typename T, typename U> DEVICE HOST INLINE auto operator*(const vec3_t<T> &a, const vec3_t<U> &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ============================================================
// Vector * scalar
// ============================================================

template <typename T, typename S> DEVICE HOST INLINE auto operator*(const vec2_t<T> &a, const S &b)
{
    using R = decltype(a.x * b);

    return vec2_t<R>{a.x * b, a.y * b};
}

template <typename T, typename S> DEVICE HOST INLINE auto operator*(const vec3_t<T> &a, const S &b)
{
    using R = decltype(a.x * b);

    return vec3_t<R>{a.x * b, a.y * b, a.z * b};
}

// ============================================================
// Scalar * vector
// ============================================================

template <typename S, typename T> DEVICE HOST INLINE auto operator*(const S &a, const vec2_t<T> &b)
{
    return b * a;
}

template <typename S, typename T> DEVICE HOST INLINE auto operator*(const S &a, const vec3_t<T> &b)
{
    return b * a;
}

// ============================================================
// Vector / scalar
// ============================================================

template <typename T, typename S> DEVICE HOST INLINE auto operator/(const vec2_t<T> &a, const S &b)
{
    using R = decltype(a.x / b);

    return vec2_t<R>{a.x / b, a.y / b};
}

template <typename T, typename S> DEVICE HOST INLINE auto operator/(const vec3_t<T> &a, const S &b)
{
    using R = decltype(a.x / b);

    return vec3_t<R>{a.x / b, a.y / b, a.z / b};
}

// ============================================================
// Constructors
// ============================================================

template <typename T, typename U> DEVICE HOST INLINE auto make_vec2(const T &x, const U &y)
{
    using R = decltype(x + y);

    return vec2_t<R>{static_cast<R>(x), static_cast<R>(y)};
}

template <typename T, typename U, typename V> DEVICE HOST INLINE auto make_vec3(const T &x, const U &y, const V &z)
{
    using R = decltype(x + y + z);

    return vec3_t<R>{static_cast<R>(x), static_cast<R>(y), static_cast<R>(z)};
}

// ============================================================
// Cross product
// ============================================================

// 2D:
//     a × b -> scalar
template <typename T, typename U> DEVICE HOST INLINE auto cross(const vec2_t<T> &a, const vec2_t<U> &b)
{
    return a.x * b.y - a.y * b.x;
}

// 3D:
//     a × b -> vector
template <typename T, typename U> DEVICE HOST INLINE auto cross(const vec3_t<T> &a, const vec3_t<U> &b)
{
    using R = decltype(a.y * b.z - a.z * b.y);

    return vec3_t<R>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// ============================================================
// Length
// ============================================================

template <typename T> DEVICE HOST INLINE auto length_squared(const vec2_t<T> &a)
{
    return a.x * a.x + a.y * a.y;
}

template <typename T> DEVICE HOST INLINE auto length_squared(const vec3_t<T> &a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

template <typename T> DEVICE HOST INLINE auto length(const vec2_t<T> &a)
{
    return real_sqrt(length_squared(a));
}

template <typename T> DEVICE HOST INLINE auto length(const vec3_t<T> &a)
{
    return real_sqrt(length_squared(a));
}

// ============================================================
// 2D rotation
// ============================================================

template <typename T, typename A> DEVICE HOST INLINE auto rotate(const vec2_t<T> &v, const A &angle)
{
    const auto c = real_cos(angle);
    const auto s = real_sin(angle);

    using R = decltype(c * v.x);

    return vec2_t<R>{c * v.x - s * v.y, s * v.x + c * v.y};
}

// ============================================================
// Misc
// ============================================================

template <typename T> DEVICE HOST INLINE T signum(const T &value)
{
    return static_cast<T>((value > static_cast<T>(0)) - (value < static_cast<T>(0)));
}

struct state {
    accumulator_t t;
    state_vec2 pos;
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
    return {a * b.t, a * b.x, a * b.xdot, a * b.theta, a * b.theta_dot, a * b.phase, {a * b.q.x, a * b.q.y}};
}

struct solver_params {
    real_t alpha;
    real_t delta;
    real_t gamma_g;
    real_t gamma;
    real_t gamma_m;
};
