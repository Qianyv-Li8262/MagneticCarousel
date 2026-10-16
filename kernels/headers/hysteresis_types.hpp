// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "hysteresis_physics.hpp"
#ifndef __CUDACC__
#include <immintrin.h>
#endif
// MARK: Errors
#if defined(USE_FP64) && defined(USE_DD)
#error "USE_FP64 and USE_DD are mutually exclusive"
#endif

#if defined(USE_DD) && defined(__CUDACC__)
#error "Double-double is only supported by the CPU backend; CUDA must use FP32 or FP64"
#endif

// MARK: Macros
// Macros for CUDA and CPU C++ backends.
#ifdef __CUDACC__
#define DEVICE __device__
#define GLOBAL __global__
#define HOST __host__
#define INLINE __forceinline__
#define NOINLINE __noinline__
#define CONFIG_CONST __device__ constexpr
#define REAL_ISFINITE_IMPL isfinite
#else
#define DEVICE
#define GLOBAL
#define HOST
#define INLINE inline
#define NOINLINE
#define CONFIG_CONST inline constexpr
#define REAL_ISFINITE_IMPL std::isfinite
#endif



//MARK: Real Maths
#ifndef __CUDACC__
#include <cmath>
using std::cosf;
using std::atan2f;
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
#define REAL_ATAN2_IMPL std::atan2
#define REAL_FABS_IMPL std::fabs
#define REAL_REMAINDER_IMPL std::remainder
#define REAL_SIN_IMPL std::sin
#define REAL_SQRT_IMPL std::sqrt
#define REAL_FLOOR_IMPL std::floor
#elif defined(USE_FP64) && defined(__CUDACC__)
using real_t = double;
#define REAL_COS_IMPL cos
#define REAL_ATAN2_IMPL atan2
#define REAL_FABS_IMPL fabs
#define REAL_REMAINDER_IMPL remainder
#define REAL_SIN_IMPL sin
#define REAL_SQRT_IMPL sqrt
#define REAL_FLOOR_IMPL floor
#else
using real_t = float;
#define REAL_COS_IMPL cosf
#define REAL_ATAN2_IMPL atan2f
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
DEVICE HOST INLINE real_t real_atan2(const real_t y, const real_t x)
{
    return dd_atan2(y, x);
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
DEVICE HOST INLINE real_t real_atan2(const real_t y, const real_t x)
{
    return REAL_ATAN2_IMPL(y, x);
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
DEVICE HOST INLINE constexpr real_t real_two_pi()
{
    return static_cast<real_t>(6.28318530717958647692);
}
#endif

#undef REAL_COS_IMPL
#undef REAL_ATAN2_IMPL
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




























// MARK: Vectors

// ============================================================
// Vector types
// ============================================================

template <typename T> struct Vec2_t {
    T x;
    T y;

    // Allow component-wise conversion at state / ordinary-vector boundaries.
    template <typename U> DEVICE HOST INLINE operator Vec2_t<U>() const
    {
        return {static_cast<U>(x), static_cast<U>(y)};
    }
};

template <typename T> struct Vec3_t {
    T x;
    T y;
    T z;

    template <typename U> DEVICE HOST INLINE operator Vec3_t<U>() const
    {
        return {static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)};
    }
};

// 常规动力学量
using Vec2 = Vec2_t<real_t>;
using Vec3 = Vec3_t<real_t>;

// 积分器 / 高精度状态量
using StateVec2 = Vec2_t<accumulator_t>;
using StateVec3 = Vec3_t<accumulator_t>;

template <typename T, typename U> DEVICE HOST INLINE auto operator+(const Vec2_t<T> &a, const Vec2_t<U> &b)
{
    using R = decltype(a.x + b.x);

    return Vec2_t<R>{a.x + b.x, a.y + b.y};
}

template <typename T, typename U> DEVICE HOST INLINE auto operator+(const Vec3_t<T> &a, const Vec3_t<U> &b)
{
    using R = decltype(a.x + b.x);

    return Vec3_t<R>{a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename T, typename U> DEVICE HOST INLINE auto operator-(const Vec2_t<T> &a, const Vec2_t<U> &b)
{
    using R = decltype(a.x - b.x);

    return Vec2_t<R>{a.x - b.x, a.y - b.y};
}

template <typename T, typename U> DEVICE HOST INLINE auto operator-(const Vec3_t<T> &a, const Vec3_t<U> &b)
{
    using R = decltype(a.x - b.x);

    return Vec3_t<R>{a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T> DEVICE HOST INLINE auto operator-(const Vec2_t<T> &a)
{
    using R = decltype(-a.x);

    return Vec2_t<R>{-a.x, -a.y};
}

template <typename T> DEVICE HOST INLINE auto operator-(const Vec3_t<T> &a)
{
    using R = decltype(-a.x);

    return Vec3_t<R>{-a.x, -a.y, -a.z};
}

template <typename T, typename U> DEVICE HOST INLINE auto operator*(const Vec2_t<T> &a, const Vec2_t<U> &b)
{
    return a.x * b.x + a.y * b.y;
}

template <typename T, typename U> DEVICE HOST INLINE auto operator*(const Vec3_t<T> &a, const Vec3_t<U> &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T, typename S> DEVICE HOST INLINE auto operator*(const Vec2_t<T> &a, const S &b)
{
    using R = decltype(a.x * b);

    return Vec2_t<R>{a.x * b, a.y * b};
}

template <typename T, typename S> DEVICE HOST INLINE auto operator*(const Vec3_t<T> &a, const S &b)
{
    using R = decltype(a.x * b);

    return Vec3_t<R>{a.x * b, a.y * b, a.z * b};
}

template <typename S, typename T> DEVICE HOST INLINE auto operator*(const S &a, const Vec2_t<T> &b)
{
    return b * a;
}

template <typename S, typename T> DEVICE HOST INLINE auto operator*(const S &a, const Vec3_t<T> &b)
{
    return b * a;
}

template <typename T, typename S> DEVICE HOST INLINE auto operator/(const Vec2_t<T> &a, const S &b)
{
    using R = decltype(a.x / b);

    return Vec2_t<R>{a.x / b, a.y / b};
}

template <typename T, typename S> DEVICE HOST INLINE auto operator/(const Vec3_t<T> &a, const S &b)
{
    using R = decltype(a.x / b);

    return Vec3_t<R>{a.x / b, a.y / b, a.z / b};
}

template <typename T, typename U> DEVICE HOST INLINE auto make_Vec2(const T &x, const U &y)
{
    using R = decltype(x + y);

    return Vec2_t<R>{static_cast<R>(x), static_cast<R>(y)};
}

template <typename T, typename U, typename V> DEVICE HOST INLINE auto make_Vec3(const T &x, const U &y, const V &z)
{
    using R = decltype(x + y + z);

    return Vec3_t<R>{static_cast<R>(x), static_cast<R>(y), static_cast<R>(z)};
}

template <typename T, typename U> DEVICE HOST INLINE auto cross(const Vec2_t<T> &a, const Vec2_t<U> &b)
{
    return a.x * b.y - a.y * b.x;
}

template <typename T, typename U> DEVICE HOST INLINE auto cross(const Vec3_t<T> &a, const Vec3_t<U> &b)
{
    using R = decltype(a.y * b.z - a.z * b.y);

    return Vec3_t<R>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template <typename T> DEVICE HOST INLINE auto length_squared(const Vec2_t<T> &a)
{
    return a.x * a.x + a.y * a.y;
}

template <typename T> DEVICE HOST INLINE auto length_squared(const Vec3_t<T> &a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

template <typename T> DEVICE HOST INLINE auto length(const Vec2_t<T> &a)
{
    return real_sqrt(length_squared(a));
}

template <typename T> DEVICE HOST INLINE auto length(const Vec3_t<T> &a)
{
    return real_sqrt(length_squared(a));
}

template <typename T, typename A> DEVICE HOST INLINE auto rotate(const Vec2_t<T> &v, const A &angle)
{
    const auto c = real_cos(angle);
    const auto s = real_sin(angle);

    using R = decltype(c * v.x);

    return Vec2_t<R>{c * v.x - s * v.y, s * v.x + c * v.y};
}

template <typename T> DEVICE HOST INLINE T signum(const T &value)
{
    return static_cast<T>((value > static_cast<T>(0)) - (value < static_cast<T>(0)));
}

template <typename To, typename From> DEVICE HOST INLINE Vec2_t<To> vec_cast(const Vec2_t<From> &v)
{
    return {static_cast<To>(v.x), static_cast<To>(v.y)};
}

template <typename To, typename From> DEVICE HOST INLINE Vec3_t<To> vec_cast(const Vec3_t<From> &v)
{
    return {static_cast<To>(v.x), static_cast<To>(v.y), static_cast<To>(v.z)};
}

























// MARK: Quaternions


template <typename T> struct Quaternion {
    T real;
    Vec3_t<T> imagine;
};

template <typename T> DEVICE HOST INLINE Quaternion<T> operator+(const Quaternion<T> &a, const Quaternion<T> &b)
{
    return {a.real + b.real, a.imagine + b.imagine};
}

template <typename T, typename S> DEVICE HOST INLINE Quaternion<T> operator*(const Quaternion<T> &a, const S &b)
{
    const T scale = static_cast<T>(b);
    return {a.real * scale, a.imagine * scale};
}

template <typename S, typename T> DEVICE HOST INLINE Quaternion<T> operator*(const S &b, const Quaternion<T> &a)
{
    return a * b;
}

template <typename T> DEVICE HOST INLINE Quaternion<T> operator*(const Quaternion<T> &a, const Quaternion<T> &b)
{
    return {a.real * b.real - a.imagine * b.imagine,
            a.real * b.imagine + b.real * a.imagine + cross(a.imagine, b.imagine)};
}

template <typename T> DEVICE HOST INLINE auto norm(const Quaternion<T> &a)
{
    return real_sqrt(a.real * a.real + a.imagine * a.imagine);
}

template <typename T> DEVICE HOST INLINE auto norm_sq(const Quaternion<T> &a)
{
    return a.real * a.real + a.imagine * a.imagine;
}

template <typename T> DEVICE HOST INLINE Quaternion<T> conjugate(const Quaternion<T> &a)
{
    return {a.real, -a.imagine};
}

template <typename S, typename T> DEVICE HOST INLINE Quaternion<T> make_Quaternion(const S a, const Vec3_t<T> b)
{
    return {static_cast<T>(a), b};
}

template <typename T> DEVICE HOST INLINE Vec3_t<T> rotate(const Quaternion<T> &q, const Vec3_t<T> &v)
{
    return v + REAL_LITERAL(2.0) * q.real * cross(q.imagine, v) +
           REAL_LITERAL(2.0) * cross(q.imagine, cross(q.imagine, v));
}




























// MARK: States
template <int n>
struct State {
    accumulator_t down_disk_angle;
    StateVec2 pos;
    StateVec2 speed;
    Quaternion<accumulator_t> attitude;
    StateVec3 omega;
    // accumulator_t phase;
    StateVec3 q;
    #ifndef __CUDACC__
    __m256d yx[n];
    __m256d yy[n];
    __m256d yz[n];
    #else
    real_t yx[n*4];
    real_t yy[n*4];
    real_t yz[n*4];
    #endif
};

template <int n>
DEVICE HOST INLINE State<n> operator+(const State<n> &a, const State<n> &b)
{
    return {a.down_disk_angle + b.down_disk_angle,
            a.pos + b.pos,
            a.speed + b.speed,
            a.attitude + b.attitude,
            a.omega + b.omega,
            // a. + b.phase,
            {a.q.x + b.q.x, a.q.y + b.q.y, a.q.z + b.q.z}};
}

template <int n>
DEVICE HOST INLINE State<n> operator*(const real_t a, const State<n> &b)
{
    const accumulator_t scale = static_cast<accumulator_t>(a);
    return {scale * b.down_disk_angle, scale * b.pos, scale * b.speed,
            scale * b.attitude, scale * b.omega, scale * b.q};
}

template <int n>
DEVICE HOST INLINE State<n> operator*(const State<n> &b, const real_t a)
{
    return a * b;
}

struct SolverParams {
    real_t alpha;
    real_t delta;
    real_t gamma_g;
    real_t gamma;  // We should ensure that gamma = Eta * params.alpha * params.alpha * params.gamma_m!!!!!!!!!
    real_t gamma_m;
    // tau_star[j] = eddy_current_scale * tau_ref_star[j].
    // Reference: 6 mm ball radius and 100 rpm; runtime scale = somega * sr^2.
    real_t eddy_current_scale = REAL_LITERAL(1.0);
    // Optional following reverse COM force, in solver acceleration units.
    real_t reverse_push_acceleration = REAL_LITERAL(0.0);
};

// Physical radius = Rref * sr, disk rate = omega_ref * somega.
// At the reference magnet count: x* = k*x, k = (Nref/2)/L, t* = Omega*t.
// Preserve the existing model's magnet-count scaling convention:
// multiply gamma_g, gamma and gamma_m by (magnet_count/Nref)^2, while
// alpha and delta keep their reference-count normalization.
// Preconditions: magnet_count is positive and even; sr and somega are finite
// and positive, as is eta. Pass the existing Eta from hysteresis_config.hpp;
// its value is not duplicated in the physical reference configuration.
// Callers validate their runtime inputs before using this helper.
// Magnet count is explicit to avoid a dependency cycle with the field config.
DEVICE HOST INLINE SolverParams make_solver_params(const int magnet_count,
                                                   const real_t sr, const real_t somega,
                                                   const real_t eta)
{
    namespace physics = hysteresis_physics;
    const real_t radius = static_cast<real_t>(physics::Rref) * sr;
    const real_t omega = static_cast<real_t>(physics::omega_ref) * somega;
    const real_t k = static_cast<real_t>(physics::reference_magnet_count) /
                     (REAL_LITERAL(2.0) * static_cast<real_t>(physics::L));
    const real_t count_scale = static_cast<real_t>(magnet_count) /
                              static_cast<real_t>(physics::reference_magnet_count);
    const real_t count_scale2 = count_scale * count_scale;
    const real_t alpha = k * radius;
    const real_t omega2 = omega * omega;
    const real_t magnetization = static_cast<real_t>(physics::M_ref);
    const real_t gamma = REAL_LITERAL(3.0) * static_cast<real_t>(physics::mu0) *
                         magnetization * magnetization * k * k /
                         (static_cast<real_t>(physics::rho_b) * omega2) * count_scale2;
    const real_t delta = REAL_LITERAL(3.0) * static_cast<real_t>(physics::rho_air) *
                         static_cast<real_t>(physics::Cd) /
                         (REAL_LITERAL(8.0) * static_cast<real_t>(physics::rho_b) * radius * k);
    return {alpha, delta, k * static_cast<real_t>(physics::g) / omega2 * count_scale2, gamma,
            gamma / (eta * alpha * alpha),
            somega * sr * sr, REAL_LITERAL(0.0)};
}
