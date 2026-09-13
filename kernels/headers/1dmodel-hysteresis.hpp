#pragma once

#if defined(USE_FP64) && defined(USE_DD)
#error "USE_FP64 and USE_DD are mutually exclusive"
#endif

#if (defined(USE_FP64) || defined(USE_DD)) && defined(__CUDACC__)
#error "FP64 and double-double are only supported by the CPU backend; CUDA must use FP32"
#endif

// macros for cuda / cpu c++ backends
#ifdef __CUDACC__
#define DEVICE __device__
#define GLOBAL __global__
#define HOST __host__
#define INLINE __forceinline__
#define REAL_ISFINITE_IMPL isfinite
#else
#define DEVICE
#define GLOBAL
#define HOST
#define INLINE inline
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

#define SOLVER_PARAMS_DEFINED
// parameter macros
#ifdef SOLVER_PARAMS_DEFINED

// Fundamental magnetic constants
#define MU_R REAL_LITERAL(2.2926667e3)
#define ALPHA_JA REAL_LITERAL(3.2e-3)
#define C REAL_LITERAL(0.38)
#define K_HYSTER REAL_LITERAL(1.0909091e-3)
// Fundamental magnetic constants

#define XI_JA (REAL_LITERAL(1.0) - (REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + ALPHA_JA * (MU_R - REAL_LITERAL(1.0)))))

// Magetic field strength
#define H_EXT REAL_LITERAL(1.4661546e-1)
// Magetic field strength

// Mechanical params
#define ETA REAL_LITERAL(0.4)
#define MU REAL_LITERAL(0.02)
// Mechanical params

#endif
// parameter macros

struct solver_params {
    real_t alpha;
    real_t delta;
    real_t gamma_g;
    real_t gamma;
    real_t gamma_m;
};

// Remove roundoff drift from the sticking constraint while preserving the
// contact impulse balance used at a sliding-to-sticking event.
DEVICE HOST INLINE void project_pure_rolling(state &ball, const solver_params &params)
{
    const real_t slip = ball.xdot + params.alpha * ball.theta_dot;
    const real_t impulse = -ETA / (REAL_LITERAL(1.0) + ETA) * slip;
    ball.xdot += impulse;
    ball.theta_dot += impulse / (ETA * params.alpha);
}

DEVICE HOST INLINE bool evalRhs(const state &in, state &out, const solver_params &params, const bool sliding,
                                real_t slip_sign, bool &transit)
{
    // ============================================================================
    vec2 h_ext = {real_cos(in.phase) * H_EXT, real_sin(in.phase) * H_EXT};
    // ============================================================================
    real_t N = params.gamma_g - params.gamma / (REAL_LITERAL(2.0) + MU_R) *
                                    (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - XI_JA) * (in.q * h_ext));
    // the ball can fly
    // CHINESE CAN FLY TOO!!!
    if (N < 0)
        return false;
    if (sliding) {
        // friction
        real_t F = -slip_sign * MU * N;
        real_t M_magnet = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(in.q, h_ext);
        real_t F_magnet = ETA * params.alpha * params.alpha * M_magnet;
        out.xdot = F_magnet + F - params.delta * in.xdot * real_fabs(in.xdot);
        out.theta_dot = M_magnet + F / (ETA * params.alpha);
        // in the sliding scenario we do not have dt so we cannot check transition
        // when (sliding == true) the value of transit is UNDEFINED!
    } else {
        real_t M_magnet = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(in.q, h_ext);
        out.theta_dot = ETA / (1 + ETA) * (1 - params.alpha) * M_magnet -
                        params.alpha / (1 + ETA) * params.delta * in.theta_dot * real_fabs(in.theta_dot);
        // to reduce recomputation we can check transition in this step

        // if (real_fabs(out.theta_dot - M_magnet) * ETA * params.alpha > MU * N)
        //     transit = 1;

        out.xdot = -params.alpha * out.theta_dot;
    }
    out.x = in.xdot;
    out.theta = in.theta_dot;
    out.t = REAL_LITERAL(1.0);
    out.phase = REAL_LITERAL(1.0) - in.xdot - in.theta_dot;
    vec2 h_ext_dot = out.phase * make_vec2(-real_sin(in.phase) * H_EXT, real_cos(in.phase) * H_EXT);

    const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);

    vec2 h0_dot = h_ext_dot * (REAL_LITERAL(3.0) / (MU_R + REAL_LITERAL(2.0)));
    // assume ReLU use the 0 branch
    real_t interme = ALPHA_JA * (REAL_LITERAL(1.0) - C * XI_JA) / XI_JA - E;
    vec2 q_dot = (REAL_LITERAL(1.0) / interme) * h0_dot;
    // IMPORTANT: self-consistent H_e_dot
    vec2 he_dot = (h0_dot + (E - ALPHA_JA * (REAL_LITERAL(1.0) - C)) * q_dot) / (REAL_LITERAL(1.0) - XI_JA);

    // if reject we should select ReLU x branch
    if (in.q * he_dot > REAL_LITERAL(0.0)) {
        const real_t c = XI_JA / ALPHA_JA;
        const real_t d = -length(in.q) / K_HYSTER;
        const real_t a = REAL_LITERAL(1.0) - C * XI_JA - E * c;
        const real_t b = -ALPHA_JA * (REAL_LITERAL(1.0) - C) / K_HYSTER * length(in.q) - E * d;
        const real_t a_final = c / a;
        const real_t b_final = d / a - b * (c + d) / a / (a + b);
        const vec2 qnorm = in.q * (REAL_LITERAL(1.0) / length(in.q));
        q_dot = a_final * h0_dot + b_final * (qnorm * h0_dot) * qnorm;
    }
    out.q.x = q_dot.x;
    out.q.y = q_dot.y;
    return true;
}

struct rk2_res {
    state result;
    int flag;
};
// 0 is ok, 1 is fly
DEVICE HOST INLINE rk2_res rk2(const state &ball, const real_t dt, const solver_params &params, bool sliding,
                               real_t slip_sign)
{
    state trial;
    bool transit = false;
    if (!evalRhs(ball, trial, params, sliding, slip_sign, transit))
        return {ball, 1};
    state trial_mid = ball + (dt / 2) * trial;
    state trial2;
    if (!evalRhs(trial_mid, trial2, params, sliding, slip_sign, transit))
        return {ball, 1};
    return {ball + (dt)*trial2, 0};
}

// use int for result. 0=normal, 1=N<0, 2=undefined sliding direction
DEVICE HOST INLINE int fwd1step(state &ball, const real_t dt, const solver_params &params, bool &sliding,
                                real_t &slip_sign)
{
    ball.phase = real_remainder(ball.phase, real_two_pi());
    ball.theta = real_remainder(ball.theta, real_two_pi());
    real_t step_slip_sign = slip_sign;
    if (sliding && step_slip_sign == REAL_LITERAL(0.0)) {
        step_slip_sign = signum(ball.xdot + params.alpha * ball.theta_dot);
        if (step_slip_sign == REAL_LITERAL(0.0))
            return 2;
    }
    // trial rk2 step
    const auto [trial_final, trial_error] = rk2(ball, dt, params, sliding, step_slip_sign);
    if (trial_error != 0)
        return trial_error;
    if (sliding) {
        real_t s1 = ball.xdot + params.alpha * ball.theta_dot;
        real_t s2 = trial_final.xdot + params.alpha * trial_final.theta_dot;
        const bool crossed =
            (s1 > REAL_LITERAL(0.0) && s2 <= REAL_LITERAL(0.0)) || (s1 < REAL_LITERAL(0.0) && s2 >= REAL_LITERAL(0.0));
        if (crossed) {
            const bool event_at_step_end = s2 == REAL_LITERAL(0.0);
            real_t dt_transit = dt * s1 / (s1 - s2);
            auto [state_dttrans, transit_error] =
                event_at_step_end ? rk2_res{trial_final, 0} : rk2(ball, dt_transit, params, sliding, step_slip_sign);
            if (transit_error != 0)
                return transit_error;
            // project the event state onto the pure-rolling constraint
            project_pure_rolling(state_dttrans, params);
            // calc N and Freq at this transit point
            vec2 h_ext = {H_EXT * real_cos(state_dttrans.phase), H_EXT * real_sin(state_dttrans.phase)};
            real_t N =
                params.gamma_g - params.gamma / (REAL_LITERAL(2.0) + MU_R) *
                                     (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - XI_JA) * (state_dttrans.q * h_ext));
            if (N < REAL_LITERAL(0.0))
                return 1;
            real_t M_magnet = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(state_dttrans.q, h_ext);
            real_t theta_dot =
                ETA / (1 + ETA) * (1 - params.alpha) * M_magnet -
                params.alpha / (1 + ETA) * params.delta * state_dttrans.theta_dot * real_fabs(state_dttrans.theta_dot);
            bool next_sliding = sliding;
            real_t next_slip_sign = step_slip_sign;
            if (real_fabs(theta_dot - M_magnet) * ETA * params.alpha > MU * N) {
                // keep sliding but slip_sign reverses
                next_slip_sign *= REAL_LITERAL(-1.0);
            } else {
                // turn into sticking
                next_sliding = false;
                next_slip_sign = REAL_LITERAL(0.0);
            }
            // complete rk2 step after transition
            const auto [final_state, final_error] =
                event_at_step_end ? rk2_res{state_dttrans, 0}
                                  : rk2(state_dttrans, dt - dt_transit, params, next_sliding, next_slip_sign);
            if (final_error != 0)
                return final_error;
            ball = final_state;
            sliding = next_sliding;
            slip_sign = next_slip_sign;
            if (!sliding)
                project_pure_rolling(ball, params);
            return 0;
        } else {
            ball = trial_final;
            slip_sign = step_slip_sign;
            return 0;
        }
    } else {
        // check if still stick at the end of the step
        // calc N and Freq at this point
        vec2 h_ext = {H_EXT * real_cos(trial_final.phase), H_EXT * real_sin(trial_final.phase)};
        real_t N = params.gamma_g - params.gamma / (REAL_LITERAL(2.0) + MU_R) *
                                        (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - XI_JA) * (trial_final.q * h_ext));
        if (N < REAL_LITERAL(0.0))
            return 1;
        real_t M_magnet = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(trial_final.q, h_ext);
        real_t theta_dot = ETA / (1 + ETA) * (1 - params.alpha) * M_magnet - params.alpha / (1 + ETA) * params.delta *
                                                                                 trial_final.theta_dot *
                                                                                 real_fabs(trial_final.theta_dot);
        real_t g_final = real_fabs(theta_dot - M_magnet) * ETA * params.alpha - MU * N;
        if (g_final > 0) {
            // turn into sliding. lerp the transit time.
            // i think shading the outer vars is ok here because we do not use outer vars
            vec2 h_ext_init = {H_EXT * real_cos(ball.phase), H_EXT * real_sin(ball.phase)};
            real_t N_init =
                params.gamma_g - params.gamma / (REAL_LITERAL(2.0) + MU_R) *
                                     (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - XI_JA) * (ball.q * h_ext_init));
            real_t M_magnet_init = -params.gamma_m * (1 - C) / ((1 - XI_JA) * (MU_R + 2)) * cross(ball.q, h_ext_init);
            real_t theta_dot_init =
                ETA / (1 + ETA) * (1 - params.alpha) * M_magnet_init -
                params.alpha / (1 + ETA) * params.delta * ball.theta_dot * real_fabs(ball.theta_dot);
            real_t g_init = real_fabs(theta_dot_init - M_magnet_init) * ETA * params.alpha - MU * N_init;
            real_t dt_transit = dt * g_init / (g_init - g_final);
            const auto [final1, transit_error] = rk2(ball, dt_transit, params, sliding, step_slip_sign);
            if (transit_error != 0)
                return transit_error;
            const bool next_sliding = true;
            const real_t next_slip_sign = -signum(theta_dot - M_magnet);
            const auto [final_state, final_error] = rk2(final1, dt - dt_transit, params, next_sliding, next_slip_sign);
            if (final_error != 0)
                return final_error;
            ball = final_state;
            sliding = next_sliding;
            slip_sign = next_slip_sign;
            return 0;
        } else {
            // keep sticking
            ball = trial_final;
            project_pure_rolling(ball, params);
            return 0;
        }
    }
}

#ifndef __CUDACC__
state evolve(const state &initial, const real_t dt, const int steps, const solver_params &params);
state evolve_with_mode(const state &initial, const real_t dt, const int steps, const solver_params &params,
                       bool &sliding);
#endif
