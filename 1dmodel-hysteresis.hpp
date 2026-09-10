#pragma once

// macros for cuda / cpu c++ backends
#ifdef __CUDACC__
#define DEVICE __device__
#define GLOBAL __global__
#define HOST __host__
#define INLINE __forceinline__
#else
#define DEVICE
#define GLOBAL
#define HOST
#define INLINE inline
#endif

#ifndef __CUDACC__
#include <cmath>
using std::cosf;
using std::fabsf;
using std::sinf;
using std::sqrtf;
#endif

struct vec2 {
    float x;
    float y;
};

DEVICE HOST INLINE vec2 operator+(const vec2 &a, const vec2 &b)
{
    return {a.x + b.x, a.y + b.y};
}

DEVICE HOST INLINE vec2 operator-(const vec2 &a, const vec2 &b)
{
    return {a.x - b.x, a.y - b.y};
}

DEVICE HOST INLINE float operator*(const vec2 &a, const vec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

DEVICE HOST INLINE vec2 operator*(const vec2 &a, const float b)
{
    return {b * a.x, b * a.y};
}

DEVICE HOST INLINE vec2 operator*(const float b, const vec2 &a)
{
    return a * b;
}

DEVICE HOST INLINE vec2 make_vec2(const float a, const float b)
{
    return {a, b};
}

DEVICE HOST INLINE float cross(const vec2 &a, const vec2 &b)
{
    return a.x * b.y - a.y * b.x;
}

DEVICE HOST INLINE float signum(const float value)
{
    return static_cast<float>((value > 0.0f) - (value < 0.0f));
}

DEVICE HOST INLINE vec2 rotate(const vec2 vec, const float angle)
{
    // 逆时针旋转矢量angle弧度
    return {cosf(angle) * vec.x - sinf(angle) * vec.y, sinf(angle) * vec.x + cosf(angle) * vec.y};
}

DEVICE HOST INLINE float length(const vec2 &a)
{
    return sqrtf(a.x * a.x + a.y * a.y);
}

struct state {
    float t;
    float x;
    float xdot;
    float theta;
    float theta_dot;
    float phase;
    vec2 q;
};

DEVICE HOST INLINE state operator+(const state &a, const state &b)
{
    return {a.t + b.t,         a.x + b.x, a.xdot + b.xdot, a.theta + b.theta, a.theta_dot + b.theta_dot,
            a.phase + b.phase, a.q + b.q};
}

DEVICE HOST INLINE state operator*(const float a, const state &b)
{
    return {a * b.t, a * b.x, a * b.xdot, a * b.theta, a * b.theta_dot, a * b.phase, a * b.q};
}

#define SOLVER_PARAMS_DEFINED
// parameter macros
#ifdef SOLVER_PARAMS_DEFINED
#define GAMMA_G 2.5f
#define GAMMA 1.0f
#define MU_R 10000.0f
#define C 0.9f
#define GAMMA_M 1.0f
#define ETA 0.4f
#define DELTA 6.4e-5f
#define MU 0.02f
#define K_HYSTER 1.0f
#define ALPHA_JA 0.9f
// these should be runtime params. let ai change.
#define ALPHA 0.918f
#define H_EXT 1.0f
// these should be runtime params. let ai change.
#define SMALL_KHI_JA (1.0f-(1.0f/(1.0f+ALPHA_JA * (MU_R-1.0f))))
#endif
// parameter macros

DEVICE HOST INLINE bool evalRhs(const state &in, state &out, const bool sliding, float slip_sign, bool &transit)
{
    vec2 q_rotated = rotate(in.q, in.theta);
    vec2 h_0 = {H_EXT * cosf(in.t - in.x), H_EXT * sinf(in.t - in.x)};
    float N = GAMMA_G -
              GAMMA / (2.0f + MU_R) * (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - SMALL_KHI_JA) * (q_rotated * h_0));
    // the ball can fly
    if (N < 0)
        return false;
    if (sliding) {
        // contact point speed
        float S = in.xdot + ALPHA * in.theta_dot;
        if (slip_sign == 0.0f) {
            slip_sign = signum(S);
        }
        // friction
        float F = -slip_sign * MU * N;
        float M_magnet = -GAMMA_M * (1 - C) / ((1 - SMALL_KHI_JA) * (MU_R + 2)) * cross(q_rotated, h_0);
        float F_magnet = ETA * ALPHA * ALPHA * M_magnet;
        out.xdot = F_magnet + F - DELTA * in.xdot * fabsf(in.xdot);
        out.theta_dot = M_magnet + F / (ETA * ALPHA);
        // in the sliding scenario we do not have dt so we cannot check transition
        // when (sliding == true) the value of transit is UNDEFINED!
    } else {
        float M_magnet = -GAMMA_M * (1 - C) / ((1 - SMALL_KHI_JA) * (MU_R + 2)) * cross(q_rotated, h_0);
        float F_magnet = ETA * ALPHA * ALPHA * M_magnet;
        out.theta_dot =
            ETA / (1 + ETA) * (1 - ALPHA) * M_magnet - ALPHA / (1 + ETA) * DELTA * in.theta_dot * fabsf(in.theta_dot);
        // to reduce recomputation we can check transition in this step

        // if (fabsf(out.theta_dot - M_magnet) * ETA * ALPHA > MU * N)
        //     transit = 1;

        out.xdot = -ALPHA * out.theta_dot;
    }
    out.x = in.xdot;
    out.theta = in.theta_dot;
    out.t = 1.0f;
    out.phase = 1.0f - in.xdot - in.theta_dot;
    vec2 h_ext = {cosf(in.phase) * H_EXT, sinf(in.phase) * H_EXT};
    vec2 h_ext_dot = out.phase * make_vec2(-sinf(in.phase) * H_EXT, cosf(in.phase) * H_EXT);
    // assume ReLU use the 0 branch
    vec2 q_dot = SMALL_KHI_JA / ALPHA_JA / (1 - C * SMALL_KHI_JA) * h_ext_dot * (3.0f / (MU_R + 2.0f));
    vec2 he_dot =
        h_ext_dot * (3.0f / (MU_R + 2.0f)) + ALPHA_JA * (3.0f * (MU_R - 1.0f) / (MU_R + 2.0f) * h_ext_dot -
                                                         (1 - C) / (1 - SMALL_KHI_JA) * q_dot);
    // if reject we should select ReLU x branch
    if (in.q * he_dot > 0) {
        // reject
        float a = 1.0f - C * SMALL_KHI_JA;
        float b = -ALPHA_JA * (1.0f - C) / K_HYSTER * length(in.q);
        float c = SMALL_KHI_JA / ALPHA_JA;
        float d = -length(in.q) / K_HYSTER;
        float a_final = c / a;
        float b_final = d / a - b * (c + d) / a / (a + b);
        vec2 qnorm = in.q * (1 / length(in.q));
        q_dot = a_final * h_ext_dot * (3.0f / (MU_R + 2.0f)) +
                b_final * (qnorm * h_ext_dot * (3.0f / (MU_R + 2.0f))) * qnorm;
    }
    out.q = q_dot;
    return true;
}

struct rk2_res{
    state result;
    int flag;
};
//0 is ok, 1 is fly
DEVICE HOST INLINE rk2_res rk2(const state &ball, const float dt, bool sliding, float slip_sign)
{
    state trial;
    bool transit = false;
    if (!evalRhs(ball, trial, sliding, slip_sign, transit)) return {ball,1};
    state trial_mid = ball + (dt / 2) * trial;
    state trial2;
    if (!evalRhs(trial_mid, trial2, sliding, slip_sign, transit)) return {ball,1};
    return {ball + (dt)*trial2,0};
}

// use int for result. 0=normal, 1=N<0
DEVICE HOST INLINE int fwd1step(state &ball, const float dt, bool &sliding, float &slip_sign)
{
    // trial rk2 step
    const auto [trial_final, trial_error] = rk2(ball, dt, sliding, slip_sign);
    if (trial_error != 0)
        return trial_error;
    if (sliding) {
        float s1 = ball.xdot + ALPHA * ball.theta_dot;
        float s2 = trial_final.xdot + ALPHA * trial_final.theta_dot;
        if (s1 * s2 < 0) {
            float dt_transit = dt * s1 / (s1 - s2);
            const auto [state_dttrans, transit_error] = rk2(ball, dt_transit, sliding, slip_sign);
            if (transit_error != 0)
                return transit_error;
            // calc N and Freq at this transit point
            vec2 q_rotated = rotate(state_dttrans.q, state_dttrans.theta);
            vec2 h_0 = {H_EXT * cosf(state_dttrans.t - state_dttrans.x),
                        H_EXT * sinf(state_dttrans.t - state_dttrans.x)};
            float N = GAMMA_G - GAMMA / (2.0f + MU_R) *
                                    (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - SMALL_KHI_JA) * (q_rotated * h_0));
            float M_magnet = -GAMMA_M * (1 - C) / ((1 - SMALL_KHI_JA) * (MU_R + 2)) * cross(q_rotated, h_0);
            float theta_dot = ETA / (1 + ETA) * (1 - ALPHA) * M_magnet -
                              ALPHA / (1 + ETA) * DELTA * state_dttrans.theta_dot * fabsf(state_dttrans.theta_dot);
            bool next_sliding = sliding;
            float next_slip_sign = slip_sign;
            if (fabsf(theta_dot - M_magnet) * ETA * ALPHA > MU * N) {
                // keep sliding but slip_sign reverses
                next_slip_sign *= -1.0f;
            } else {
                // turn into sticking
                next_sliding = false;
            }
            // complete rk2 step after transition
            const auto [final_state, final_error] =
                rk2(state_dttrans, dt - dt_transit, next_sliding, next_slip_sign);
            if (final_error != 0)
                return final_error;
            ball = final_state;
            sliding = next_sliding;
            slip_sign = next_slip_sign;
            return 0;
        } else {
            ball = trial_final;
            return 0;
        }
    } else {
        // check if still stick at the end of the step
        // calc N and Freq at this point
        vec2 q_rotated = rotate(trial_final.q, trial_final.theta);
        vec2 h_0 = {H_EXT * cosf(trial_final.t - trial_final.x), H_EXT * sinf(trial_final.t - trial_final.x)};
        float N = GAMMA_G - GAMMA / (2.0f + MU_R) *
                                (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - SMALL_KHI_JA) * (q_rotated * h_0));
        float M_magnet = -GAMMA_M * (1 - C) / ((1 - SMALL_KHI_JA) * (MU_R + 2)) * cross(q_rotated, h_0);
        float theta_dot = ETA / (1 + ETA) * (1 - ALPHA) * M_magnet -
                          ALPHA / (1 + ETA) * DELTA * trial_final.theta_dot * fabsf(trial_final.theta_dot);
        float g_final = fabsf(theta_dot - M_magnet) * ETA * ALPHA - MU * N;
        if (g_final > 0) {
            // turn into sliding. lerp the transit time.
            // i think shading the outer vars is ok here because we do not use outer vars
            vec2 q_rotated_init = rotate(ball.q, ball.theta);
            vec2 h_0_init = {H_EXT * cosf(ball.t - ball.x), H_EXT * sinf(ball.t - ball.x)};
            float N_init = GAMMA_G - GAMMA / (2.0f + MU_R) *
                                    (-(MU_R - 1) * H_EXT * H_EXT + (1 - C) / (1 - SMALL_KHI_JA) * (q_rotated_init * h_0_init));
            float M_magnet_init = -GAMMA_M * (1 - C) / ((1 - SMALL_KHI_JA) * (MU_R + 2)) * cross(q_rotated_init, h_0_init);
            float theta_dot_init = ETA / (1 + ETA) * (1 - ALPHA) * M_magnet_init -
                               ALPHA / (1 + ETA) * DELTA * ball.theta_dot * fabsf(ball.theta_dot);
            float g_init = fabsf(theta_dot_init - M_magnet_init) * ETA * ALPHA - MU * N_init;
            float dt_transit = dt * g_init / (g_init - g_final);
            const auto [final1, transit_error] = rk2(ball, dt_transit, sliding, slip_sign);
            if (transit_error != 0)
                return transit_error;
            const bool next_sliding = true;
            const float next_slip_sign = -signum(theta_dot - M_magnet);
            const auto [final_state, final_error] =
                rk2(final1, dt - dt_transit, next_sliding, next_slip_sign);
            if (final_error != 0)
                return final_error;
            ball = final_state;
            sliding = next_sliding;
            slip_sign = next_slip_sign;
            return 0;
        } else {
            // keep sticking
            ball = trial_final;
            return 0;
        }
    }
}
