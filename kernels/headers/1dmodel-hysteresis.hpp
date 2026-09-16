#pragma once

#include "hysteresis_config.hpp"
#include "hysteresis_types.hpp"

#ifdef __CUDACC__
#include "../generated/field_coefficients_nvrtc.generated.cuh"
#else
#include "../generated/field_coefficients_cpu.generated.hpp"
#endif

// Evaluate the dimensionless field and its analytic derivatives with respect
// to the phase argument chi and the dimensionless height z / lambda.  All
// three quantities share one Fourier pass so each harmonic needs only one
// sine/cosine evaluation.
struct field_evaluation {
    vec2 field;
    vec2 phase_derivative;
    vec2 height_derivative;
};

DEVICE INLINE field_evaluation evaluate_field(const real_t chi)
{
#if FIELD_LEGACY_FIRST_HARMONIC_COMPATIBILITY
    static_assert(FIELD_FOURIER_MODE_COUNT == 1, "legacy first-harmonic compatibility requires exactly one mode");
#ifdef __CUDACC__
    float sine;
    float cosine;
    sincosf(chi, &sine, &cosine);
#else
    const real_t sine = real_sin(chi);
    const real_t cosine = real_cos(chi);
#endif
    // The current state stores the laboratory phase chi=t-x.  Rotating this
    // field by -theta below gives the legacy relative phase t-x-theta.
    return {
        {H_EXT * cosine, H_EXT * sine},
        {-H_EXT * sine, H_EXT * cosine},
        {-H_EXT * cosine, -H_EXT * sine},
    };
#else
    field_evaluation result = {
        {REAL_LITERAL(0.0), REAL_LITERAL(0.0)},
        {REAL_LITERAL(0.0), REAL_LITERAL(0.0)},
        {REAL_LITERAL(0.0), REAL_LITERAL(0.0)},
    };
    for (int mode = 0; mode < FIELD_FOURIER_MODE_COUNT; ++mode) {
        const int harmonic = 2 * mode + 1;
        const real_t n = static_cast<real_t>(harmonic);
        const real_t angle = n * chi;
#ifdef __CUDACC__
        float sine;
        float cosine;
        sincosf(angle, &sine, &cosine);
#else
        const real_t sine = real_sin(angle);
        const real_t cosine = real_cos(angle);
#endif
        const field_mode_coefficient coefficient = FIELD_COEFFICIENTS[mode];
        result.field.x -= coefficient.a * sine;
        result.field.y += coefficient.c * cosine;
        result.phase_derivative.x -= n * coefficient.a * cosine;
        result.phase_derivative.y -= n * coefficient.c * sine;
        result.height_derivative.x += real_two_pi() * n * coefficient.c * sine;
        result.height_derivative.y += coefficient.dz * cosine;
    }
    result.field = H_EXT * result.field;
    result.phase_derivative = H_EXT * result.phase_derivative;
    result.height_derivative = H_EXT * result.height_derivative;
    return result;
#endif
}

// Remove roundoff drift from the sticking constraint while preserving the
// contact impulse balance used at a sliding-to-sticking event.
DEVICE HOST INLINE void project_pure_rolling(state &ball, const solver_params &params)
{
    const real_t slip = ball.xdot + params.alpha * ball.theta_dot;
    const real_t impulse = -ETA / (REAL_LITERAL(1.0) + ETA) * slip;
    ball.xdot += impulse;
    ball.theta_dot += impulse / (ETA * params.alpha);
}

DEVICE INLINE bool evalRhs(const state &in, state &out, const solver_params &params, const bool sliding,
                           real_t slip_sign, bool &transit)
{
    const field_evaluation field_lab = evaluate_field(static_cast<real_t>(in.phase));
    const real_t body_angle = -static_cast<real_t>(in.theta);
    const vec2 h_ext = rotate(field_lab.field, body_angle);
    const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
    const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
    const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
    const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
    const real_t N =
        params.gamma_g - params.gamma * (A * (h_ext * h_ext_height_derivative) - E * (in.q * h_ext_height_derivative));
    // the ball can fly
    // CHINESE CAN FLY TOO!!!
    if (N < REAL_LITERAL(0.0))
        return false;

    const real_t M_magnet = -params.gamma_m * E * cross(in.q, h_ext);
    const real_t F_magnet = ETA * params.alpha * params.alpha * params.gamma_m *
                            (E * (in.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
    if (sliding) {
        // friction
        const real_t F = -slip_sign * MU * N;
        out.xdot = F_magnet + F - params.delta * in.xdot * real_fabs(in.xdot);
        out.theta_dot = M_magnet + F / (ETA * params.alpha);
        // in the sliding scenario we do not have dt so we cannot check transition
        // when (sliding == true) the value of transit is UNDEFINED!
    } else {
        out.theta_dot =
            ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet - F_magnet / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
            params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * in.theta_dot * real_fabs(in.theta_dot);
        // to reduce recomputation we can check transition in this step

        // if (real_fabs(out.theta_dot - M_magnet) * ETA * params.alpha > MU * N)
        //     transit = 1;

        out.xdot = -params.alpha * out.theta_dot;
    }
    out.x = in.xdot;
    out.theta = in.theta_dot;
    out.t = REAL_LITERAL(1.0);
    out.phase = REAL_LITERAL(1.0) - in.xdot;
    const vec2 h_ext_dot = static_cast<real_t>(out.phase) * h_ext_phase_derivative -
                           static_cast<real_t>(out.theta) * make_vec2(-h_ext.y, h_ext.x);

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
DEVICE INLINE rk2_res rk2(const state &ball, const real_t dt, const solver_params &params, bool sliding,
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
#ifdef __CUDACC__
DEVICE NOINLINE
#else
inline
#endif
    int fwd1step(state &ball, const real_t dt, const solver_params &params, bool &sliding, real_t &slip_sign)
{
    const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
    const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
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
            const field_evaluation field_lab = evaluate_field(static_cast<real_t>(state_dttrans.phase));
            const real_t body_angle = -static_cast<real_t>(state_dttrans.theta);
            const vec2 h_ext = rotate(field_lab.field, body_angle);
            const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
            const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
            const real_t N = params.gamma_g - params.gamma * (A * (h_ext * h_ext_height_derivative) -
                                                              E * (state_dttrans.q * h_ext_height_derivative));
            if (N < REAL_LITERAL(0.0))
                return 1;
            const real_t M_magnet = -params.gamma_m * E * cross(state_dttrans.q, h_ext);
            const real_t F_magnet =
                ETA * params.alpha * params.alpha * params.gamma_m *
                (E * (state_dttrans.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
            const real_t theta_dot = ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet -
                                     F_magnet / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
                                     params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * state_dttrans.theta_dot *
                                         real_fabs(state_dttrans.theta_dot);
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
        const field_evaluation field_lab = evaluate_field(static_cast<real_t>(trial_final.phase));
        const real_t body_angle = -static_cast<real_t>(trial_final.theta);
        const vec2 h_ext = rotate(field_lab.field, body_angle);
        const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
        const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
        const real_t N = params.gamma_g - params.gamma * (A * (h_ext * h_ext_height_derivative) -
                                                          E * (trial_final.q * h_ext_height_derivative));
        if (N < REAL_LITERAL(0.0))
            return 1;
        const real_t M_magnet = -params.gamma_m * E * cross(trial_final.q, h_ext);
        const real_t F_magnet = ETA * params.alpha * params.alpha * params.gamma_m *
                                (E * (trial_final.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
        const real_t theta_dot = ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet -
                                 F_magnet / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
                                 params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * trial_final.theta_dot *
                                     real_fabs(trial_final.theta_dot);
        const real_t g_final = real_fabs(theta_dot - M_magnet) * ETA * params.alpha - MU * N;
        if (g_final > 0) {
            // turn into sliding. lerp the transit time.
            // i think shading the outer vars is ok here because we do not use outer vars
            const field_evaluation field_lab_init = evaluate_field(static_cast<real_t>(ball.phase));
            const real_t body_angle_init = -static_cast<real_t>(ball.theta);
            const vec2 h_ext_init = rotate(field_lab_init.field, body_angle_init);
            const vec2 h_ext_phase_derivative_init = rotate(field_lab_init.phase_derivative, body_angle_init);
            const vec2 h_ext_height_derivative_init = rotate(field_lab_init.height_derivative, body_angle_init);
            const real_t N_init = params.gamma_g - params.gamma * (A * (h_ext_init * h_ext_height_derivative_init) -
                                                                   E * (ball.q * h_ext_height_derivative_init));
            const real_t M_magnet_init = -params.gamma_m * E * cross(ball.q, h_ext_init);
            const real_t F_magnet_init =
                ETA * params.alpha * params.alpha * params.gamma_m *
                (E * (ball.q * h_ext_phase_derivative_init) - A * (h_ext_init * h_ext_phase_derivative_init));
            const real_t theta_dot_init =
                ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet_init -
                F_magnet_init / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
                params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * ball.theta_dot * real_fabs(ball.theta_dot);
            const real_t g_init = real_fabs(theta_dot_init - M_magnet_init) * ETA * params.alpha - MU * N_init;
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
// Continue a chunked trajectory without reconstructing its hybrid mode.
// Set initialize_mode=true only for the first chunk; subsequent calls must
// pass back both sliding and slip_sign returned by the previous call.
state evolve_with_discrete_mode(const state &initial, const real_t dt, const int steps, const solver_params &params,
                                bool &sliding, real_t &slip_sign, bool initialize_mode);
#endif
