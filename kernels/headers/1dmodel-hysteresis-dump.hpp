#pragma once

#ifdef __CUDACC__
#error "1dmodel-hysteresis-dump.hpp is a CPU-only tracing integrator"
#endif

#include "1dmodel-hysteresis.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hysteresis_dump_core
{

enum class relu_branch : std::int8_t { zero = 0, x = 1 };
enum class rk_role : std::int8_t { full_trial = 0, pre_event = 1, post_event = 2 };
enum class rk_stage : std::int8_t { start = 0, midpoint = 1 };
enum class step_event : std::int8_t {
    none = 0,
    sliding_to_sticking = 1,
    sliding_reverse = 2,
    sticking_to_sliding = 3,
};

struct trace_options {
    bool enabled = false;
    std::int64_t stride = 1000;
    std::int64_t overview_stride = 10000;
    std::int64_t start_step = 0;
    std::int64_t end_step = -1;
    bool log_branch_changes = true;
    bool log_rhs_stages = false;
    bool log_rejected_trials = false;
    std::int64_t max_records = 2000000;
};

struct step_record {
    std::int64_t step;
    real_t time;
    real_t integrator_t;
    real_t x;
    real_t xdot;
    real_t theta;
    real_t theta_dot;
    real_t phase;
    real_t qx;
    real_t qy;
    real_t qnorm;
    bool sliding;
    real_t slip_sign;
    std::int64_t rhs_count;
    std::int64_t zero_branch_count;
    std::int64_t x_branch_count;
    std::int64_t branch_switch_count;
    real_t min_relu_test;
    real_t max_relu_test;
    real_t closest_relu_test;
    std::int8_t last_branch;
    std::int64_t sliding_to_sticking_count;
    std::int64_t sliding_reverse_count;
    std::int64_t sticking_to_sliding_count;
};

struct overview_record {
    std::int64_t step;
    real_t time;
    real_t integrator_t;
    real_t x;
    real_t xdot;
    real_t theta;
    real_t theta_dot;
    real_t phase;
    real_t qx;
    real_t qy;
    real_t qnorm;
    bool sliding;
    real_t slip_sign;
};

struct branch_change_record {
    std::int64_t step;
    real_t step_time;
    real_t t_stage;
    real_t qx;
    real_t qy;
    real_t relu_test;
    std::int8_t previous_branch;
    std::int8_t new_branch;
    std::int8_t role;
    std::int8_t stage;
    bool sliding;
};

struct rhs_record {
    std::int64_t step;
    real_t step_time;
    real_t t_stage;
    real_t qx;
    real_t qy;
    real_t qdotx;
    real_t qdoty;
    real_t h0_dot_x;
    real_t h0_dot_y;
    real_t he_dot_test_x;
    real_t he_dot_test_y;
    real_t relu_test;
    std::int8_t branch;
    std::int8_t role;
    std::int8_t stage;
    bool sliding;
    bool accepted;
};

struct trace_result {
    bool enabled = false;
    bool truncated = false;
    std::vector<overview_record> overview;
    std::vector<step_record> steps;
    std::vector<branch_change_record> branch_changes;
    std::vector<rhs_record> rhs;
};

struct evolution_result {
    state final_state;
    bool sliding;
    trace_result trace;
};

struct rhs_stage_trace {
    real_t t_stage = REAL_LITERAL(0.0);
    real_t qx = REAL_LITERAL(0.0);
    real_t qy = REAL_LITERAL(0.0);
    real_t qdotx = REAL_LITERAL(0.0);
    real_t qdoty = REAL_LITERAL(0.0);
    real_t h0_dot_x = REAL_LITERAL(0.0);
    real_t h0_dot_y = REAL_LITERAL(0.0);
    real_t he_dot_test_x = REAL_LITERAL(0.0);
    real_t he_dot_test_y = REAL_LITERAL(0.0);
    real_t relu_test = REAL_LITERAL(0.0);
    bool x_branch = false;
    bool sliding = false;
};

struct traced_rk2_result {
    state result;
    int flag;
    std::array<rhs_stage_trace, 2> stages;
    int stage_count;
};

class trace_collector
{
  public:
    trace_collector(const trace_options &options, const real_t dt) : options_(options), dt_(dt)
    {
        result_.enabled = options.enabled;
    }

    bool enabled() const
    {
        return options_.enabled;
    }

    void record_initial(const state &ball, const bool sliding, const real_t slip_sign)
    {
        if (!options_.enabled)
            return;
        append_overview(make_overview_record(0, ball, sliding, slip_sign));
        if (active_state(0))
            append_step(make_step_record(0, ball, sliding, slip_sign));
    }

    void commit_rk(const std::int64_t step, const rk_role role, const traced_rk2_result &rk, const bool accepted)
    {
        if (!active_rhs(step))
            return;
        for (int i = 0; i < rk.stage_count; ++i) {
            const auto stage = static_cast<rk_stage>(i);
            if (accepted)
                observe_accepted(step, role, stage, rk.stages[i]);
            if (options_.log_rhs_stages && (accepted || options_.log_rejected_trials))
                append_rhs(step, role, stage, rk.stages[i], accepted);
        }
    }

    void note_event(const std::int64_t step, const step_event event)
    {
        if (!active_rhs(step))
            return;
        switch (event) {
        case step_event::sliding_to_sticking:
            ++sliding_to_sticking_count_;
            break;
        case step_event::sliding_reverse:
            ++sliding_reverse_count_;
            break;
        case step_event::sticking_to_sliding:
            ++sticking_to_sliding_count_;
            break;
        case step_event::none:
            break;
        }
    }

    void step_completed(const std::int64_t step, const state &ball, const bool sliding, const real_t slip_sign,
                        const bool final_step)
    {
        if (step % options_.overview_stride == 0 || final_step)
            append_overview(make_overview_record(step, ball, sliding, slip_sign));

        if (!active_state(step))
            return;
        const bool range_start = step == options_.start_step;
        const bool range_end = options_.end_step >= 0 && step == options_.end_step;
        const bool periodic = step >= options_.start_step && (step - options_.start_step) % options_.stride == 0;
        if (range_start || periodic || range_end || final_step) {
            append_step(make_step_record(step, ball, sliding, slip_sign));
            reset_block();
        }
    }

    trace_result take_result()
    {
        return std::move(result_);
    }

  private:
    trace_options options_;
    real_t dt_;
    trace_result result_;
    std::int64_t rhs_count_ = 0;
    std::int64_t zero_branch_count_ = 0;
    std::int64_t x_branch_count_ = 0;
    std::int64_t branch_switch_count_ = 0;
    std::int64_t sliding_to_sticking_count_ = 0;
    std::int64_t sliding_reverse_count_ = 0;
    std::int64_t sticking_to_sliding_count_ = 0;
    real_t min_relu_test_ = std::numeric_limits<real_t>::infinity();
    real_t max_relu_test_ = -std::numeric_limits<real_t>::infinity();
    real_t closest_relu_test_ = std::numeric_limits<real_t>::quiet_NaN();
    bool have_closest_ = false;
    bool have_last_branch_ = false;
    bool last_branch_ = false;

    bool active_state(const std::int64_t step) const
    {
        return options_.enabled && step >= options_.start_step && (options_.end_step < 0 || step <= options_.end_step);
    }

    bool active_rhs(const std::int64_t step) const
    {
        return options_.enabled && step > options_.start_step && (options_.end_step < 0 || step <= options_.end_step);
    }

    real_t grid_time(const std::int64_t step) const
    {
        return static_cast<real_t>(step) * dt_;
    }

    bool can_append(const std::size_t size)
    {
        if (size < static_cast<std::size_t>(options_.max_records))
            return true;
        result_.truncated = true;
        return false;
    }

    void append_step(const step_record &record)
    {
        if (can_append(result_.steps.size()))
            result_.steps.push_back(record);
    }

    void append_overview(const overview_record &record)
    {
        if (can_append(result_.overview.size()))
            result_.overview.push_back(record);
    }

    void append_branch_change(const branch_change_record &record)
    {
        if (can_append(result_.branch_changes.size()))
            result_.branch_changes.push_back(record);
    }

    void append_rhs(const std::int64_t step, const rk_role role, const rk_stage stage, const rhs_stage_trace &trace,
                    const bool accepted)
    {
        if (!can_append(result_.rhs.size()))
            return;
        result_.rhs.push_back(
            {step, grid_time(step), trace.t_stage, trace.qx, trace.qy, trace.qdotx, trace.qdoty, trace.h0_dot_x,
             trace.h0_dot_y, trace.he_dot_test_x, trace.he_dot_test_y, trace.relu_test,
             static_cast<std::int8_t>(trace.x_branch ? relu_branch::x : relu_branch::zero),
             static_cast<std::int8_t>(role), static_cast<std::int8_t>(stage), trace.sliding, accepted});
    }

    void observe_accepted(const std::int64_t step, const rk_role role, const rk_stage stage,
                          const rhs_stage_trace &trace)
    {
        ++rhs_count_;
        if (trace.x_branch)
            ++x_branch_count_;
        else
            ++zero_branch_count_;
        if (trace.relu_test < min_relu_test_)
            min_relu_test_ = trace.relu_test;
        if (trace.relu_test > max_relu_test_)
            max_relu_test_ = trace.relu_test;
        if (!have_closest_ || real_fabs(trace.relu_test) < real_fabs(closest_relu_test_)) {
            closest_relu_test_ = trace.relu_test;
            have_closest_ = true;
        }

        const bool changed = have_last_branch_ && trace.x_branch != last_branch_;
        if (changed)
            ++branch_switch_count_;
        if (options_.log_branch_changes && (!have_last_branch_ || changed)) {
            append_branch_change(
                {step, grid_time(step), trace.t_stage, trace.qx, trace.qy, trace.relu_test,
                 static_cast<std::int8_t>(have_last_branch_ ? (last_branch_ ? relu_branch::x : relu_branch::zero)
                                                            : static_cast<relu_branch>(-1)),
                 static_cast<std::int8_t>(trace.x_branch ? relu_branch::x : relu_branch::zero),
                 static_cast<std::int8_t>(role), static_cast<std::int8_t>(stage), trace.sliding});
        }
        last_branch_ = trace.x_branch;
        have_last_branch_ = true;
    }

    step_record make_step_record(const std::int64_t step, const state &ball, const bool sliding,
                                 const real_t slip_sign) const
    {
        const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
        return {step,
                grid_time(step),
                static_cast<real_t>(ball.t),
                static_cast<real_t>(ball.x),
                static_cast<real_t>(ball.xdot),
                static_cast<real_t>(ball.theta),
                static_cast<real_t>(ball.theta_dot),
                static_cast<real_t>(ball.phase),
                static_cast<real_t>(ball.q.x),
                static_cast<real_t>(ball.q.y),
                length(ball.q),
                sliding,
                slip_sign,
                rhs_count_,
                zero_branch_count_,
                x_branch_count_,
                branch_switch_count_,
                rhs_count_ ? min_relu_test_ : nan,
                rhs_count_ ? max_relu_test_ : nan,
                have_closest_ ? closest_relu_test_ : nan,
                static_cast<std::int8_t>(have_last_branch_ ? (last_branch_ ? relu_branch::x : relu_branch::zero)
                                                           : static_cast<relu_branch>(-1)),
                sliding_to_sticking_count_,
                sliding_reverse_count_,
                sticking_to_sliding_count_};
    }

    overview_record make_overview_record(const std::int64_t step, const state &ball, const bool sliding,
                                         const real_t slip_sign) const
    {
        return {step,
                grid_time(step),
                static_cast<real_t>(ball.t),
                static_cast<real_t>(ball.x),
                static_cast<real_t>(ball.xdot),
                static_cast<real_t>(ball.theta),
                static_cast<real_t>(ball.theta_dot),
                static_cast<real_t>(ball.phase),
                static_cast<real_t>(ball.q.x),
                static_cast<real_t>(ball.q.y),
                length(ball.q),
                sliding,
                slip_sign};
    }

    void reset_block()
    {
        rhs_count_ = 0;
        zero_branch_count_ = 0;
        x_branch_count_ = 0;
        branch_switch_count_ = 0;
        sliding_to_sticking_count_ = 0;
        sliding_reverse_count_ = 0;
        sticking_to_sliding_count_ = 0;
        min_relu_test_ = std::numeric_limits<real_t>::infinity();
        max_relu_test_ = -std::numeric_limits<real_t>::infinity();
        closest_relu_test_ = std::numeric_limits<real_t>::quiet_NaN();
        have_closest_ = false;
    }
};

// This is the original RHS with observation added after the exact ReLU test.
INLINE bool eval_rhs(const state &in, state &out, const solver_params &params, const bool sliding,
                     const real_t slip_sign, rhs_stage_trace &trace)
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
    if (N < 0)
        return false;
    const real_t M_magnet = -params.gamma_m * E * cross(in.q, h_ext);
    const real_t F_magnet = ETA * params.alpha * params.alpha * params.gamma_m *
                            (E * (in.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
    if (sliding) {
        const real_t F = -slip_sign * MU * N;
        out.xdot = F_magnet + F - params.delta * in.xdot * real_fabs(in.xdot);
        out.theta_dot = M_magnet + F / (ETA * params.alpha);
    } else {
        out.theta_dot =
            ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet - F_magnet / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
            params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * in.theta_dot * real_fabs(in.theta_dot);
        out.xdot = -params.alpha * out.theta_dot;
    }
    out.x = in.xdot;
    out.theta = in.theta_dot;
    out.t = REAL_LITERAL(1.0);
    out.phase = REAL_LITERAL(1.0) - in.xdot;
    const vec2 h_ext_dot = static_cast<real_t>(out.phase) * h_ext_phase_derivative -
                           static_cast<real_t>(out.theta) * make_vec2(-h_ext.y, h_ext.x);

    vec2 h0_dot = h_ext_dot * (REAL_LITERAL(3.0) / (MU_R + REAL_LITERAL(2.0)));
    real_t interme = ALPHA_JA * (REAL_LITERAL(1.0) - C * XI_JA) / XI_JA - E;
    vec2 q_dot = (REAL_LITERAL(1.0) / interme) * h0_dot;
    vec2 he_dot = (h0_dot + (E - ALPHA_JA * (REAL_LITERAL(1.0) - C)) * q_dot) / (REAL_LITERAL(1.0) - XI_JA);

    const real_t relu_test = in.q * he_dot;
    const bool x_branch = relu_test > REAL_LITERAL(0.0);
    if (x_branch) {
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

    trace = {static_cast<real_t>(in.t),
             static_cast<real_t>(in.q.x),
             static_cast<real_t>(in.q.y),
             q_dot.x,
             q_dot.y,
             h0_dot.x,
             h0_dot.y,
             he_dot.x,
             he_dot.y,
             relu_test,
             x_branch,
             sliding};
    return true;
}

INLINE traced_rk2_result rk2_traced(const state &ball, const real_t dt, const solver_params &params, const bool sliding,
                                    const real_t slip_sign)
{
    traced_rk2_result result{};
    state trial;
    if (!eval_rhs(ball, trial, params, sliding, slip_sign, result.stages[0])) {
        result.result = ball;
        result.flag = 1;
        result.stage_count = 0;
        return result;
    }
    result.stage_count = 1;
    state trial_mid = ball + (dt / 2) * trial;
    state trial2;
    if (!eval_rhs(trial_mid, trial2, params, sliding, slip_sign, result.stages[1])) {
        result.result = ball;
        result.flag = 1;
        return result;
    }
    result.stage_count = 2;
    result.result = ball + (dt)*trial2;
    result.flag = 0;
    return result;
}

INLINE int fwd1step(state &ball, const real_t dt, const solver_params &params, bool &sliding, real_t &slip_sign,
                    const std::int64_t completed_step, trace_collector &collector)
{
    ball.phase = real_remainder(ball.phase, REAL_LITERAL(6.28318530717958647692));
    ball.theta = real_remainder(ball.theta, REAL_LITERAL(6.28318530717958647692));
    real_t step_slip_sign = slip_sign;
    if (sliding && step_slip_sign == REAL_LITERAL(0.0)) {
        step_slip_sign = signum(ball.xdot + params.alpha * ball.theta_dot);
        if (step_slip_sign == REAL_LITERAL(0.0))
            return 2;
    }

    const auto trial = rk2_traced(ball, dt, params, sliding, step_slip_sign);
    if (trial.flag != 0)
        return trial.flag;
    const state &trial_final = trial.result;
    if (sliding) {
        real_t s1 = ball.xdot + params.alpha * ball.theta_dot;
        real_t s2 = trial_final.xdot + params.alpha * trial_final.theta_dot;
        const bool crossed =
            (s1 > REAL_LITERAL(0.0) && s2 <= REAL_LITERAL(0.0)) || (s1 < REAL_LITERAL(0.0) && s2 >= REAL_LITERAL(0.0));
        if (crossed) {
            const bool event_at_step_end = s2 == REAL_LITERAL(0.0);
            real_t dt_transit = dt * s1 / (s1 - s2);
            if (event_at_step_end) {
                collector.commit_rk(completed_step, rk_role::full_trial, trial, true);
                state state_dttrans = trial_final;
                project_pure_rolling(state_dttrans, params);
                const field_evaluation field_lab = evaluate_field(static_cast<real_t>(state_dttrans.phase));
                const real_t body_angle = -static_cast<real_t>(state_dttrans.theta);
                const vec2 h_ext = rotate(field_lab.field, body_angle);
                const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
                const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
                const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
                const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
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
                                         params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta *
                                             state_dttrans.theta_dot * real_fabs(state_dttrans.theta_dot);
                if (real_fabs(theta_dot - M_magnet) * ETA * params.alpha > MU * N) {
                    slip_sign = step_slip_sign * REAL_LITERAL(-1.0);
                    collector.note_event(completed_step, step_event::sliding_reverse);
                } else {
                    sliding = false;
                    slip_sign = REAL_LITERAL(0.0);
                    collector.note_event(completed_step, step_event::sliding_to_sticking);
                }
                ball = state_dttrans;
                if (!sliding)
                    project_pure_rolling(ball, params);
                return 0;
            }

            collector.commit_rk(completed_step, rk_role::full_trial, trial, false);
            const auto pre_event = rk2_traced(ball, dt_transit, params, sliding, step_slip_sign);
            if (pre_event.flag != 0)
                return pre_event.flag;
            collector.commit_rk(completed_step, rk_role::pre_event, pre_event, true);
            state state_dttrans = pre_event.result;
            project_pure_rolling(state_dttrans, params);
            const field_evaluation field_lab = evaluate_field(static_cast<real_t>(state_dttrans.phase));
            const real_t body_angle = -static_cast<real_t>(state_dttrans.theta);
            const vec2 h_ext = rotate(field_lab.field, body_angle);
            const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
            const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
            const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
            const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
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
                next_slip_sign *= REAL_LITERAL(-1.0);
                collector.note_event(completed_step, step_event::sliding_reverse);
            } else {
                next_sliding = false;
                next_slip_sign = REAL_LITERAL(0.0);
                collector.note_event(completed_step, step_event::sliding_to_sticking);
            }
            const auto post_event = rk2_traced(state_dttrans, dt - dt_transit, params, next_sliding, next_slip_sign);
            if (post_event.flag != 0)
                return post_event.flag;
            collector.commit_rk(completed_step, rk_role::post_event, post_event, true);
            ball = post_event.result;
            sliding = next_sliding;
            slip_sign = next_slip_sign;
            if (!sliding)
                project_pure_rolling(ball, params);
            return 0;
        }

        collector.commit_rk(completed_step, rk_role::full_trial, trial, true);
        ball = trial_final;
        slip_sign = step_slip_sign;
        return 0;
    }

    const field_evaluation field_lab = evaluate_field(static_cast<real_t>(trial_final.phase));
    const real_t body_angle = -static_cast<real_t>(trial_final.theta);
    const vec2 h_ext = rotate(field_lab.field, body_angle);
    const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
    const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
    const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
    const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
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
        collector.commit_rk(completed_step, rk_role::full_trial, trial, false);
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
        const auto pre_event = rk2_traced(ball, dt_transit, params, sliding, step_slip_sign);
        if (pre_event.flag != 0)
            return pre_event.flag;
        collector.commit_rk(completed_step, rk_role::pre_event, pre_event, true);
        const bool next_sliding = true;
        const real_t next_slip_sign = -signum(theta_dot - M_magnet);
        const auto post_event = rk2_traced(pre_event.result, dt - dt_transit, params, next_sliding, next_slip_sign);
        if (post_event.flag != 0)
            return post_event.flag;
        collector.commit_rk(completed_step, rk_role::post_event, post_event, true);
        collector.note_event(completed_step, step_event::sticking_to_sliding);
        ball = post_event.result;
        sliding = next_sliding;
        slip_sign = next_slip_sign;
        return 0;
    }

    collector.commit_rk(completed_step, rk_role::full_trial, trial, true);
    ball = trial_final;
    project_pure_rolling(ball, params);
    return 0;
}

INLINE evolution_result evolve(const state &initial, const real_t dt, const int steps, const solver_params &params,
                               const trace_options &options)
{
    if (dt <= REAL_LITERAL(0.0))
        throw std::invalid_argument("evolve: dt must be positive");
    if (steps < 0)
        throw std::invalid_argument("evolve: steps must be non-negative");
    if (options.enabled) {
        if (options.stride <= 0 || options.overview_stride <= 0)
            throw std::invalid_argument("dump_every and overview_every must be positive");
        if (options.start_step < 0)
            throw std::invalid_argument("start_step must be non-negative");
        if (options.end_step >= 0 && options.end_step < options.start_step)
            throw std::invalid_argument("end_step must be -1 or at least start_step");
        if (options.max_records <= 0)
            throw std::invalid_argument("max_records must be positive");
    }

    state ball = initial;
    const real_t slip_velocity = ball.xdot + params.alpha * ball.theta_dot;
    bool sliding = real_fabs(slip_velocity) > REAL_LITERAL(1.0e-10);
    real_t slip_sign = sliding ? signum(slip_velocity) : REAL_LITERAL(0.0);
    if (!sliding) {
        const field_evaluation field_lab = evaluate_field(static_cast<real_t>(ball.phase));
        const real_t body_angle = -static_cast<real_t>(ball.theta);
        const vec2 h_ext = rotate(field_lab.field, body_angle);
        const vec2 h_ext_phase_derivative = rotate(field_lab.phase_derivative, body_angle);
        const vec2 h_ext_height_derivative = rotate(field_lab.height_derivative, body_angle);
        const real_t A = (MU_R - REAL_LITERAL(1.0)) / (MU_R + REAL_LITERAL(2.0));
        const real_t E = (REAL_LITERAL(1.0) - C) / (MU_R + REAL_LITERAL(2.0)) / (REAL_LITERAL(1.0) - XI_JA);
        const real_t N = params.gamma_g - params.gamma * (A * (h_ext * h_ext_height_derivative) -
                                                          E * (ball.q * h_ext_height_derivative));
        if (N < REAL_LITERAL(0.0))
            throw std::runtime_error("evolve: N < 0 at the initial state");
        const real_t M_magnet = -params.gamma_m * E * cross(ball.q, h_ext);
        const real_t F_magnet = ETA * params.alpha * params.alpha * params.gamma_m *
                                (E * (ball.q * h_ext_phase_derivative) - A * (h_ext * h_ext_phase_derivative));
        const real_t theta_dot =
            ETA / (REAL_LITERAL(1.0) + ETA) * M_magnet - F_magnet / ((REAL_LITERAL(1.0) + ETA) * params.alpha) -
            params.alpha / (REAL_LITERAL(1.0) + ETA) * params.delta * ball.theta_dot * real_fabs(ball.theta_dot);
        if (real_fabs(theta_dot - M_magnet) * ETA * params.alpha > MU * N) {
            sliding = true;
            slip_sign = -signum(theta_dot - M_magnet);
        } else {
            slip_sign = REAL_LITERAL(0.0);
            project_pure_rolling(ball, params);
        }
    }

    trace_collector collector(options, dt);
    collector.record_initial(ball, sliding, slip_sign);
    for (int i = 0; i < steps; ++i) {
        const std::int64_t completed_step = static_cast<std::int64_t>(i) + 1;
        const int flag = fwd1step(ball, dt, params, sliding, slip_sign, completed_step, collector);
        if (flag != 0)
            throw std::runtime_error("evolve: fwd1step failed at step " + std::to_string(i) + " with flag " +
                                     std::to_string(flag));
        collector.step_completed(completed_step, ball, sliding, slip_sign, i + 1 == steps);
    }
    return {ball, sliding, collector.take_result()};
}

} // namespace hysteresis_dump_core
