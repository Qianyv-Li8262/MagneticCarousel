// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include "2dmodel-hysteresis.hpp"

#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include <limits>
#include <sstream>
#include <iomanip>
#include <array>

namespace {

const LUTNode *trajectory_lut();

const char *step_error_description(AdvanceStepResult result)
{
    switch (result) {
    case AdvanceStepResult::kStepOk: return "kStepOk: step completed";
    case AdvanceStepResult::kStepBallFly: return "kStepBallFly: normal contact load became negative; ball lost contact";
    case AdvanceStepResult::kStepFinerRetry: return "kStepFinerRetry: contact transition requires a smaller time step";
    case AdvanceStepResult::kStepNumericalUnstable: return "kStepNumericalUnstable: numerical consistency check failed";
    case AdvanceStepResult::kStepNaNOrInf: return "kStepNaNOrInf: a computed state or force is not finite";
    case AdvanceStepResult::kStepSolverSelfContradict: return "kStepSolverSelfContradict: contact-event solution is inconsistent";
    case AdvanceStepResult::kStepLUTOutOfDomain: return "kStepLUTOutOfDomain: ball position or an integration trial position is outside the magnetic-field LUT radius range";
    case AdvanceStepResult::kStepRefinementRecursionLimit: return "kStepRefinementRecursionLimit: time-step refinement reached its depth or floating-point limit";
    case AdvanceStepResult::kRetrySlidingHalfTcZero: return "kRetrySlidingHalfTcZero: sliding contact-event half-time rounded to zero";
    case AdvanceStepResult::kRetrySlidingContactCannotStick: return "kRetrySlidingContactCannotStick: sliding contact event cannot satisfy sticking";
    case AdvanceStepResult::kRetrySlidingRollingMidRelease: return "kRetrySlidingRollingMidRelease: rolling midpoint loses contact consistency after sliding";
    case AdvanceStepResult::kRetrySlidingRollingFinalRelease: return "kRetrySlidingRollingFinalRelease: rolling final state releases after sliding";
    case AdvanceStepResult::kRetrySlidingMidZero: return "kRetrySlidingMidZero: sliding midpoint slip is zero";
    case AdvanceStepResult::kRetrySlidingMidPredictsContact: return "kRetrySlidingMidPredictsContact: sliding midpoint predicts a contact transition";
    case AdvanceStepResult::kRetrySlidingFinalZero: return "kRetrySlidingFinalZero: sliding final slip is zero";
    case AdvanceStepResult::kRetryRollingThresholdRecross: return "kRetryRollingThresholdRecross: rolling state recrosses the friction threshold";
    case AdvanceStepResult::kRetryRollingEventInvalid: return "kRetryRollingEventInvalid: rolling release event is invalid";
    case AdvanceStepResult::kRetryRollingReleaseMidZero: return "kRetryRollingReleaseMidZero: midpoint slip after rolling release is zero";
    case AdvanceStepResult::kRetryRollingReleaseMidContact: return "kRetryRollingReleaseMidContact: midpoint after rolling release predicts sticking";
    case AdvanceStepResult::kRetryRollingReleaseFinalZero: return "kRetryRollingReleaseFinalZero: final slip after rolling release is zero";
    }
    return "unknown AdvanceStepResult";
}

void validate_params(const SolverParams &p)
{
    if (!real_isfinite(p.alpha) || p.alpha <= REAL_LITERAL(0.0) ||
        !real_isfinite(p.delta) || p.delta < REAL_LITERAL(0.0) ||
        !real_isfinite(p.gamma_g) || !real_isfinite(p.gamma) || !real_isfinite(p.gamma_m) ||
        !real_isfinite(p.eddy_current_scale) || p.eddy_current_scale <= REAL_LITERAL(0.0) ||
        !real_isfinite(p.reverse_push_acceleration) || p.reverse_push_acceleration < REAL_LITERAL(0.0))
        throw std::invalid_argument("evolve: invalid solver parameters");
}

void initialize_mode(State<kEddyCurrentModeCount> &state, const SolverParams &params, bool &sliding, Vec2 &sliding_dir)
{
    const Vec2 slip = vec_cast<real_t>(state.speed) +
                      params.alpha * make_Vec2(-static_cast<real_t>(state.omega.y),
                                                static_cast<real_t>(state.omega.x));
    const real_t magnitude = length(slip);
    if (!real_isfinite(magnitude))
        throw std::invalid_argument("evolve: non-finite initial slip");
    sliding = magnitude > REAL_LITERAL(1.0e-10);
    sliding_dir = sliding ? slip / magnitude : make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    if (!sliding) {
        project_pure_rolling(state, params);
        FieldCache cache{};
        if (resolve_zero_slip_mode(state, params, sliding, sliding_dir, &cache, trajectory_lut()) !=
            AdvanceStepResult::kStepOk)
            throw std::invalid_argument("evolve: invalid initial contact state");
    }
}

const LUTNode *trajectory_lut()
{
#if FIELD_EVAL_MODE == 1
#ifndef IYPT16_FP64_LUT_PATH
#error "LUT mode requires IYPT16_FP64_LUT_PATH from the build"
#endif
    static const std::vector<LUTNode> nodes = [] {
        constexpr std::size_t count = static_cast<std::size_t>(kFieldFourierModeCount) * kSplineNodes;
        std::vector<LUTNode> result(count);
        std::ifstream file(IYPT16_FP64_LUT_PATH, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(LUTNode)))
            throw std::runtime_error("evolve: field LUT is missing or has the wrong size");
        file.seekg(0);
        file.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(count * sizeof(LUTNode)));
        if (!file)
            throw std::runtime_error("evolve: could not read field LUT");
        return result;
    }();
    return nodes.data();
#else
    return nullptr;
#endif
}

} // namespace

// Mechanical power per unit mass, in the solver's dimensionless 2D units.
// M in eval_rhs is angular acceleration: physical torque / mass = Eta*alpha^2*M.
std::array<real_t, 3> magnetic_mechanical_power(const State<kEddyCurrentModeCount> &state, const SolverParams &params)
{
    validate_params(params);
    FieldCache cache{};
    const Vec3 q_world = vec_cast<real_t>(rotate(state.attitude, state.q));
    const auto field = eval_mag_field_and_force(params, state.pos, q_world,
                                               state.down_disk_angle, &cache, trajectory_lut());
    constexpr real_t E = (REAL_LITERAL(1.0) - C_JA) /
                        (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    const Vec3 angular_acceleration = -params.gamma_m * E * cross(q_world, field.h);
    const real_t translation = field.Force.x * state.speed.x + field.Force.y * state.speed.y;
    const real_t rotation = Eta * params.alpha * params.alpha * (angular_acceleration * state.omega);
    return {translation, rotation, translation + rotation};
}

// Each call advances one trajectory chunk. Contact mode and direction are
// carried into the next chunk; the field cache remains local to this call.
State<kEddyCurrentModeCount> evolve_trajectory(const State<kEddyCurrentModeCount> &initial, real_t dt, int steps, const SolverParams &params,
                        bool &sliding, Vec2 &sliding_dir, bool initialize, std::int64_t start_step)
{
    validate_params(params);
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || steps < 0)
        throw std::invalid_argument("evolve: dt must be positive and steps non-negative");
    if (start_step < 0 || start_step > std::numeric_limits<std::int64_t>::max() - steps)
        throw std::invalid_argument("evolve: invalid start_step");

    State<kEddyCurrentModeCount> state = initial;
    if (initialize)
        initialize_mode(state, params, sliding, sliding_dir);
    else if (!real_isfinite(sliding_dir.x) || !real_isfinite(sliding_dir.y))
        throw std::invalid_argument("evolve: non-finite sliding direction");

    bool field_cache_valid = false;
    FieldCache cache{};
    FieldResult field_cache{};
    RetryDiagnostics retry_diagnostics{};
    const LUTNode *lut = trajectory_lut();
    for (int i = 0; i < steps; ++i) {
        const AdvanceStepResult result = fwd1step(dt, params, state, sliding, sliding_dir,
                                                  field_cache_valid, &cache, lut, &field_cache,
                                                  &retry_diagnostics);
        if (result != AdvanceStepResult::kStepOk) {
            // A failure need not have used refinement; its default retry state
            // is all zeros and must not be presented as the actual ball state.
            const bool has_retry_state = retry_diagnostics.last_depth >= 0;
            const State<kEddyCurrentModeCount> &diagnostic_state = has_retry_state ? retry_diagnostics.last_state : state;
            std::ostringstream message;
            message << std::setprecision(17)
                    << "evolve: global step " << start_step + i << " (chunk step " << i << ")"
                    << " failed with status " << static_cast<int>(result)
                    << " (" << step_error_description(result) << ")"
                    << "; step_start_pos=(" << state.pos.x << "," << state.pos.y << ")"
                    << "; step_start_radius=" << length(vec_cast<real_t>(state.pos));
            if (result == AdvanceStepResult::kStepLUTOutOfDomain) {
                constexpr real_t ring_radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
                message << "; valid_lut_radius=[" << kLUTLowerBoundRatio * ring_radius
                        << "," << kLUTUpperBoundRatio * ring_radius << "] (dimensionless)";
            }
            message << "; diagnostic_state_source=" << (has_retry_state ? "last_refinement_retry" : "step_start")
                    << "; last_retry_id=" << retry_diagnostics.last_reason + 1
                    << "; depth=" << retry_diagnostics.last_depth
                    << "; substep_dt=" << retry_diagnostics.last_dt
                    << "; retry_counts=";
            for (int r = 0; r < 12; ++r)
                message << (r == 0 ? "" : ",") << r + 1 << ":" << retry_diagnostics.counts[r];
            message << "; last_state_pos=(" << diagnostic_state.pos.x << ","
                    << diagnostic_state.pos.y << ")"
                    << "; last_state_speed=(" << diagnostic_state.speed.x << ","
                    << diagnostic_state.speed.y << ")"
                    << "; last_state_omega=(" << diagnostic_state.omega.x << ","
                    << diagnostic_state.omega.y << ","
                    << diagnostic_state.omega.z << ")"
                    << "; release=(g_start=" << retry_diagnostics.release_g_start
                    << ",g_mid=" << retry_diagnostics.release_g_mid
                    << ",g_final=" << retry_diagnostics.release_g_final
                    << ",A_event=" << retry_diagnostics.release_A_event
                    << ",b_event=" << retry_diagnostics.release_b_event
                    << ",tr=" << retry_diagnostics.release_tr << ")";
            throw std::runtime_error(message.str());
        }
    }
    return state;
}

// Hold the ball fixed while the magnets turn: evolve JA memory only.
State<kEddyCurrentModeCount> magnetize_held_ball(const State<kEddyCurrentModeCount> &initial, real_t dt, int steps, const SolverParams &params)
{
    validate_params(params);
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || steps < 0)
        throw std::invalid_argument("magnetize: dt must be positive and steps non-negative");
    if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        if (!state_in_lut_domain(initial))
            throw std::invalid_argument("magnetize: position outside the field LUT");
    }
    State<kEddyCurrentModeCount> state = initial;
    state.speed = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    state.omega = make_Vec3(REAL_LITERAL(0.0), REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    const real_t start_angle = static_cast<real_t>(state.down_disk_angle);
    const Vec2 zero = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    FieldCache cache{};
    for (int i = 0; i < steps; ++i) {
        state.down_disk_angle = start_angle + static_cast<real_t>(i) * dt;
        State<kEddyCurrentModeCount> first{}, second{};
        Vec3 hb_dot_first, hb_dot_second;
        if (!eval_rhs<false>(state, first, hb_dot_first, params, false, zero, &cache, trajectory_lut()))
            throw std::runtime_error("magnetize: ball lost contact");
        State<kEddyCurrentModeCount> mid = state;
        mid.q = state.q + first.q * (dt * REAL_LITERAL(0.5));
        mid.down_disk_angle = start_angle + (static_cast<real_t>(i) + REAL_LITERAL(0.5)) * dt;
        if (!eval_rhs<false>(mid, second, hb_dot_second, params, false, zero, &cache, trajectory_lut()))
            throw std::runtime_error("magnetize: ball lost contact");
        state.q = state.q + second.q * dt;
        if (!real_isfinite(length(state.q)))
            throw std::runtime_error("magnetize: non-finite JA state");
    }
    state.down_disk_angle = start_angle + static_cast<real_t>(steps) * dt;
    return state;
}

// The model header defines non-inline evolution functions. Keep the scan in
// this translation unit so the extension has exactly one copy of them.
#include "scan_initial_cpu.cpp"
