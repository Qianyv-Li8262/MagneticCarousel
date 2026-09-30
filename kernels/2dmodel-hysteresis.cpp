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

void validate_params(const SolverParams &p)
{
    if (!real_isfinite(p.alpha) || p.alpha <= REAL_LITERAL(0.0) ||
        !real_isfinite(p.delta) || p.delta < REAL_LITERAL(0.0) ||
        !real_isfinite(p.gamma_g) || !real_isfinite(p.gamma) || !real_isfinite(p.gamma_m))
        throw std::invalid_argument("evolve: invalid solver parameters");
}

void initialize_mode(State &state, const SolverParams &params, bool &sliding, Vec2 &sliding_dir)
{
    const Vec2 slip = vec_cast<real_t>(state.speed) +
                      params.alpha * make_Vec2(-static_cast<real_t>(state.omega.y),
                                                static_cast<real_t>(state.omega.x));
    const real_t magnitude = length(slip);
    if (!real_isfinite(magnitude))
        throw std::invalid_argument("evolve: non-finite initial slip");
    sliding = magnitude > REAL_LITERAL(1.0e-10);
    sliding_dir = sliding ? slip / magnitude : make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
    if (!sliding)
        project_pure_rolling(state, params);
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
std::array<real_t, 3> magnetic_mechanical_power(const State &state, const SolverParams &params)
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
State evolve_trajectory(const State &initial, real_t dt, int steps, const SolverParams &params,
                        bool &sliding, Vec2 &sliding_dir, bool initialize, std::int64_t start_step)
{
    validate_params(params);
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || steps < 0)
        throw std::invalid_argument("evolve: dt must be positive and steps non-negative");
    if (start_step < 0 || start_step > std::numeric_limits<std::int64_t>::max() - steps)
        throw std::invalid_argument("evolve: invalid start_step");

    State state = initial;
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
            std::ostringstream message;
            message << std::setprecision(17)
                    << "evolve: global step " << start_step + i << " (chunk step " << i << ")"
                    << " failed with status " << static_cast<int>(result)
                    << "; last_retry_id=" << retry_diagnostics.last_reason + 1
                    << "; depth=" << retry_diagnostics.last_depth
                    << "; substep_dt=" << retry_diagnostics.last_dt
                    << "; retry_counts=";
            for (int r = 0; r < 12; ++r)
                message << (r == 0 ? "" : ",") << r + 1 << ":" << retry_diagnostics.counts[r];
            message << "; last_state_pos=(" << retry_diagnostics.last_state.pos.x << ","
                    << retry_diagnostics.last_state.pos.y << ")"
                    << "; last_state_speed=(" << retry_diagnostics.last_state.speed.x << ","
                    << retry_diagnostics.last_state.speed.y << ")"
                    << "; last_state_omega=(" << retry_diagnostics.last_state.omega.x << ","
                    << retry_diagnostics.last_state.omega.y << ","
                    << retry_diagnostics.last_state.omega.z << ")"
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

// The model header defines non-inline evolution functions. Keep the scan in
// this translation unit so the extension has exactly one copy of them.
#include "scan_initial_cpu.cpp"
