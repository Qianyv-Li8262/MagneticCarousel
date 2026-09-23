#include "2dmodel-hysteresis.hpp"

#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>

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

// Each call advances one trajectory chunk. Contact mode and direction are
// carried into the next chunk; the field cache remains local to this call.
State evolve_trajectory(const State &initial, real_t dt, int steps, const SolverParams &params,
                        bool &sliding, Vec2 &sliding_dir, bool initialize)
{
    validate_params(params);
    if (!real_isfinite(dt) || dt <= REAL_LITERAL(0.0) || steps < 0)
        throw std::invalid_argument("evolve: dt must be positive and steps non-negative");

    State state = initial;
    if (initialize)
        initialize_mode(state, params, sliding, sliding_dir);
    else if (!real_isfinite(sliding_dir.x) || !real_isfinite(sliding_dir.y))
        throw std::invalid_argument("evolve: non-finite sliding direction");

    bool field_cache_valid = false;
    FieldCache cache{};
    FieldResult field_cache{};
    const LUTNode *lut = trajectory_lut();
    for (int i = 0; i < steps; ++i) {
        const AdvanceStepResult result = fwd1step(dt, params, state, sliding, sliding_dir,
                                                  field_cache_valid, &cache, lut, &field_cache);
        if (result != AdvanceStepResult::kStepOk)
            throw std::runtime_error("evolve: step " + std::to_string(i) +
                                     " failed with status " + std::to_string(static_cast<int>(result)));
    }
    return state;
}
