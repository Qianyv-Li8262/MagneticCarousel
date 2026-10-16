// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "hysteresis_config.hpp"

// Portable scalar state format, independent of AVX alignment and Kahan storage:
// legacy 15 values, then all yx modes, all yy modes, all yz modes.
template <bool IncludeEddy>
DEVICE HOST INLINE State<kEddyCurrentModeCount> read_state_values(const real_t *v)
{
    State<kEddyCurrentModeCount> state{v[0], {v[1], v[2]}, {v[3], v[4]},
        {v[5], {v[6], v[7], v[8]}}, {v[9], v[10], v[11]}, {v[12], v[13], v[14]}};
    if constexpr (IncludeEddy) {
#ifndef __CUDACC__
        __m256d *axes[3] = {state.yx, state.yy, state.yz};
        for (int axis = 0; axis < 3; ++axis) {
            for (int group = 0; group < kEddyCurrentModeCount; ++group) {
                alignas(32) double lane[4];
                for (int k = 0; k < 4; ++k)
                    lane[k] = static_cast<double>(v[15 + axis * 4 * kEddyCurrentModeCount + 4 * group + k]);
                axes[axis][group] = _mm256_load_pd(lane);
            }
        }
#else
        for (int j = 0; j < 4 * kEddyCurrentModeCount; ++j) {
            state.yx[j] = v[15 + j];
            state.yy[j] = v[15 + 4 * kEddyCurrentModeCount + j];
            state.yz[j] = v[15 + 8 * kEddyCurrentModeCount + j];
        }
#endif
    }
    return state;
}

template <bool IncludeEddy>
DEVICE HOST INLINE void write_state_values(const State<kEddyCurrentModeCount> &state, real_t *v)
{
    const accumulator_t legacy[15] = {state.down_disk_angle, state.pos.x, state.pos.y,
        state.speed.x, state.speed.y, state.attitude.real, state.attitude.imagine.x,
        state.attitude.imagine.y, state.attitude.imagine.z, state.omega.x, state.omega.y,
        state.omega.z, state.q.x, state.q.y, state.q.z};
    for (int j = 0; j < 15; ++j) v[j] = static_cast<real_t>(legacy[j]);
    if constexpr (IncludeEddy) {
#ifndef __CUDACC__
        const __m256d *axes[3] = {state.yx, state.yy, state.yz};
        for (int axis = 0; axis < 3; ++axis) {
            for (int group = 0; group < kEddyCurrentModeCount; ++group) {
                alignas(32) double lane[4];
                _mm256_store_pd(lane, axes[axis][group]);
                for (int k = 0; k < 4; ++k)
                    v[15 + axis * 4 * kEddyCurrentModeCount + 4 * group + k] = static_cast<real_t>(lane[k]);
            }
        }
#else
        for (int j = 0; j < 4 * kEddyCurrentModeCount; ++j) {
            v[15 + j] = state.yx[j];
            v[15 + 4 * kEddyCurrentModeCount + j] = state.yy[j];
            v[15 + 8 * kEddyCurrentModeCount + j] = state.yz[j];
        }
#endif
    }
}
