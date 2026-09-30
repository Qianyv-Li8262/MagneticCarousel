// SPDX-License-Identifier: AGPL-3.0-only
#include "scan_initial_common.hpp"

extern "C" __launch_bounds__(128, 3) GLOBAL void scan_initial_cuda(
    const real_t *v0_values, const real_t *s0_values, int k, real_t dt, int steps,
    int window_steps, int sample_stride, real_t sr, real_t somega,
    real_t reference_alpha, real_t reference_delta, real_t reference_gamma_g,
    real_t reference_gamma, real_t reference_gamma_m, const LUTNode *lut_global,
    real_t *output_state, int *output_status, real_t *output_metrics)
{
#if FIELD_EVAL_MODE == 1
    constexpr int lut_count = kFieldFourierModeCount * kSplineNodes;
    __shared__ LUTNode lut_shared[lut_count];
    const int thread_in_block = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int threads_in_block = static_cast<int>(blockDim.x * blockDim.y);
    for (int i = thread_in_block; i < lut_count; i += threads_in_block)
        lut_shared[i] = lut_global[i];
    __syncthreads();
    const LUTNode *field_lut = lut_shared;
#else
    const LUTNode *field_lut = nullptr;
#endif
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= k || y >= k) return;
    const int index = y * k + x;
    output_status[index] = -2;
    if (!real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) ||
        somega <= REAL_LITERAL(0.0) || !real_isfinite(dt) || dt <= REAL_LITERAL(0.0) ||
        steps <= 0 || window_steps <= 1 || sample_stride <= 0) return;
    const real_t inv_sr = REAL_LITERAL(1.0) / sr;
    const real_t inv_somega2 = REAL_LITERAL(1.0) / (somega * somega);
    const real_t scale = static_cast<real_t>(magnet_number) / REAL_LITERAL(8.0);
    const real_t scale2 = scale * scale;
    const SolverParams params = {reference_alpha * sr, reference_delta * inv_sr,
        reference_gamma_g * inv_somega2 * scale2,
        reference_gamma * inv_somega2 * scale2,
        reference_gamma_m * inv_sr * inv_sr * inv_somega2 * scale2};
    output_status[index] = initial_scan::analyze(v0_values[x], s0_values[y], dt,
        steps, window_steps, sample_stride, params, field_lut,
        output_state + 15 * index, output_metrics + initial_scan::metric_count * index);
}
