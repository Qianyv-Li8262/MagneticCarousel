// SPDX-License-Identifier: AGPL-3.0-only
#include "scan_initial_common.hpp"
#include "scan_initial_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
const LUTNode *scan_lut()
{
#if FIELD_EVAL_MODE == 1
#ifndef IYPT16_FP64_LUT_PATH
#error "LUT mode requires IYPT16_FP64_LUT_PATH from CMake"
#endif
    static const std::vector<LUTNode> nodes = [] {
        std::vector<LUTNode> result(kFieldFourierModeCount * kSplineNodes);
        std::ifstream file(IYPT16_FP64_LUT_PATH, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(result.size() * sizeof(LUTNode)))
            throw std::runtime_error("initial scan: field LUT missing or wrong size");
        file.seekg(0);
        file.read(reinterpret_cast<char *>(result.data()),
                  static_cast<std::streamsize>(result.size() * sizeof(LUTNode)));
        if (!file) throw std::runtime_error("initial scan: could not read field LUT");
        return result;
    }();
    return nodes.data();
#else
    return nullptr;
#endif
}
}

void scan_initial_grid_cpu(const real_t *v0_values, const real_t *s0_values, std::size_t k,
                           real_t dt, int steps, int window_steps, int sample_stride,
                           real_t sr, real_t somega,
                           int num_threads, std::int32_t *status, real_t *metrics,
                           real_t *states)
{
    if (k == 0 || k > 46340 || num_threads < 0 || steps < 1 || window_steps < 2 ||
        window_steps > steps || sample_stride < 1 || !real_isfinite(dt) || dt <= REAL_LITERAL(0.0) ||
        !real_isfinite(sr) || sr <= REAL_LITERAL(0.0) || !real_isfinite(somega) ||
        somega <= REAL_LITERAL(0.0))
        throw std::invalid_argument("initial scan: invalid grid, time, scale, or thread settings");
    const SolverParams params = make_solver_params(magnet_number, sr, somega, Eta);
    const LUTNode *field_lut = scan_lut();
    const std::size_t cells = k * k;
    const unsigned available = std::max(1u, std::thread::hardware_concurrency());
    const int workers = std::min<int>(num_threads == 0 ? available : num_threads,
                                      static_cast<int>(cells));
    std::atomic<std::size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= cells) return;
            const std::size_t x = index % k, y = index / k;
            status[index] = initial_scan::analyze(v0_values[x], s0_values[y], dt,
                steps, window_steps, sample_stride, params, field_lut,
                states + 15 * index, metrics + initial_scan::metric_count * index);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int i = 0; i < workers; ++i) pool.emplace_back(worker);
    for (auto &thread : pool) thread.join();
}
