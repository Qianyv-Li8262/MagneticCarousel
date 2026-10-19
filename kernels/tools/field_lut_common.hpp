// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once
#include "hysteresis_config_field.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace field_lut
{
using mp_real = boost::multiprecision::cpp_dec_float_100;
using clock_type = std::chrono::steady_clock;

double elapsed_seconds(const clock_type::time_point start, const clock_type::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

struct radial_sample
{
    mp_real A;
    mp_real dA_drho;
    mp_real Z;
    mp_real dZ_drho;
};

template <typename Real>
struct SplineNode
{
    Real A;
    Real M_A;
    Real Z;
    Real M_Z;
};

std::vector<mp_real> solve_clamped_second_derivatives(const std::vector<mp_real> &x,
                                                       const std::vector<mp_real> &y,
                                                       const mp_real &left_derivative,
                                                       const mp_real &right_derivative)
{
    const std::size_t count = x.size();
    if (count < 2 || y.size() != count)
        throw std::runtime_error("a clamped cubic spline requires at least two matching nodes");

    std::vector<mp_real> lower(count), diagonal(count), upper(count), rhs(count);
    const mp_real first_width = x[1] - x[0];
    diagonal[0] = 2 * first_width;
    upper[0] = first_width;
    rhs[0] = 6 * ((y[1] - y[0]) / first_width - left_derivative);

    for (std::size_t i = 1; i + 1 < count; ++i) {
        const mp_real left_width = x[i] - x[i - 1];
        const mp_real right_width = x[i + 1] - x[i];
        lower[i] = left_width;
        diagonal[i] = 2 * (left_width + right_width);
        upper[i] = right_width;
        rhs[i] = 6 * ((y[i + 1] - y[i]) / right_width - (y[i] - y[i - 1]) / left_width);
    }

    const mp_real last_width = x[count - 1] - x[count - 2];
    lower[count - 1] = last_width;
    diagonal[count - 1] = 2 * last_width;
    rhs[count - 1] = 6 * (right_derivative - (y[count - 1] - y[count - 2]) / last_width);

    for (std::size_t i = 1; i < count; ++i) {
        const mp_real multiplier = lower[i] / diagonal[i - 1];
        diagonal[i] -= multiplier * upper[i - 1];
        rhs[i] -= multiplier * rhs[i - 1];
    }

    std::vector<mp_real> result(count);
    result[count - 1] = rhs[count - 1] / diagonal[count - 1];
    for (std::size_t i = count - 1; i-- > 0;)
        result[i] = (rhs[i] - upper[i] * result[i + 1]) / diagonal[i];
    return result;
}

bool native_is_big_endian()
{
    const unsigned short marker = 0x0102;
    unsigned char bytes[sizeof(marker)];
    std::memcpy(bytes, &marker, sizeof(marker));
    return bytes[0] == 0x01;
}

template <typename Output>
void write_scalar(std::FILE *file, const Output value, const bool big_endian)
{
    static_assert(std::is_same<Output, float>::value || std::is_same<Output, double>::value,
                  "only IEEE float and double LUTs are supported");
    if (!std::isfinite(value))
        throw std::runtime_error("a LUT value did not fit in the requested output type");

    std::array<unsigned char, sizeof(Output)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(Output));
    if (native_is_big_endian() != big_endian)
        std::reverse(bytes.begin(), bytes.end());
    if (std::fwrite(bytes.data(), sizeof(Output), 1, file) != 1)
        throw std::runtime_error("failed while writing a LUT value");
}

template <typename Output>
void write_lut(const std::string &path, const std::vector<SplineNode<mp_real>> &nodes, const bool big_endian)
{
    std::FILE *file = nullptr;
#ifdef _WIN32
    if (fopen_s(&file, path.c_str(), "wb") != 0)
        file = nullptr;
#else
    file = std::fopen(path.c_str(), "wb");
#endif
    if (!file)
        throw std::runtime_error("could not open output file: " + path);

    try {
        // Headerless AoS layout: mode -> radial node -> [A, M_A, Z, M_Z].
        for (const SplineNode<mp_real> &node : nodes) {
            const SplineNode<Output> rounded = {
                node.A.template convert_to<Output>(),
                node.M_A.template convert_to<Output>(),
                node.Z.template convert_to<Output>(),
                node.M_Z.template convert_to<Output>(),
            };
            write_scalar(file, rounded.A, big_endian);
            write_scalar(file, rounded.M_A, big_endian);
            write_scalar(file, rounded.Z, big_endian);
            write_scalar(file, rounded.M_Z, big_endian);
        }
        const int close_result = std::fclose(file);
        file = nullptr;
        if (close_result != 0)
            throw std::runtime_error("failed to close output file: " + path);
    } catch (...) {
        if (file)
            std::fclose(file);
        throw;
    }
}
template <typename Factory, typename Prepare>
int generate_lut(int argc, char **argv, Factory factory, Prepare prepare,
                 const char *sample_stage = "sample")
{
    try {
        const auto run_start = clock_type::now();
        if (argc != 3) {
            std::cerr << "usage: generate_field_lut <fp32-output.bin> <fp64-output.bin>\n";
            return 2;
        }
        if (magnet_number <= 0 || magnet_number % 2 != 0)
            throw std::runtime_error("magnet_number must be a positive even number");
        if (kFieldFourierModeCount <= 0)
            throw std::runtime_error("kFieldFourierModeCount must be positive");
        if (kSplineNodes < 2)
            throw std::runtime_error("kSplineNodes must be at least two");

        const mp_real z(IYPT16_STRINGIZE(IYPT16_FIELD_HEIGHT));
        const mp_real lower_ratio(IYPT16_STRINGIZE(IYPT16_LUT_LOWER_BOUND_RATIO));
        const mp_real upper_ratio(IYPT16_STRINGIZE(IYPT16_LUT_UPPER_BOUND_RATIO));
        const mp_real R = mp_real(magnet_number) / 2;
        const mp_real lower = lower_ratio * R;
        const mp_real upper = upper_ratio * R;
        if (z <= 0 || lower <= 0 || upper <= lower)
            throw std::runtime_error("invalid field height or LUT radial bounds");

        std::vector<mp_real> rho(kSplineNodes);
        const mp_real spacing = (upper - lower) / (kSplineNodes - 1);
        for (int node = 0; node < kSplineNodes; ++node)
            rho[node] = lower + spacing * node;

        std::vector<SplineNode<mp_real>> nodes;
        nodes.reserve(static_cast<std::size_t>(kFieldFourierModeCount) * kSplineNodes);
        const auto setup_end = clock_type::now();
        double prepare_total = 0, sample_total = 0, solve_a_total = 0, solve_z_total = 0, pack_total = 0;
        std::cout << std::fixed << std::setprecision(4);
        // Model-wide work must finish before timing individual Fourier modes.
        prepare(upper);
        const auto shared_end = clock_type::now();
        const double shared = elapsed_seconds(setup_end, shared_end);
        for (int mode = 0; mode < kFieldFourierModeCount; ++mode) {
            const auto mode_start = clock_type::now();
            const int harmonic = (2 * mode + 1) * magnet_number / 2;
            const auto evaluate = factory(harmonic, upper);
            std::vector<radial_sample> samples(kSplineNodes);
            std::vector<mp_real> A(kSplineNodes), Z(kSplineNodes);
            std::exception_ptr first_error;
            const auto sample_start = clock_type::now();
#pragma omp parallel for schedule(dynamic, 16)
            for (int node = 0; node < kSplineNodes; ++node) {
                try {
                    samples[node] = evaluate(rho[node], z, R, magnet_number);
                    A[node] = samples[node].A;
                    Z[node] = samples[node].Z;
                } catch (...) {
#pragma omp critical(field_lut_error)
                    {
                        if (!first_error)
                            first_error = std::current_exception();
                    }
                }
            }
            if (first_error)
                std::rethrow_exception(first_error);
            const auto sample_end = clock_type::now();

            const std::vector<mp_real> M_A = solve_clamped_second_derivatives(
                rho, A, samples.front().dA_drho, samples.back().dA_drho);
            const auto solve_a_end = clock_type::now();
            const std::vector<mp_real> M_Z = solve_clamped_second_derivatives(
                rho, Z, samples.front().dZ_drho, samples.back().dZ_drho);
            const auto solve_z_end = clock_type::now();
            for (int node = 0; node < kSplineNodes; ++node)
                nodes.push_back({A[node], M_A[node], Z[node], M_Z[node]});
            const auto pack_end = clock_type::now();
            const double prepare = elapsed_seconds(mode_start, sample_start);
            const double sample = elapsed_seconds(sample_start, sample_end);
            const double solve_a = elapsed_seconds(sample_end, solve_a_end);
            const double solve_z = elapsed_seconds(solve_a_end, solve_z_end);
            const double pack = elapsed_seconds(solve_z_end, pack_end);
            prepare_total += prepare;
            sample_total += sample;
            solve_a_total += solve_a;
            solve_z_total += solve_z;
            pack_total += pack;
            std::cout << "mode " << mode << " (harmonic " << harmonic << "): prepare=" << prepare
                      << " s, " << sample_stage << "=" << sample << " s, solve_A=" << solve_a
                      << " s, solve_Z=" << solve_z << " s, pack=" << pack << " s\n" << std::flush;
        }

        const auto write_start = clock_type::now();
        write_lut<float>(argv[1], nodes, kUseBigEndian);
        const auto write_fp32_end = clock_type::now();
        write_lut<double>(argv[2], nodes, kUseBigEndian);
        const auto write_fp64_end = clock_type::now();
        const double setup = elapsed_seconds(run_start, setup_end);
        const double write_fp32 = elapsed_seconds(write_start, write_fp32_end);
        const double write_fp64 = elapsed_seconds(write_fp32_end, write_fp64_end);
        const double wall = elapsed_seconds(run_start, write_fp64_end);
        const double accounted = setup + shared + prepare_total + sample_total + solve_a_total
                                 + solve_z_total + pack_total + write_fp32 + write_fp64;
        std::cout << "stage totals: setup=" << setup << " s, shared=" << shared
                  << " s, mode_prepare=" << prepare_total << " s, " << sample_stage << "=" << sample_total
                  << " s, solve_A=" << solve_a_total << " s, solve_Z=" << solve_z_total
                  << " s, pack=" << pack_total
                  << " s, write_fp32=" << write_fp32
                  << " s, write_fp64=" << write_fp64
                  << " s, other=" << std::max(0.0, wall - accounted)
                  << " s, wall=" << wall << " s\n";
        std::cout << "generated " << nodes.size() << " AoS nodes per file (A, M_A, Z, M_Z)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "field LUT generation failed: " << error.what() << '\n';
        return 1;
    }
}

} // namespace field_lut
