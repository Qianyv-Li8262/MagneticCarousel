#include "hysteresis_config_field.hpp"

#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using mp_real = boost::multiprecision::cpp_dec_float_100;

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

struct radial_function
{
    mp_real f;
    mp_real df;
    mp_real d2f;
};

mp_real hypergeometric_2f1(const mp_real &a, const mp_real &b, const mp_real &c, const mp_real &x)
{
    return boost::math::hypergeometric_pFq({a, b}, {c}, x);
}

radial_function evaluate_radial_function(const int harmonic, const mp_real &t)
{
    const mp_real m = harmonic;
    const mp_real three_halves("1.5");
    const mp_real p = m + three_halves;
    const mp_real a = p;
    const mp_real b = three_halves;
    const mp_real c = m + 1;
    const mp_real x = t * t;

    mp_real pochhammer_over_factorial = 1;
    for (int j = 0; j < harmonic; ++j)
        pochhammer_over_factorial *= (three_halves + j) / (j + 1);

    const mp_real F0 = hypergeometric_2f1(a, b, c, x);
    const mp_real F1 = hypergeometric_2f1(a + 1, b + 1, c + 1, x);
    const mp_real F2 = hypergeometric_2f1(a + 2, b + 2, c + 2, x);
    const mp_real ab_over_c = a * b / c;
    const mp_real shifted_ab_over_c = (a + 1) * (b + 1) / (c + 1);

    const mp_real t_to_p = pow(t, p);
    const mp_real f = pochhammer_over_factorial * t_to_p * F0;
    const mp_real df = pochhammer_over_factorial *
                       (p * pow(t, p - 1) * F0 +
                        2 * ab_over_c * pow(t, p + 1) * F1);
    const mp_real d2f = pochhammer_over_factorial *
                        (p * (p - 1) * pow(t, p - 2) * F0 +
                         2 * ab_over_c * (2 * p + 1) * t_to_p * F1 +
                         4 * ab_over_c * shifted_ab_over_c * pow(t, p + 2) * F2);
    return {f, df, d2f};
}

radial_sample evaluate_mode(const int harmonic, const mp_real &rho, const mp_real &z, const mp_real &R,
                            const int magnets)
{
    const mp_real chi = (z * z + rho * rho + R * R) / (2 * rho * R);
    const mp_real discriminant = sqrt(chi * chi - 1);
    const mp_real t = 1 / (chi + discriminant);

    const mp_real dchi_drho = (rho * rho - z * z - R * R) / (2 * rho * rho * R);
    const mp_real dchi_dz = z / (rho * R);
    const mp_real d2chi_drho_dz = -z / (rho * rho * R);
    const mp_real dt_drho = -t * dchi_drho / discriminant;
    const mp_real dt_dz = -t * dchi_dz / discriminant;
    const mp_real d2t_drho_dz = dchi_drho * dchi_dz /
                                    (discriminant * discriminant * discriminant) -
                                t * d2chi_drho_dz / discriminant;

    const radial_function radial = evaluate_radial_function(harmonic, t);
    const mp_real rho_to_three_halves = rho * sqrt(rho);
    const mp_real R_to_three_halves = R * sqrt(R);
    const mp_real scale = 2 * magnets / (R_to_three_halves * rho_to_three_halves);

    const mp_real A = scale * z * radial.f;
    const mp_real dA_drho = scale * z *
                            (radial.df * dt_drho - 3 * radial.f / (2 * rho));
    const mp_real Z = scale * (radial.f + z * radial.df * dt_dz);
    const mp_real dZ_drho = scale *
                            (radial.df * dt_drho - 3 * radial.f / (2 * rho) +
                             z * (radial.d2f * dt_drho * dt_dz + radial.df * d2t_drho_dz -
                                  3 * radial.df * dt_dz / (2 * rho)));
    return {A, dA_drho, Z, dZ_drho};
}

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
        if (std::fclose(file) != 0)
            throw std::runtime_error("failed to close output file: " + path);
    } catch (...) {
        std::fclose(file);
        throw;
    }
}
} // namespace

int main(int argc, char **argv)
{
    try {
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
        for (int mode = 0; mode < kFieldFourierModeCount; ++mode) {
            const int harmonic = (2 * mode + 1) * magnet_number / 2;
            std::vector<radial_sample> samples(kSplineNodes);
            std::vector<mp_real> A(kSplineNodes), Z(kSplineNodes);
            for (int node = 0; node < kSplineNodes; ++node) {
                samples[node] = evaluate_mode(harmonic, rho[node], z, R, magnet_number);
                A[node] = samples[node].A;
                Z[node] = samples[node].Z;
            }

            const std::vector<mp_real> M_A = solve_clamped_second_derivatives(
                rho, A, samples.front().dA_drho, samples.back().dA_drho);
            const std::vector<mp_real> M_Z = solve_clamped_second_derivatives(
                rho, Z, samples.front().dZ_drho, samples.back().dZ_drho);
            for (int node = 0; node < kSplineNodes; ++node)
                nodes.push_back({A[node], M_A[node], Z[node], M_Z[node]});
        }

        write_lut<float>(argv[1], nodes, kUseBigEndian);
        write_lut<double>(argv[2], nodes, kUseBigEndian);
        std::cout << "generated " << nodes.size() << " AoS nodes per file (A, M_A, Z, M_Z)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "field LUT generation failed: " << error.what() << '\n';
        return 1;
    }
}
