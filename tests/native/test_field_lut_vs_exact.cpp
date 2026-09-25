#include "2dmodel-hysteresis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{
FieldResult exact_field(const Vec2 pos, const real_t disk_angle)
{
    Vec3 h{0, 0, 0};
    TracelessSymTensor grad{{0, 0, 0, 0, 0}};
    const real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
    real_t sign = REAL_LITERAL(1.0);
    for (int i = 0; i < magnet_number; ++i) {
        const real_t angle = static_cast<real_t>(i) * magnet_phi0 + disk_angle;
        const real_t dx = pos.x - radius * std::cos(angle);
        const real_t dy = pos.y - radius * std::sin(angle);
        const real_t r2 = dx * dx + dy * dy;
        const real_t h2 = height * height;
        const real_t q = r2 + h2;
        const real_t invq = REAL_LITERAL(1.0) / q;
        const real_t field_scale = sign * invq * invq * std::sqrt(invq);
        const real_t grad_scale = REAL_LITERAL(3.0) * field_scale * invq;
        h.x += field_scale * REAL_LITERAL(3.0) * height * dx;
        h.y += field_scale * REAL_LITERAL(3.0) * height * dy;
        h.z += field_scale * (REAL_LITERAL(2.0) * h2 - r2);
        grad.component[0] += grad_scale * height * (q - REAL_LITERAL(5.0) * dx * dx);
        grad.component[1] -= grad_scale * REAL_LITERAL(5.0) * height * dx * dy;
        grad.component[2] += grad_scale * dx * (r2 - REAL_LITERAL(4.0) * h2);
        grad.component[3] += grad_scale * height * (q - REAL_LITERAL(5.0) * dy * dy);
        grad.component[4] += grad_scale * dy * (r2 - REAL_LITERAL(4.0) * h2);
        sign = -sign;
    }
    h = h * Beta;
    for (real_t &component : grad.component)
        component *= Beta;
    return {h, grad};
}

std::array<real_t, 8> components(const FieldResult &field)
{
    return {field.h.x, field.h.y, field.h.z,
            field.hGrad3d.component[0], field.hGrad3d.component[1],
            field.hGrad3d.component[2], field.hGrad3d.component[3],
            field.hGrad3d.component[4]};
}
} // namespace

int main()
{
    const char *path = IYPT16_TEST_LUT_PATH;
    if (kUseBigEndian) {
        std::cerr << "This test requires the native-endian FP64 LUT\n";
        return EXIT_FAILURE;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(kFieldFourierModeCount * kSplineNodes * sizeof(LUTNode));
    if (!file || file.tellg() != expected_bytes) {
        std::cerr << "Missing or wrong-size FP64 field LUT: " << path << '\n';
        return EXIT_FAILURE;
    }
    std::vector<LUTNode> lut(kFieldFourierModeCount * kSplineNodes);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(lut.data()), expected_bytes);
    if (!file) {
        std::cerr << "Could not read FP64 field LUT\n";
        return EXIT_FAILURE;
    }

    const real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
    const real_t rho_min = kLUTLowerBoundRatio * radius;
    const real_t drho = (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * radius /
                         static_cast<real_t>(kSplineNodes - 1);
    std::array<real_t, 8> peak_reference{};
    std::array<real_t, 8> peak_error{};
    std::array<real_t, 8> worst_rho{};
    std::array<real_t, 8> worst_phi{};
    FieldCache cache{};
    // Sample every radial cell, including interior points where spline curvature
    // errors are largest, and phases between the alternating magnet axes.
    for (int cell = 0; cell < kSplineNodes - 1; ++cell) {
        for (const real_t fraction : {REAL_LITERAL(0.25), REAL_LITERAL(0.5), REAL_LITERAL(0.75)}) {
            const real_t rho = rho_min + (static_cast<real_t>(cell) + fraction) * drho;
            for (int phase = 0; phase < 16; ++phase) {
                const real_t phi = real_two_pi() * (static_cast<real_t>(phase) + REAL_LITERAL(0.37)) /
                                   static_cast<real_t>(16);
                const real_t disk_angle = REAL_LITERAL(0.137);
                const Vec2 pos = make_Vec2(rho * std::cos(phi), rho * std::sin(phi));
                const auto exact = components(exact_field(pos, disk_angle));
                const auto spline = components(eval_mag_field(pos, disk_angle, &cache, lut.data()));
                for (std::size_t i = 0; i < exact.size(); ++i) {
                    peak_reference[i] = std::max(peak_reference[i], std::abs(exact[i]));
                    const real_t error = std::abs(spline[i] - exact[i]);
                    if (error > peak_error[i]) {
                        peak_error[i] = error;
                        worst_rho[i] = rho;
                        worst_phi[i] = phi;
                    }
                }
            }
        }
    }

    // Normalize each component by its own peak, so weak components cannot be
    // hidden by a stronger field or gradient component.
    constexpr real_t tolerance = REAL_LITERAL(1e-4);
    bool passed = true;
    const char *names[] = {"Hx", "Hy", "Hz", "Gxx", "Gxy", "Gxz", "Gyy", "Gyz"};
    for (std::size_t i = 0; i < peak_error.size(); ++i) {
        const real_t relative_error = peak_error[i] / peak_reference[i];
        std::cout << names[i] << ": max error / peak exact = " << relative_error
                  << " (rho=" << worst_rho[i] << ", phi=" << worst_phi[i] << ")\n";
        if (!std::isfinite(relative_error) || relative_error > tolerance)
            passed = false;
    }
    if (!passed)
        std::cerr << "Current Fourier mode count and spline node count exceed the field error tolerance "
                  << tolerance << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
