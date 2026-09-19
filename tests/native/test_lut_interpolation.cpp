#include "2dmodel-hysteresis.hpp"

#include <cmath>
#include <cstdlib>
#include <type_traits>

namespace {

bool close(const real_t actual, const real_t expected)
{
    return std::abs(actual - expected) <= REAL_LITERAL(2e-5) *
           (REAL_LITERAL(1.0) + std::abs(expected));
}

} // namespace

int main()
{
    static_assert(sizeof(LUTInterpolation<false>) == 2 * sizeof(Spline012));
    static_assert(sizeof(LUTInterpolation<true>) == 2 * sizeof(Spline012) + 2 * sizeof(CubicCoeff));

    constexpr real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
    constexpr real_t h = (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * radius /
                         static_cast<real_t>(kSplineNodes - 1);
    constexpr real_t u = REAL_LITERAL(0.37);
    constexpr real_t x = u * h;

    // A(x) = 1 + 2x + 3x^2 + 4x^3; Z(x) = -2 + 0.5x - 1.25x^2 + 0.75x^3.
    const auto r = interpolate_lut_interval<true>(
        REAL_LITERAL(1.0), REAL_LITERAL(6.0), REAL_LITERAL(-2.0), REAL_LITERAL(-2.5),
        REAL_LITERAL(1.0) + REAL_LITERAL(2.0) * h + REAL_LITERAL(3.0) * h * h + REAL_LITERAL(4.0) * h * h * h,
        REAL_LITERAL(6.0) + REAL_LITERAL(24.0) * h,
        REAL_LITERAL(-2.0) + REAL_LITERAL(0.5) * h - REAL_LITERAL(1.25) * h * h + REAL_LITERAL(0.75) * h * h * h,
        REAL_LITERAL(-2.5) + REAL_LITERAL(4.5) * h,
        u);
    const auto compact = interpolate_lut_interval<false>(
        REAL_LITERAL(1.0), REAL_LITERAL(6.0), REAL_LITERAL(-2.0), REAL_LITERAL(-2.5),
        REAL_LITERAL(1.0) + REAL_LITERAL(2.0) * h + REAL_LITERAL(3.0) * h * h + REAL_LITERAL(4.0) * h * h * h,
        REAL_LITERAL(6.0) + REAL_LITERAL(24.0) * h,
        REAL_LITERAL(-2.0) + REAL_LITERAL(0.5) * h - REAL_LITERAL(1.25) * h * h + REAL_LITERAL(0.75) * h * h * h,
        REAL_LITERAL(-2.5) + REAL_LITERAL(4.5) * h,
        u);

    const bool values_ok =
        close(r.A.value, REAL_LITERAL(1.0) + REAL_LITERAL(2.0) * x + REAL_LITERAL(3.0) * x * x + REAL_LITERAL(4.0) * x * x * x) &&
        close(r.A.d1, REAL_LITERAL(2.0) + REAL_LITERAL(6.0) * x + REAL_LITERAL(12.0) * x * x) &&
        close(r.A.d2, REAL_LITERAL(6.0) + REAL_LITERAL(24.0) * x) &&
        close(r.Z.value, REAL_LITERAL(-2.0) + REAL_LITERAL(0.5) * x - REAL_LITERAL(1.25) * x * x + REAL_LITERAL(0.75) * x * x * x) &&
        close(r.Z.d1, REAL_LITERAL(0.5) - REAL_LITERAL(2.5) * x + REAL_LITERAL(2.25) * x * x) &&
        close(r.Z.d2, REAL_LITERAL(-2.5) + REAL_LITERAL(4.5) * x);

    const bool coeffs_ok =
        close(r.A_coeffs.a, REAL_LITERAL(1.0)) && close(r.A_coeffs.b, REAL_LITERAL(2.0) * h) &&
        close(r.A_coeffs.c, REAL_LITERAL(3.0) * h * h) && close(r.A_coeffs.d, REAL_LITERAL(4.0) * h * h * h) &&
        close(r.Z_coeffs.a, REAL_LITERAL(-2.0)) && close(r.Z_coeffs.b, REAL_LITERAL(0.5) * h) &&
        close(r.Z_coeffs.c, REAL_LITERAL(-1.25) * h * h) && close(r.Z_coeffs.d, REAL_LITERAL(0.75) * h * h * h);

    return values_ok && coeffs_ok && close(compact.A.value, r.A.value) && close(compact.Z.d2, r.Z.d2)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
