#include "numeric_extensions/double_double.hpp"
#define USE_DD
#include "hysteresis_types.hpp"

#include <iomanip>
#include <iostream>

namespace {
void print_parts(const char *name, const dd_real &value)
{
    std::cout << name << ".hi=" << std::setprecision(17) << value.hi
              << " " << name << ".lo=" << value.lo << '\n';
}
}

int main()
{
    const dd_real one(1.0);
    const dd_real tiny = dd_real::from_string("1e-30");
    const dd_real recovered = (one + tiny) - one;
    const dd_real third = one / dd_real(3.0);
    const dd_real division_error = third * dd_real(3.0) - one;
    const dd_real root_two = dd_sqrt(dd_real(2.0));
    const dd_real sqrt_error = root_two * root_two - dd_real(2.0);
    const dd_real pi = dd_pi_over_two() * dd_real(2.0);
    const dd_real sine_pi = dd_sin(pi);
    const dd_real atan2_perturbed = dd_atan2(one + tiny, one);
    const real_t real_atan2_perturbed = real_atan2(one + tiny, one);
    const dd_real atan2_expected = dd_pi_over_two() * dd_real(0.5) + tiny * dd_real(0.5);
    const dd_real atan2_negative_axis = dd_atan2(dd_real(0.0), -one);
    const dd_real atan2_large = dd_atan2(dd_real(std::numeric_limits<double>::max()),
                                         dd_real(std::numeric_limits<double>::max()));

    print_parts("tiny", tiny);
    print_parts("recovered", recovered);
    print_parts("division_error", division_error);
    print_parts("sqrt_error", sqrt_error);
    print_parts("sin_pi", sine_pi);
    print_parts("atan2_error", atan2_perturbed - atan2_expected);
    print_parts("atan2_negative_axis_error", atan2_negative_axis - pi);
    print_parts("atan2_large_error", atan2_large - dd_pi_over_two() * dd_real(0.5));

    // A binary64 implementation loses tiny entirely in (1 + tiny) - 1, so
    // this threshold is also a regression guard against silently falling
    // back to ordinary double precision.
    const dd_real tolerance = dd_real::from_string("5e-31");
    if (dd_abs(recovered - tiny) > tolerance || dd_abs(division_error) > tolerance ||
        dd_abs(sqrt_error) > tolerance || dd_abs(sine_pi) > tolerance ||
        dd_abs(atan2_perturbed - atan2_expected) > tolerance ||
        dd_abs(real_atan2_perturbed - atan2_expected) > tolerance ||
        dd_abs(atan2_negative_axis - pi) > tolerance ||
        dd_abs(atan2_large - dd_pi_over_two() * dd_real(0.5)) > tolerance)
        return 1;
    return 0;
}
