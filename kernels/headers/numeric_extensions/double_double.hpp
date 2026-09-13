#pragma once

// A compact CPU-only double-double type.  A value is represented as the
// unevaluated sum hi + lo, with |lo| normally no larger than half an ulp of hi.
// This provides about 106 significand bits (roughly 31-32 decimal digits).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

struct dd_real {
    double hi;
    double lo;

    constexpr dd_real() noexcept : hi(0.0), lo(0.0) {}
    constexpr dd_real(const double value) noexcept : hi(value), lo(0.0) {}
    constexpr dd_real(const double high, const double low) noexcept : hi(high), lo(low) {}

    template <typename Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
    constexpr dd_real(const Integer value) noexcept : hi(static_cast<double>(value)), lo(0.0) {}

    explicit constexpr operator double() const noexcept { return hi + lo; }

    static dd_real from_string(const char *text);

    dd_real &operator+=(const dd_real &other) noexcept;
    dd_real &operator-=(const dd_real &other) noexcept;
    dd_real &operator*=(const dd_real &other) noexcept;
    dd_real &operator/=(const dd_real &other) noexcept;
};

namespace dd_detail {
inline dd_real quick_two_sum(const double a, const double b) noexcept
{
    const double sum = a + b;
    return {sum, b - (sum - a)};
}

inline dd_real two_sum(const double a, const double b) noexcept
{
    const double sum = a + b;
    const double bb = sum - a;
    return {sum, (a - (sum - bb)) + (b - bb)};
}

inline dd_real two_product(const double a, const double b) noexcept
{
    const double product = a * b;
    return {product, std::fma(a, b, -product)};
}

inline dd_real normalized(const double high, const double low) noexcept
{
    return quick_two_sum(high, low);
}
} // namespace dd_detail

inline dd_real operator+(const dd_real &a, const dd_real &b) noexcept
{
    dd_real s = dd_detail::two_sum(a.hi, b.hi);
    dd_real t = dd_detail::two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = dd_detail::quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return dd_detail::quick_two_sum(s.hi, s.lo);
}

inline dd_real operator-(const dd_real &a, const dd_real &b) noexcept
{
    return a + dd_real(-b.hi, -b.lo);
}

inline dd_real operator-(const dd_real &value) noexcept { return {-value.hi, -value.lo}; }
inline dd_real operator+(const dd_real &value) noexcept { return value; }

inline dd_real operator*(const dd_real &a, const dd_real &b) noexcept
{
    dd_real product = dd_detail::two_product(a.hi, b.hi);
    product.lo += a.hi * b.lo + a.lo * b.hi;
    product = dd_detail::quick_two_sum(product.hi, product.lo);
    product.lo += a.lo * b.lo;
    return dd_detail::quick_two_sum(product.hi, product.lo);
}

inline dd_real operator/(const dd_real &a, const dd_real &b) noexcept
{
    const double q1 = a.hi / b.hi;
    dd_real remainder = a - b * dd_real(q1);
    const double q2 = remainder.hi / b.hi;
    remainder -= b * dd_real(q2);
    const double q3 = remainder.hi / b.hi;
    return dd_detail::normalized(q1, q2) + dd_real(q3);
}

inline dd_real &dd_real::operator+=(const dd_real &other) noexcept { return *this = *this + other; }
inline dd_real &dd_real::operator-=(const dd_real &other) noexcept { return *this = *this - other; }
inline dd_real &dd_real::operator*=(const dd_real &other) noexcept { return *this = *this * other; }
inline dd_real &dd_real::operator/=(const dd_real &other) noexcept { return *this = *this / other; }

inline dd_real dd_real::from_string(const char *text)
{
    if (text == nullptr)
        throw std::invalid_argument("dd_real: null string");
    const char *cursor = text;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r')
        ++cursor;
    bool negative = false;
    if (*cursor == '+' || *cursor == '-') {
        negative = *cursor == '-';
        ++cursor;
    }
    dd_real value(0.0);
    int fractional_digits = 0;
    bool after_point = false;
    bool have_digit = false;
    while ((*cursor >= '0' && *cursor <= '9') || *cursor == '.') {
        if (*cursor == '.') {
            if (after_point)
                throw std::invalid_argument("dd_real: multiple decimal points");
            after_point = true;
        } else {
            have_digit = true;
            value = value * dd_real(10.0) + dd_real(*cursor - '0');
            if (after_point)
                ++fractional_digits;
        }
        ++cursor;
    }
    if (!have_digit)
        throw std::invalid_argument("dd_real: no digits");
    int exponent = -fractional_digits;
    if (*cursor == 'e' || *cursor == 'E') {
        ++cursor;
        char *end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end == cursor)
            throw std::invalid_argument("dd_real: malformed exponent");
        exponent += static_cast<int>(parsed);
        cursor = end;
    }
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r')
        ++cursor;
    if (*cursor != '\0')
        throw std::invalid_argument("dd_real: trailing characters");
    dd_real factor(1.0), base(10.0);
    unsigned int power = static_cast<unsigned int>(exponent < 0 ? -exponent : exponent);
    while (power != 0) {
        if (power & 1U)
            factor *= base;
        base *= base;
        power >>= 1U;
    }
    if (exponent < 0)
        value /= factor;
    else
        value *= factor;
    return negative ? -value : value;
}

inline bool operator==(const dd_real &a, const dd_real &b) noexcept { return a.hi == b.hi && a.lo == b.lo; }
inline bool operator!=(const dd_real &a, const dd_real &b) noexcept { return !(a == b); }
inline bool operator<(const dd_real &a, const dd_real &b) noexcept
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool operator>(const dd_real &a, const dd_real &b) noexcept { return b < a; }
inline bool operator<=(const dd_real &a, const dd_real &b) noexcept { return !(b < a); }
inline bool operator>=(const dd_real &a, const dd_real &b) noexcept { return !(a < b); }

inline bool dd_isfinite(const dd_real &value) noexcept
{
    return std::isfinite(value.hi) && std::isfinite(value.lo);
}

inline dd_real dd_abs(const dd_real &value) noexcept { return value < dd_real(0.0) ? -value : value; }

inline dd_real dd_sqrt(const dd_real &value) noexcept
{
    if (value < dd_real(0.0))
        return {std::numeric_limits<double>::quiet_NaN(), 0.0};
    if (value == dd_real(0.0) || !dd_isfinite(value))
        return value;
    dd_real estimate(std::sqrt(value.hi));
    estimate += (value - estimate * estimate) / (dd_real(2.0) * estimate);
    estimate += (value - estimate * estimate) / (dd_real(2.0) * estimate);
    return estimate;
}

inline const dd_real &dd_pi_over_two() noexcept
{
    static const dd_real value(1.5707963267948966, 6.12323399573676603587e-17);
    return value;
}

inline const dd_real &dd_two_pi() noexcept
{
    static const dd_real value(6.2831853071795862, 2.44929359829470641435e-16);
    return value;
}

inline dd_real dd_remainder(const dd_real &value, const dd_real &divisor) noexcept
{
    if (!dd_isfinite(value) || !dd_isfinite(divisor) || divisor == dd_real(0.0))
        return {std::numeric_limits<double>::quiet_NaN(), 0.0};
    const double quotient = std::nearbyint(static_cast<double>(value / divisor));
    dd_real result = value - divisor * dd_real(quotient);
    const dd_real half = dd_abs(divisor) * dd_real(0.5);
    if (result > half)
        result -= dd_abs(divisor);
    else if (result < -half)
        result += dd_abs(divisor);
    return result;
}

inline void dd_sincos(const dd_real &angle, dd_real &sine, dd_real &cosine) noexcept
{
    const double quadrant_value = std::nearbyint(static_cast<double>(angle / dd_pi_over_two()));
    const auto quadrant = static_cast<std::int64_t>(quadrant_value);
    const dd_real reduced = angle - dd_pi_over_two() * dd_real(quadrant_value);
    const dd_real minus_square = -(reduced * reduced);

    dd_real sin_term = reduced;
    dd_real cos_term(1.0);
    dd_real sin_sum = sin_term;
    dd_real cos_sum = cos_term;
    for (int n = 1; n <= 32; ++n) {
        sin_term *= minus_square / dd_real((2 * n) * (2 * n + 1));
        cos_term *= minus_square / dd_real((2 * n - 1) * (2 * n));
        sin_sum += sin_term;
        cos_sum += cos_term;
        if (std::fabs(sin_term.hi) < 1e-35 && std::fabs(cos_term.hi) < 1e-35)
            break;
    }
    switch (static_cast<int>((quadrant % 4 + 4) % 4)) {
    case 0: sine = sin_sum; cosine = cos_sum; break;
    case 1: sine = cos_sum; cosine = -sin_sum; break;
    case 2: sine = -sin_sum; cosine = -cos_sum; break;
    default: sine = -cos_sum; cosine = sin_sum; break;
    }
}

inline dd_real dd_sin(const dd_real &value) noexcept
{
    dd_real sine, cosine;
    dd_sincos(dd_remainder(value, dd_two_pi()), sine, cosine);
    return sine;
}

inline dd_real dd_cos(const dd_real &value) noexcept
{
    dd_real sine, cosine;
    dd_sincos(dd_remainder(value, dd_two_pi()), sine, cosine);
    return cosine;
}

namespace std {
template <> class numeric_limits<dd_real> {
public:
    static constexpr bool is_specialized = true;
    static constexpr int digits = 106;
    static constexpr int digits10 = 31;
    static constexpr int max_digits10 = 33;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr dd_real min() noexcept { return dd_real(numeric_limits<double>::min()); }
    static constexpr dd_real max() noexcept { return dd_real(numeric_limits<double>::max()); }
    static constexpr dd_real lowest() noexcept { return dd_real(-numeric_limits<double>::max()); }
    static constexpr dd_real epsilon() noexcept { return dd_real(4.9303806576313238e-32); }
    static constexpr dd_real round_error() noexcept { return dd_real(0.5); }
    static constexpr dd_real infinity() noexcept { return dd_real(numeric_limits<double>::infinity()); }
    static constexpr dd_real quiet_NaN() noexcept { return dd_real(numeric_limits<double>::quiet_NaN()); }
    static constexpr dd_real signaling_NaN() noexcept { return dd_real(numeric_limits<double>::signaling_NaN()); }
    static constexpr dd_real denorm_min() noexcept { return dd_real(numeric_limits<double>::denorm_min()); }
};
} // namespace std
