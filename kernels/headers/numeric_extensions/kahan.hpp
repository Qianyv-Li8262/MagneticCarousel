#pragma once

#ifndef __CUDACC__
#include <cmath>
#endif

// Compensated floating-point accumulator.  The correction records the part of
// an increment lost when it is added to the main value.  Copying the object
// preserves that correction, which lets an RK state carry it between steps.
template <typename T> struct kahan {
    T value;
    T correction;

    DEVICE HOST INLINE kahan() : value(static_cast<T>(0)), correction(static_cast<T>(0)) {}
    DEVICE HOST INLINE kahan(const T initial) : value(initial), correction(static_cast<T>(0)) {}

    DEVICE HOST INLINE operator T() const { return value; }

    DEVICE HOST INLINE kahan &operator=(const T replacement)
    {
        value = replacement;
        correction = static_cast<T>(0);
        return *this;
    }

    DEVICE HOST INLINE void add(const T increment)
    {
        // TwoSum retains the low component even under near cancellation,
        // which is essential when computing a division residual.
        const T sum = value + increment;
        const T recovered = sum - value;
        const T error = (value - (sum - recovered)) + (increment - recovered);
        const T low = error - correction;
        const T next = sum + low;
        const T recovered_low = next - sum;
        correction = -((sum - (next - recovered_low)) + (low - recovered_low));
        value = next;
    }

    DEVICE HOST INLINE kahan &operator+=(const T increment)
    {
        add(increment);
        return *this;
    }

    DEVICE HOST INLINE kahan &operator-=(const T decrement)
    {
        add(-decrement);
        return *this;
    }

    DEVICE HOST INLINE kahan &operator+=(const kahan &other)
    {
        add(other.value);
        add(-other.correction);
        return *this;
    }

    DEVICE HOST INLINE kahan &operator-=(const kahan &other)
    {
        add(-other.value);
        add(other.correction);
        return *this;
    }
};

template <typename T> DEVICE HOST INLINE kahan<T> operator+(kahan<T> left, const kahan<T> &right)
{
    left += right;
    return left;
}

template <typename T> DEVICE HOST INLINE kahan<T> operator-(kahan<T> left, const kahan<T> &right)
{
    left -= right;
    return left;
}

// A stored pair represents value - correction. Keep the rounding residual of
// the leading product, as well as both operands' low components.
template <typename T> DEVICE HOST INLINE T kahan_product_error(T a, T b, T product)
{
    return a * b - product;
}
DEVICE HOST INLINE float kahan_product_error(float a, float b, float product)
{
#ifdef __CUDACC__
    return fmaf(a, b, -product);
#else
    return std::fma(a, b, -product);
#endif
}
DEVICE HOST INLINE double kahan_product_error(double a, double b, double product)
{
#ifdef __CUDACC__
    return fma(a, b, -product);
#else
    return std::fma(a, b, -product);
#endif
}

template <typename T> DEVICE HOST INLINE kahan<T> operator-(const kahan<T> &a)
{
    kahan<T> result;
    result.value = -a.value;
    result.correction = -a.correction;
    return result;
}

template <typename T> DEVICE HOST INLINE kahan<T> operator*(const kahan<T> &a, const kahan<T> &b)
{
    kahan<T> result(a.value * b.value);
    const T error = kahan_product_error(a.value, b.value, result.value);
    result += error;
    result -= a.value * b.correction;
    result -= a.correction * b.value;
    result += a.correction * b.correction;
    return result;
}

template <typename T> DEVICE HOST INLINE kahan<T> operator/(const kahan<T> &a, const kahan<T> &b)
{
    kahan<T> result(a.value / b.value);
    // Refine the leading quotient using the compensated residual a - q*b.
    const kahan<T> residual = a - result * b;
    result += residual.value / b.value;
    result -= residual.correction / b.value;
    return result;
}

// Exact scalar matches avoid intercepting vector and quaternion operators.
template <typename T> DEVICE HOST INLINE kahan<T> operator*(const kahan<T> &a, const T &b)
{ return a * kahan<T>(b); }
template <typename T> DEVICE HOST INLINE kahan<T> operator*(const T &a, const kahan<T> &b)
{ return kahan<T>(a) * b; }
template <typename T> DEVICE HOST INLINE kahan<T> operator/(const kahan<T> &a, const T &b)
{ return a / kahan<T>(b); }
template <typename T> DEVICE HOST INLINE kahan<T> operator/(const T &a, const kahan<T> &b)
{ return kahan<T>(a) / b; }
template <typename T> DEVICE HOST INLINE kahan<T> operator+(kahan<T> a, const T &b)
{ a += b; return a; }
template <typename T> DEVICE HOST INLINE kahan<T> operator+(const T &a, kahan<T> b)
{ b += a; return b; }
template <typename T> DEVICE HOST INLINE kahan<T> operator-(kahan<T> a, const T &b)
{ a -= b; return a; }
template <typename T> DEVICE HOST INLINE kahan<T> operator-(const T &a, const kahan<T> &b)
{ return kahan<T>(a) - b; }
template <typename T> DEVICE HOST INLINE kahan<T> &operator*=(kahan<T> &a, const kahan<T> &b)
{ a = a * b; return a; }
template <typename T> DEVICE HOST INLINE kahan<T> &operator/=(kahan<T> &a, const kahan<T> &b)
{ a = a / b; return a; }
template <typename T> DEVICE HOST INLINE kahan<T> &operator*=(kahan<T> &a, const T &b)
{ a = a * b; return a; }
template <typename T> DEVICE HOST INLINE kahan<T> &operator/=(kahan<T> &a, const T &b)
{ a = a / b; return a; }
