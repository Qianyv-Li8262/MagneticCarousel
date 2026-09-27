// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#pragma once

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
        const T corrected = increment - correction;
        const T next = value + corrected;
        correction = (next - value) - corrected;
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
