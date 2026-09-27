// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <iomanip>
#include <iostream>

int main()
{
    using real_t = boost::multiprecision::cpp_dec_float_100;

    const real_t root_two = sqrt(real_t(2));
    const real_t residual = abs(root_two * root_two - real_t(2));

    if (residual > real_t("1e-90")) {
        std::cerr << "boost::multiprecision accuracy check failed: " << residual << '\n';
        return 1;
    }

    std::cout << std::setprecision(50) << root_two << '\n';
    return 0;
}
