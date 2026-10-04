// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "finite_cylinder_config.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <array>
#include <vector>

// Independent real-space reference: exact surface-charge field and its
// analytically differentiated kernel, integrated over the two circular faces.
// No Fourier truncation, Bessel functions, or generated LUT is used here.
namespace cylinder_reference {
struct DiskNode { double x, y, weight; };

template<int Order> std::vector<DiskNode> disk_nodes(int angles)
{
    const double a = IYPT16_CYLINDER_RADIUS;
    const auto &xs = boost::math::quadrature::gauss<double, Order>::abscissa();
    const auto &ws = boost::math::quadrature::gauss<double, Order>::weights();
    std::vector<DiskNode> nodes;
    for (std::size_t j = 0; j < xs.size(); ++j) {
        for (int sign : {-1, 1}) {
            const double u = (1 + sign * xs[j]) / 2;
            const double radius = a * std::sqrt(u);
            for (int p = 0; p < angles; ++p) {
                const double theta = real_two_pi() * p / angles;
                // Normalized disk area: du dtheta/(2*pi).
                nodes.push_back({radius * std::cos(theta), radius * std::sin(theta), ws[j]/(2*angles)});
            }
        }
    }
    return nodes;
}

inline FieldResult field(Vec2 pos, real_t disk_angle, bool refined=false)
{
    static const auto base = disk_nodes<16>(64);
    static const auto fine = disk_nodes<32>(128);
    const auto &nodes = refined ? fine : base;
    const double L = IYPT16_CYLINDER_THICKNESS;
    const double R = magnet_number / 2.0;
    std::array<long double, 8> sum{};
    for (int magnet = 0; magnet < magnet_number; ++magnet) {
        const double angle = magnet * real_two_pi() / magnet_number + disk_angle;
        const double cx = R * std::cos(angle), cy = R * std::sin(angle);
        const double polarity = magnet % 2 ? -1 : 1;
        for (const auto &node : nodes) {
            const double dx = pos.x-cx-node.x, dy=pos.y-cy-node.y;
            for (int face : {-1, 1}) {
                const double dz = height - face*L/2;
                const double r2=dx*dx+dy*dy+dz*dz;
                const double inv3=1/(r2*std::sqrt(r2));
                const double factor=polarity*face*node.weight/L*Beta;
                const double inv5=inv3/r2;
                sum[0]+=factor*dx*inv3; sum[1]+=factor*dy*inv3; sum[2]+=factor*dz*inv3;
                sum[3]+=factor*(inv3-3*dx*dx*inv5);
                sum[4]-=factor*3*dx*dy*inv5;
                sum[5]-=factor*3*dx*dz*inv5;
                sum[6]+=factor*(inv3-3*dy*dy*inv5);
                sum[7]-=factor*3*dy*dz*inv5;
            }
        }
    }
    return {{static_cast<real_t>(sum[0]),static_cast<real_t>(sum[1]),static_cast<real_t>(sum[2])},
            {{static_cast<real_t>(sum[3]),static_cast<real_t>(sum[4]),static_cast<real_t>(sum[5]),
              static_cast<real_t>(sum[6]),static_cast<real_t>(sum[7])}}};
}
} // namespace cylinder_reference
