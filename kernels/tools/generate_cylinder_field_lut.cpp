// SPDX-License-Identifier: AGPL-3.0-only
// Uniformly axially magnetized finite cylinders; Beta is NOT baked into the LUT.
#include "field_lut_common.hpp"
#include "finite_cylinder_config.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <memory>

namespace {
using field_lut::radial_sample;
using mp_real = boost::multiprecision::cpp_dec_float_50;
using ModeValues = std::array<mp_real, kFieldFourierModeCount>;
struct SpectralNode { mp_real k; ModeValues weights; };
ModeValues bessel_modes(const mp_real &x, ModeValues *derivatives = nullptr)
{
    // Start with two accurately evaluated high-order Bessels, then recurse
    // downward; this avoids unstable upward recurrence when x is small.
    const int highest = (2*kFieldFourierModeCount-1)*magnet_number/2;
    mp_real next = boost::math::cyl_bessel_j(highest+1, x);
    mp_real current = boost::math::cyl_bessel_j(highest, x);
    ModeValues values{};
    for (int n = highest; n >= magnet_number/2; --n) {
        if (n % magnet_number == magnet_number/2) {
            const int mode = (n-magnet_number/2)/magnet_number;
            values[mode] = current;
            if (derivatives) (*derivatives)[mode] = mp_real(n)*current/x-next;
        }
        const mp_real previous = 2*mp_real(n)*current/x-next;
        next = current;
        current = previous;
    }
    return values;
}

struct CylinderQuadrature {
    std::vector<SpectralNode> nodes;

    std::array<radial_sample, kFieldFourierModeCount> evaluate(const mp_real &rho) const
    {
        std::array<radial_sample, kFieldFourierModeCount> result{};
        for (const auto &node : nodes) {
            ModeValues derivatives;
            const auto values = bessel_modes(node.k*rho, &derivatives);
            for (int mode = 0; mode < kFieldFourierModeCount; ++mode) {
                const mp_real term = node.weights[mode]*values[mode];
                const mp_real derivative = node.weights[mode]*node.k*derivatives[mode];
                result[mode].A += field_lut::mp_real(term);
                result[mode].dA_drho += field_lut::mp_real(derivative);
                result[mode].Z -= field_lut::mp_real(node.k*term);
                result[mode].dZ_drho -= field_lut::mp_real(node.k*derivative);
            }
        }
        return result;
    }
};

CylinderQuadrature make_quadrature( const mp_real &upper, int refinement)
{
    const mp_real a(IYPT16_STRINGIZE(IYPT16_CYLINDER_RADIUS));
    const mp_real L(IYPT16_STRINGIZE(IYPT16_CYLINDER_THICKNESS));
    const mp_real h(IYPT16_STRINGIZE(IYPT16_FIELD_HEIGHT));
    const mp_real R = mp_real(magnet_number) / 2;
    const mp_real gap = h - L / 2;
    if (a <= 0 || L <= 0 || gap <= 0)
        throw std::runtime_error("cylinder radius/thickness must be positive and height must exceed thickness/2");
    // |F_a|, |J_m| <= 1 and G <= k exp(-gap*k). Bound the tails of
    // A, A_rho, Z, Z_rho with powers k ... k^3, independently of cancellation.
    mp_real cutoff = 80 / gap;
    for (;;) {
        mp_real worst = 0;
        for (int p = 1; p <= 3; ++p) {
            mp_real tail = 1 / gap;
            for (int j = 1; j <= p; ++j)
                tail = pow(cutoff, j) / gap + mp_real(j) * tail / gap;
            worst = std::max(worst, mp_real(mp_real(2 * magnet_number) * exp(-gap * cutoff) * tail));
        }
        if (worst < mp_real("1e-25")) break;
        cutoff += 20 / gap;
    }
    const mp_real max_width = 16 / (upper + R + a + gap) / refinement;
    const int panels = mp_real(ceil(cutoff / max_width)).convert_to<int>();
    if (panels > 100000)
        throw std::runtime_error("cylinder quadrature needs more than 100000 panels; increase the physical gap");
    const mp_real width = cutoff / panels;
    const auto &xs = boost::math::quadrature::gauss<mp_real, 32>::abscissa();
    const auto &ws = boost::math::quadrature::gauss<mp_real, 32>::weights();
    CylinderQuadrature mode;
    mode.nodes.reserve(static_cast<std::size_t>(panels) * 32);
    for (int panel = 0; panel < panels; ++panel) {
        const mp_real mid = (mp_real(panel) + mp_real("0.5")) * width;
        for (std::size_t j = 0; j < xs.size(); ++j) {
            for (int sign : {-1, 1}) {
                const mp_real k = mid + sign * width * xs[j] / 2;
                const mp_real form = 2 * boost::math::cyl_bessel_j(1, k * a) / (k * a);
                // Stable at small k*L, and no exp(+k*L/2) overflow.
                const mp_real G = exp(-k * gap) * (-expm1(-k * L)) / L;
                const mp_real weight = width / 2 * ws[j] * (2 * magnet_number) * G * form;
                auto weights = bessel_modes(k*R);
                for (auto &value : weights) value *= weight;
                mode.nodes.push_back({k, weights});
            }
        }
    }
    return mode;
}

CylinderQuadrature checked_quadrature( const mp_real &upper)
{
    auto coarse = make_quadrature(upper, 1);
    // Compare independent panel resolutions before sampling the full table.
    const mp_real lower = mp_real(IYPT16_STRINGIZE(IYPT16_LUT_LOWER_BOUND_RATIO)) * magnet_number / 2;
    for (int refinement : {2, 4, 8}) {
        auto fine = make_quadrature(upper, refinement);
        mp_real error = 0;
        for (int j = 0; j < 5; ++j) {
            const mp_real rho = lower + (upper - lower) * j / 4;
            const auto cs = coarse.evaluate(rho);
            const auto fs = fine.evaluate(rho);
            for (int mode = 0; mode < kFieldFourierModeCount; ++mode) {
                const auto &c = cs[mode], &f = fs[mode];
                for (const mp_real &difference : {mp_real(c.A-f.A), mp_real(c.dA_drho-f.dA_drho),
                                                 mp_real(c.Z-f.Z), mp_real(c.dZ_drho-f.dZ_drho)})
                    error = std::max(error, mp_real(abs(difference)));
            }
        }
        if (error < mp_real("1e-20")) {
            std::cout << "cylinder (all harmonics): " << coarse.nodes.size()
                      << " spectral nodes, panel-refinement error=" << std::scientific << error
                      << std::fixed << '\n' << std::flush;
            return coarse;
        }
        coarse = std::move(fine);
    }
    throw std::runtime_error("cylinder spectral quadrature did not converge to 1e-20");
}
} // namespace

int main(int argc, char **argv)
{
    using Samples = std::vector<std::array<radial_sample, kFieldFourierModeCount>>;
    std::shared_ptr<Samples> samples;
    return field_lut::generate_lut(argc, argv, [&samples](int harmonic, const field_lut::mp_real &upper) {
        const mp_real lower = mp_real(IYPT16_STRINGIZE(IYPT16_LUT_LOWER_BOUND_RATIO))*magnet_number/2;
        const mp_real spacing = (mp_real(upper)-lower)/(kSplineNodes-1);
        if (!samples) {
            auto quadrature = checked_quadrature(mp_real(upper));
            samples = std::make_shared<Samples>(kSplineNodes);
            std::exception_ptr first_error;
#pragma omp parallel for schedule(dynamic, 1)
            for (int node = 0; node < kSplineNodes; ++node) {
                try {
                    (*samples)[node] = quadrature.evaluate(lower+spacing*node);
                } catch (...) {
#pragma omp critical(cylinder_sampling_error)
                    { if (!first_error) first_error = std::current_exception(); }
                }
            }
            if (first_error) std::rethrow_exception(first_error);
        }
        const int mode = (harmonic-magnet_number/2)/magnet_number;
        return [samples, lower, spacing, mode](const field_lut::mp_real &rho,
                    const field_lut::mp_real &, const field_lut::mp_real &, int) {
            const int node = mp_real(floor((mp_real(rho)-lower)/spacing+mp_real("0.5"))).convert_to<int>();
            return samples->at(node)[mode];
        };
    });
}
