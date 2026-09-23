#include "2dmodel-hysteresis.hpp"

#include <cassert>

int main()
{
    const SolverParams params = {REAL_LITERAL(0.33333333), REAL_LITERAL(0.0),
                                 REAL_LITERAL(0.16560356), REAL_LITERAL(489.39639),
                                 REAL_LITERAL(11011.419)};
    const Vec2 above_magnet = {static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0),
                               REAL_LITERAL(0.0)};
    const Vec3 zero_q{};
    const auto result = eval_mag_field_and_force(params, above_magnet, zero_q, REAL_LITERAL(0.0));
    // The induced magnetic moment is attracted toward the lower magnet.
    assert(result.Force.z < REAL_LITERAL(0.0));
    assert(params.gamma_g - result.Force.z > REAL_LITERAL(0.0));
}
