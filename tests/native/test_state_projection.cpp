#include "2dmodel-hysteresis.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

using A = accumulator_t;
static_assert(!std::is_same_v<A, real_t>, "Run this test with USE_KAHAN");
static_assert(std::is_same_v<decltype(A{} * A{}), A>);
static_assert(std::is_same_v<decltype(A{} / A{}), A>);
static_assert(std::is_same_v<decltype((A{} + A{} * A{} * A{}) / A{}), A>);
static_assert(std::is_same_v<decltype(A{} * StateVec2{} + StateVec2{}), StateVec2>);
static_assert(std::is_same_v<decltype(real_t{} * State{} + State{}), State>);

std::array<A *, 15> components(State &s)
{
    return {&s.down_disk_angle, &s.pos.x, &s.pos.y, &s.speed.x, &s.speed.y,
            &s.attitude.real, &s.attitude.imagine.x, &s.attitude.imagine.y,
            &s.attitude.imagine.z, &s.omega.x, &s.omega.y, &s.omega.z,
            &s.q.x, &s.q.y, &s.q.z};
}

int main()
{
    const real_t epsilon = std::numeric_limits<real_t>::epsilon();
    const real_t tiny = epsilon / REAL_LITERAL(8.0);
    State out{}, in{};
    for (A *v : components(out)) {
        *v = REAL_LITERAL(1.0);
        v->correction = -tiny; // Represents 1 + tiny; real_t alone loses tiny.
    }
    for (A *v : components(in)) {
        *v = REAL_LITERAL(-2.0);
        v->correction = -tiny;
    }
    const real_t dt = REAL_LITERAL(2.0);
    // This is a State multiply-add expression, not a hardware fused FMA.
    State ans = dt * out + in;
    const auto result = components(ans);
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (result[i]->value != REAL_LITERAL(3.0) * tiny || result[i]->correction != REAL_LITERAL(0.0)) {
            std::cerr << "State multiply-add lost compensation at component " << i << '\n';
            return 1;
        }
    }

    SolverParams params{};
    params.alpha = REAL_LITERAL(1.0);
    State baseline{};
    baseline.speed.x = REAL_LITERAL(1.0);
    State perturbed = baseline;
    perturbed.speed.x.correction = -tiny;
    perturbed.omega.z = REAL_LITERAL(2.0);
    perturbed.omega.z.correction = -tiny;
    project_pure_rolling(baseline, params);
    project_pure_rolling(perturbed, params);

    // Linear projection: a tiny change in vx gives tiny/(1+eta) in both
    // vx and omega_y for alpha=1. Subtract as accumulators before observing
    // the small difference, so rounding the large output cannot hide loss.
    const real_t expected = tiny / (REAL_LITERAL(1.0) + Eta);
    const real_t tolerance = REAL_LITERAL(64.0) * epsilon * tiny;
    const A dv = perturbed.speed.x - baseline.speed.x;
    const A dw = perturbed.omega.y - baseline.omega.y;
    if (std::abs((dv.value - dv.correction) - expected) > tolerance ||
        std::abs((dw.value - dw.correction) - expected) > tolerance) {
        std::cerr << "Projection lost the input low component\n";
        return 1;
    }
    if (perturbed.omega.z.value != REAL_LITERAL(2.0) || perturbed.omega.z.correction != -tiny) {
        std::cerr << "Projection changed omega_z compensation\n";
        return 1;
    }
    const A slip_x = perturbed.speed.x - params.alpha * perturbed.omega.y;
    const A slip_y = perturbed.speed.y + params.alpha * perturbed.omega.x;
    if (std::abs(slip_x.value) > REAL_LITERAL(16.0) * epsilon ||
        std::abs(slip_y.value) > REAL_LITERAL(16.0) * epsilon) {
        std::cerr << "Projection violated the rolling constraint\n";
        return 1;
    }
    std::cout << "PASS: projection and all 15 State multiply-add components retain compensation\n";
}
