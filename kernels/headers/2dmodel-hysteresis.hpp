#pragma once
#include "hysteresis_config.hpp"
#include "hysteresis_config_field.hpp"
#include "hysteresis_types.hpp"

struct TracelessSymTensor {
    real_t component[5];
};
struct FieldResult {
    Vec3 h;
    TracelessSymTensor hGrad3d;
};
struct FieldForceResult {
    Vec3 h;
    Vec3 Force;
    TracelessSymTensor hGrad3d;
};

struct CubicCoeff {
    real_t a, b, c, d; // if cache mode is caching endpoint, u should interpret them as A_left, M_left, A_right, M_right
};

struct Spline012 {
    real_t value, d1, d2;
};

template <bool ReturnCoefficients> struct LUTInterpolation {
    Spline012 A, Z;
};

template <> struct LUTInterpolation<true> {
    Spline012 A, Z;
    CubicCoeff A_coeffs, Z_coeffs;
};

DEVICE INLINE CubicCoeff spline_coefficients(const real_t left, const real_t left_M, const real_t right,
                                             const real_t right_M, const real_t drho2)
{
    return {left, right - left - drho2 * (REAL_LITERAL(2.0) * left_M + right_M) / REAL_LITERAL(6.0),
            drho2 * left_M / REAL_LITERAL(2.0), drho2 * (right_M - left_M) / REAL_LITERAL(6.0)};
}

DEVICE INLINE Spline012 evaluate_spline(const CubicCoeff &c, const real_t u, const real_t inv_drho)
{
    const real_t d1_u = c.b + u * (REAL_LITERAL(2.0) * c.c + REAL_LITERAL(3.0) * c.d * u);
    return {c.a + u * (c.b + u * (c.c + u * c.d)), d1_u * inv_drho,
            (REAL_LITERAL(2.0) * c.c + REAL_LITERAL(6.0) * c.d * u) * inv_drho * inv_drho};
}

// Node layout: A, M_A, Z, M_Z. Derivatives are with respect to rho, not u.
template <bool ReturnCoefficients = false>
DEVICE INLINE LUTInterpolation<ReturnCoefficients>
interpolate_lut_interval(const real_t left_A, const real_t left_M_A, const real_t left_Z, const real_t left_M_Z,
                         const real_t right_A, const real_t right_M_A, const real_t right_Z, const real_t right_M_Z,
                         const real_t u)
{
    constexpr real_t radius = static_cast<real_t>(magnet_number) / REAL_LITERAL(2.0);
    constexpr real_t drho =
        (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * radius / static_cast<real_t>(kSplineNodes - 1);
    constexpr real_t inv_drho = REAL_LITERAL(1.0) / drho;
    constexpr real_t drho2 = drho * drho;
    const CubicCoeff A = spline_coefficients(left_A, left_M_A, right_A, right_M_A, drho2);
    const CubicCoeff Z = spline_coefficients(left_Z, left_M_Z, right_Z, right_M_Z, drho2);

    LUTInterpolation<ReturnCoefficients> result{evaluate_spline(A, u, inv_drho), evaluate_spline(Z, u, inv_drho)};
    if constexpr (ReturnCoefficients) {
        result.A_coeffs = A;
        result.Z_coeffs = Z;
    }
    return result;
}

struct LUTNode {
    real_t A;
    real_t M_A;
    real_t Z;
    real_t M_Z;
};
static_assert(sizeof(LUTNode) == 4 * sizeof(real_t));

struct NodeCache {
    LUTNode left[kFieldFourierModeCount];
    LUTNode right[kFieldFourierModeCount];
};

struct CoeffCache {
    CubicCoeff A[kFieldFourierModeCount];
    CubicCoeff Z[kFieldFourierModeCount];
};

struct FieldCache {
    int interval_eff = -1;
    union {
        NodeCache nodes;
        CoeffCache coeffs;
    };
};

// MARK: H&F calc



DEVICE NOINLINE FieldResult eval_mag_field(const Vec2 &pos, const real_t down_disk_angle, FieldCache *cache = nullptr,
                                           const LUTNode *LUTptr = nullptr)
// returns SCALED field and gradient
{
    Vec3 h = {0};
    real_t nabla_h[5] = {0};
    constexpr int R_magnet = magnet_number / 2;
    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) { // cache and LUTptr should be nullptr in this mode
        real_t s = REAL_LITERAL(1.0);

        for (int i = 0; i < magnet_number; i++) {
            const Vec2 dk =
                pos - make_Vec2(R_magnet * real_cos(static_cast<real_t>(i) * magnet_phi0 + down_disk_angle),
                                R_magnet * real_sin(static_cast<real_t>(i) * magnet_phi0 + down_disk_angle));
            const real_t rhoksq = dk * dk;
            const real_t qk = rhoksq + height * height;
            const real_t invqk = REAL_LITERAL(1.0) / (qk);
            const real_t sqrtinvqk = real_sqrt(invqk);
            h = h + s * invqk * invqk * sqrtinvqk *
                        make_Vec3(REAL_LITERAL(3.0) * height * dk.x, REAL_LITERAL(3.0) * height * dk.y,
                                  REAL_LITERAL(2.0) * height * height - rhoksq);
            const real_t overall = REAL_LITERAL(3.0) * s * invqk * invqk * invqk * sqrtinvqk;
            nabla_h[0] += overall * height * (qk - REAL_LITERAL(5.0) * dk.x * dk.x);
            nabla_h[1] -= overall * REAL_LITERAL(5.0) * height * dk.x * dk.y;
            nabla_h[2] += overall * dk.x * (rhoksq - REAL_LITERAL(4.0) * height * height);
            nabla_h[3] += overall * height * (qk - REAL_LITERAL(5.0) * dk.y * dk.y);
            nabla_h[4] += overall * dk.y * (rhoksq - REAL_LITERAL(4.0) * height * height);
            s *= -1;
        }
        TracelessSymTensor Grad;
        h = h * Beta;
        for (int i = 0; i < 5; i++) {
            nabla_h[i] *= Beta;
        }
        for (int i = 0; i < 5; i++)
            Grad.component[i] = nabla_h[i];
        return {h, Grad};
    } else if constexpr (kFieldEvalMode == FieldEvalMode::kFourierSpline) {
        constexpr real_t rho_min = kLUTLowerBoundRatio * R_magnet;
        constexpr real_t drho = (kLUTUpperBoundRatio - kLUTLowerBoundRatio) * R_magnet / (kSplineNodes - 1);
        constexpr real_t inv_drho = REAL_LITERAL(1.0) / drho;
        constexpr real_t drho2 = drho * drho;
        const real_t rho = real_sqrt(pos * pos);
        const real_t phi = real_atan2(pos.y, pos.x);
        const real_t node_coordinate = (rho - rho_min) * inv_drho;
        const int left_node_index = static_cast<int>(real_floor(node_coordinate));
        const real_t u = node_coordinate - static_cast<real_t>(left_node_index);

        real_t h_rho = REAL_LITERAL(0.0);
        real_t h_phi = REAL_LITERAL(0.0);
        real_t h_z = REAL_LITERAL(0.0);
        real_t gradient_rho_rho = REAL_LITERAL(0.0);
        real_t gradient_rho_phi = REAL_LITERAL(0.0);
        real_t gradient_rho_z = REAL_LITERAL(0.0);
        real_t gradient_phi_phi = REAL_LITERAL(0.0);
        real_t gradient_phi_z = REAL_LITERAL(0.0);
        const real_t inv_rho = REAL_LITERAL(1.0) / rho;
        const real_t inv_rho2 = inv_rho * inv_rho;

        const LUTNode *left_node = nullptr;
        const LUTNode *right_node = nullptr;
        for (int i = 0; i < kFieldFourierModeCount; i++) {
            CubicCoeff local_A;
            CubicCoeff local_Z;
            CubicCoeff *coeffsA = &local_A;
            CubicCoeff *coeffsZ = &local_Z;
            if constexpr (kLUTCacheMode == LUTCacheMode::kCacheCoeffs) {
                coeffsA = &cache->coeffs.A[i];
                coeffsZ = &cache->coeffs.Z[i];
                if (cache->interval_eff == left_node_index)
                    goto CACHE_HIT_COEFFICIENTS;
                goto CACHE_MISS;
            } else if constexpr (kLUTCacheMode == LUTCacheMode::kCacheNode) {
                if (cache->interval_eff == left_node_index) {
                    left_node = &cache->nodes.left[i];
                    right_node = &cache->nodes.right[i];
                    goto CACHE_HIT_NODES;
                }
                goto CACHE_MISS;
            }
        CACHE_MISS:
            left_node = LUTptr + i * kSplineNodes + left_node_index;
            right_node = left_node + 1;
            if constexpr (kLUTCacheMode == LUTCacheMode::kCacheNode) {
                // 直接复制整个 LUTNode
                cache->nodes.left[i] = *left_node;
                cache->nodes.right[i] = *right_node;
            }
        CACHE_HIT_NODES:
            *coeffsA = spline_coefficients(left_node->A, left_node->M_A, right_node->A, right_node->M_A, drho2);
            *coeffsZ = spline_coefficients(left_node->Z, left_node->M_Z, right_node->Z, right_node->M_Z, drho2);
        CACHE_HIT_COEFFICIENTS:
            const Spline012 A = evaluate_spline(*coeffsA, u, inv_drho);
            const Spline012 Z = evaluate_spline(*coeffsZ, u, inv_drho);

            const int harmonic_int = (2 * i + 1) * magnet_number / 2;
            const real_t harmonic = static_cast<real_t>(harmonic_int);

            // The Fourier phase follows the magnet pattern, whereas the
            // cylindrical basis is fixed by the laboratory position.  Keep
            // these two angles separate when the lower disk is rotated.
            const real_t mode_angle = harmonic * (phi - down_disk_angle);
            const real_t mode_cos = real_cos(mode_angle);
            const real_t mode_sin = real_sin(mode_angle);

            // Phi_i = A_i(rho, z) cos(m_i (phi - disk_angle)); h = -grad Phi.
            h_rho -= A.d1 * mode_cos;
            h_phi += harmonic * A.value * inv_rho * mode_sin;
            h_z -= Z.value * mode_cos;

            // Independent components of grad(h) in the orthonormal
            // (e_rho, e_phi, e_z) basis.  Z = partial_z A and Z.d1 is
            // partial_rho partial_z A.
            gradient_rho_rho -= A.d2 * mode_cos;
            gradient_rho_phi += harmonic * (A.d1 * inv_rho - A.value * inv_rho2) * mode_sin;
            gradient_rho_z -= Z.d1 * mode_cos;
            gradient_phi_phi += (harmonic * harmonic * A.value * inv_rho2 - A.d1 * inv_rho) * mode_cos;
            gradient_phi_z += harmonic * Z.value * inv_rho * mode_sin;
        }
        if constexpr (kLUTCacheMode == LUTCacheMode::kCacheCoeffs || kLUTCacheMode == LUTCacheMode::kCacheNode) {
            cache->interval_eff = left_node_index;
        }

        // Rotate the vector and rank-2 tensor from the local cylindrical
        // basis into laboratory Cartesian coordinates.  This rotation uses
        // phi, not the relative Fourier phase phi - down_disk_angle.
        const real_t cos_phi = real_cos(phi);
        const real_t sin_phi = real_sin(phi);
        const real_t cos_phi2 = cos_phi * cos_phi;
        const real_t sin_phi2 = sin_phi * sin_phi;
        const real_t sin_cos_phi = sin_phi * cos_phi;
        h = make_Vec3(cos_phi * h_rho - sin_phi * h_phi, sin_phi * h_rho + cos_phi * h_phi, h_z);
        nabla_h[0] = cos_phi2 * gradient_rho_rho - REAL_LITERAL(2.0) * sin_cos_phi * gradient_rho_phi +
                     sin_phi2 * gradient_phi_phi;
        nabla_h[1] = sin_cos_phi * (gradient_rho_rho - gradient_phi_phi) + (cos_phi2 - sin_phi2) * gradient_rho_phi;
        nabla_h[2] = cos_phi * gradient_rho_z - sin_phi * gradient_phi_z;
        nabla_h[3] = sin_phi2 * gradient_rho_rho + REAL_LITERAL(2.0) * sin_cos_phi * gradient_rho_phi +
                     cos_phi2 * gradient_phi_phi;
        nabla_h[4] = sin_phi * gradient_rho_z + cos_phi * gradient_phi_z;
        TracelessSymTensor Grad;
        for (int i = 0; i < 5; i++)
            Grad.component[i] = nabla_h[i];
        h = h * Beta;
        for (int i = 0; i < 5; i++) {
            Grad.component[i] *= Beta;
        }
        return {h, Grad};
    }
}

DEVICE INLINE FieldForceResult eval_mag_field_and_force(const SolverParams &params, const Vec2 &pos, const Vec3 &q,
                                                        const real_t down_disk_angle, FieldCache *cache = nullptr,
                                                        const LUTNode *LUTptr = nullptr,
                                                        const FieldResult *H_cache = nullptr)
{
    auto [h, nabla_h] = (H_cache == nullptr) ? eval_mag_field(pos, down_disk_angle, cache, LUTptr) : *H_cache;

    constexpr real_t E_ = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    constexpr real_t D_ = (Mu_r - REAL_LITERAL(1.0)) / (Mu_r + REAL_LITERAL(2.0));
    const real_t A_ = -D_ * params.gamma_m * params.alpha * params.alpha * Eta;
    const real_t B_ = E_ * params.gamma_m * params.alpha * params.alpha * Eta;
    Vec3 v = A_ * h + B_ * q;
    Vec3 Force = make_Vec3((nabla_h.component[0] * v.x + nabla_h.component[1] * v.y + nabla_h.component[2] * v.z),
                           (nabla_h.component[1] * v.x + nabla_h.component[3] * v.y + nabla_h.component[4] * v.z),
                           (nabla_h.component[2] * v.x + nabla_h.component[4] * v.y +
                            (-nabla_h.component[0] - nabla_h.component[3]) * v.z));
    return {h, Force, nabla_h};
}

DEVICE INLINE auto eval_body_field_derivative(const State &in, const Vec3 &h, const TracelessSymTensor &grad)
{
    // The lower disk angular velocity is one in the dimensionless system.
    const auto relative_velocity = make_Vec2(in.speed.x + in.pos.y, in.speed.y - in.pos.x);
    const auto h_dot_world =
        make_Vec3(grad.component[0] * relative_velocity.x + grad.component[1] * relative_velocity.y - h.y,
                  grad.component[1] * relative_velocity.x + grad.component[3] * relative_velocity.y + h.x,
                  grad.component[2] * relative_velocity.x + grad.component[4] * relative_velocity.y);

    return rotate(conjugate(in.attitude), h_dot_world - cross(in.omega, h));
}


// MARK: RHS eval
template <bool ExportFM>
DEVICE NOINLINE bool eval_rhs(const State &in, State &out, const SolverParams &params, const bool &sliding,
                              const Vec2 &sliding_dir, FieldCache *cache = nullptr, const LUTNode *LUTptr = nullptr,
                              const FieldResult *H_cache = nullptr, Vec3 *Fext = nullptr, Vec3 *Mext = nullptr)
{
    // the caller should cope with selection by giving nullptr when magField cache is not valid
    const Vec3 q_world = vec_cast<real_t>(rotate(in.attitude, in.q));

    auto [h, F, grad] = eval_mag_field_and_force(params, in.pos, q_world, in.down_disk_angle, cache, LUTptr, H_cache);
    constexpr real_t E_ = (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
    const Vec3 M = -params.gamma_m * E_ * cross(q_world, h);
    if constexpr (ExportFM) {
        *Fext = F;
        *Mext = M;
    }
    const real_t N = params.gamma_g - F.z;
    if (N < REAL_LITERAL(0.0))
        return false;
    if (sliding) {
        const Vec2 f = -sliding_dir * Mu * N;
        out.speed = f + make_Vec2(F.x, F.y) - (params.delta * in.speed) * length(in.speed);
        out.omega = M + make_Vec3(f.y, -f.x, REAL_LITERAL(0.0)) / (Eta * params.alpha);
    } else {
        constexpr real_t inv_one_plus_eta = REAL_LITERAL(1.0) / (REAL_LITERAL(1.0) + Eta);
        const real_t inv_alpha_one_plus_eta = inv_one_plus_eta / params.alpha;
        const real_t speed_norm = length(in.speed);

        const real_t drag_factor = params.delta * speed_norm;

        const Vec3 totalF = make_Vec3(F.x - drag_factor * in.speed.x, F.y - drag_factor * in.speed.y, F.z);
        // Pure-rolling angular acceleration.
        out.omega = make_Vec3(Eta * inv_one_plus_eta * M.x - inv_alpha_one_plus_eta * totalF.y,
                              Eta * inv_one_plus_eta * M.y + inv_alpha_one_plus_eta * totalF.x, M.z);
        // Recover translational acceleration from rolling constraint.
        out.speed = make_Vec2(params.alpha * out.omega.y, -params.alpha * out.omega.x);
    }

    const auto h_ext_dot = eval_body_field_derivative(in, h, grad);

    Vec3 h0_dot = h_ext_dot * (REAL_LITERAL(3.0) / (Mu_r + REAL_LITERAL(2.0)));
    // assume ReLU use the 0 branch
    real_t interme = Alpha_JA * (REAL_LITERAL(1.0) - C_JA * Xi_JA) / Xi_JA - E_;
    Vec3 q_dot = (REAL_LITERAL(1.0) / interme) * h0_dot;
    // IMPORTANT: self-consistent H_e_dot
    Vec3 he_dot = (Alpha_JA / (Xi_JA * interme)) * h0_dot;

    // if reject we should select ReLU x branch
    if (in.q * he_dot > REAL_LITERAL(0.0)) {
        constexpr real_t c = Xi_JA / Alpha_JA;
        const real_t d = -length(in.q) / K;
        constexpr real_t a = REAL_LITERAL(1.0) - C_JA * Xi_JA - E_ * c;
        const real_t b = -Alpha_JA * (REAL_LITERAL(1.0) - C_JA) / K * length(in.q) - E_ * d;
        const real_t a_final = c / a;
        const real_t b_final = d / a - b * (c + d) / a / (a + b);
        const Vec3 qnorm = in.q * (REAL_LITERAL(1.0) / length(in.q));
        q_dot = a_final * h0_dot + b_final * (qnorm * h0_dot) * qnorm;
    }
    out.q = q_dot;
    out.pos = in.speed;
    out.attitude = REAL_LITERAL(0.5) * make_Quaternion(REAL_LITERAL(0.0), in.omega) * in.attitude;
    out.down_disk_angle = REAL_LITERAL(1.0);
    return true;
}


// MARK: Try 1 step

enum class AdvanceStepResult {
    kStepOk,
    kStepBallFly,
    kStepFinerRetry,
    kStepNumericalUnstable,
    kStepNaNOrInf,
    kStepSolverSelfContradict,
    kStepLUTOutOfDomain,
};

// RK2 is kept inline because additional checks are performed between stages.

// Remove roundoff drift from the sticking constraint while preserving the
// contact impulse balance used at a sliding-to-sticking event.
DEVICE INLINE void project_pure_rolling(State &in, const SolverParams &params)
{
    const accumulator_t eta = static_cast<accumulator_t>(Eta);
    const accumulator_t alpha = static_cast<accumulator_t>(params.alpha);
    const accumulator_t denominator = static_cast<accumulator_t>(REAL_LITERAL(1.0)) + eta;
    const accumulator_t speed_x = static_cast<accumulator_t>(in.speed.x);
    const accumulator_t speed_y = static_cast<accumulator_t>(in.speed.y);
    const accumulator_t omega_x = static_cast<accumulator_t>(in.omega.x);
    const accumulator_t omega_y = static_cast<accumulator_t>(in.omega.y);
    const accumulator_t omega_z = static_cast<accumulator_t>(in.omega.z);

    const StateVec2 vf = make_Vec2(static_cast<accumulator_t>((speed_x + eta * alpha * omega_y) / denominator),
                                   static_cast<accumulator_t>((speed_y - eta * alpha * omega_x) / denominator));
    const StateVec3 omegaf = make_Vec3(static_cast<accumulator_t>((eta * omega_x - speed_y / alpha) / denominator),
                                       static_cast<accumulator_t>((eta * omega_y + speed_x / alpha) / denominator),
                                       static_cast<accumulator_t>(omega_z));
    in.speed = vf;
    in.omega = omegaf;
}

DEVICE INLINE bool normalize_attitude(State &in)
{
    const real_t squared_norm = static_cast<real_t>(norm_sq(in.attitude));
    if (!real_isfinite(squared_norm) || squared_norm <= REAL_LITERAL(0.0))
        return false;

    real_t scale;
    if constexpr (kAttitudeNormalizeMode == AttitudeNormalizeMode::kExact) {
        scale = REAL_LITERAL(1.0) / real_sqrt(squared_norm);
    } else {
        // One Newton step for 1 / sqrt(squared_norm), starting from 1.
        scale = REAL_LITERAL(0.5) * (REAL_LITERAL(3.0) - squared_norm);
    }
    if (!real_isfinite(scale) || scale <= REAL_LITERAL(0.0))
        return false;
    in.attitude = in.attitude * scale;
    return true;
}


DEVICE NOINLINE AdvanceStepResult try_advance_1_step(const real_t dt, const SolverParams &params, State &in,
                                                     bool &sliding, Vec2 &sliding_dir, bool &isHCacheValid,
                                                     FieldCache *cache = nullptr, const LUTNode *LUTptr = nullptr,
                                                     FieldResult *H_cache = nullptr)
{
    if (sliding) {
        Vec3 F_start;
        Vec3 M_start;
        State Statedot_start;
        if (isHCacheValid) {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(in, Statedot_start, params, sliding, sliding_dir, nullptr, nullptr, H_cache,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(in, Statedot_start, params, sliding, sliding_dir, cache, LUTptr, H_cache, &F_start,
                                    &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        } else {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(in, Statedot_start, params, sliding, sliding_dir, nullptr, nullptr, nullptr,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(in, Statedot_start, params, sliding, sliding_dir, cache, LUTptr, nullptr, &F_start,
                                    &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        }
        const Vec2 G = make_Vec2(F_start.x, F_start.y) - params.delta * length(in.speed) * in.speed;
        const Vec2 a = G + params.alpha * make_Vec2(-M_start.y, M_start.x);
        real_t A = length(a);
        real_t b = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_start.z);
        if (!real_isfinite(A) || !real_isfinite(b))
            return AdvanceStepResult::kStepNaNOrInf;
        if (A < b) {
            real_t D_c = (b - A) * (b + A);
            Vec2 s0 = vec_cast<real_t>(in.speed + params.alpha * make_Vec2(-in.omega.y, in.omega.x));
            real_t Q = b * length(s0) + a * s0;
            if (!real_isfinite(Q) || !real_isfinite(D_c) || !real_isfinite(dt * D_c))
                return AdvanceStepResult::kStepNaNOrInf;
            if (Q <= dt * D_c) {
                real_t tc = Q / D_c;
                if (!real_isfinite(tc) || tc <= REAL_LITERAL(0.0) || tc > dt)
                    return AdvanceStepResult::kStepNumericalUnstable;
                State state_at_halftc = in + tc * REAL_LITERAL(0.5) * Statedot_start;
                State dot_estimate;
                Vec2 s_mid =
                    vec_cast<real_t>(state_at_halftc.speed +
                                     params.alpha * make_Vec2(-state_at_halftc.omega.y, state_at_halftc.omega.x));
                const real_t s_mid_norm = length(s_mid);
                if (!real_isfinite(s_mid_norm))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (s_mid_norm == REAL_LITERAL(0.0))
                    return AdvanceStepResult::kStepFinerRetry;
                Vec2 sliddir_mid = s_mid / s_mid_norm;
                if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                    if (!eval_rhs<false>(state_at_halftc, dot_estimate, params, sliding, sliddir_mid, nullptr, nullptr,
                                         nullptr))
                        return AdvanceStepResult::kStepBallFly;
                } else {
                    if (!eval_rhs<false>(state_at_halftc, dot_estimate, params, sliding, sliddir_mid, cache, LUTptr,
                                         nullptr))
                        return AdvanceStepResult::kStepBallFly;
                }
                State state_at_tc = in + tc * dot_estimate;
                project_pure_rolling(state_at_tc, params);
                if (!normalize_attitude(state_at_tc))
                    return AdvanceStepResult::kStepNumericalUnstable;
                const Vec3 q_world = vec_cast<real_t>(rotate(state_at_tc.attitude, state_at_tc.q));
                auto [h, F, grad] = eval_mag_field_and_force(params, state_at_tc.pos, q_world,
                                                             state_at_tc.down_disk_angle, cache, LUTptr, nullptr);
                constexpr real_t E_ =
                    (REAL_LITERAL(1.0) - C_JA) / (REAL_LITERAL(1.0) - Xi_JA) / (REAL_LITERAL(2.0) + Mu_r);
                const Vec3 M = -params.gamma_m * E_ * cross(q_world, h);
                const real_t N = params.gamma_g - F.z;
                if (!real_isfinite(N) || !real_isfinite(F.x) || !real_isfinite(F.y) || !real_isfinite(M.x) ||
                    !real_isfinite(M.y))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (N < REAL_LITERAL(0.0))
                    return AdvanceStepResult::kStepBallFly;
                const Vec2 G_ = make_Vec2(F.x, F.y) - params.delta * length(state_at_tc.speed) * state_at_tc.speed;
                const Vec2 a_ = G_ + params.alpha * make_Vec2(-M.y, M.x);
                real_t A_ = length(a_);
                real_t b_ = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N;
                if (!real_isfinite(A_) || !real_isfinite(b_))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (A_ > b_)
                    return AdvanceStepResult::kStepFinerRetry;

                const real_t remaining = dt - tc;
                State final_state = state_at_tc;
                FieldResult final_field = {h, grad};
                if (remaining > REAL_LITERAL(0.0)) {
                    const bool ksliding = false;
                    const Vec2 no_sliding_dir = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
                    State dot_rolling_start;
                    State dot_rolling_mid;
                    Vec3 F_rolling_mid;
                    Vec3 M_rolling_mid;
                    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                        if (!eval_rhs<false>(state_at_tc, dot_rolling_start, params, ksliding, no_sliding_dir, nullptr,
                                             nullptr, &final_field))
                            return AdvanceStepResult::kStepBallFly;
                    } else {
                        if (!eval_rhs<false>(state_at_tc, dot_rolling_start, params, ksliding, no_sliding_dir, cache,
                                             LUTptr, &final_field))
                            return AdvanceStepResult::kStepBallFly;
                    }
                    const State rolling_mid = state_at_tc + remaining * REAL_LITERAL(0.5) * dot_rolling_start;
                    if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                        if (!eval_rhs<true>(rolling_mid, dot_rolling_mid, params, ksliding, no_sliding_dir, nullptr,
                                            nullptr, nullptr, &F_rolling_mid, &M_rolling_mid))
                            return AdvanceStepResult::kStepBallFly;
                    } else {
                        if (!eval_rhs<true>(rolling_mid, dot_rolling_mid, params, ksliding, no_sliding_dir, cache,
                                            LUTptr, nullptr, &F_rolling_mid, &M_rolling_mid))
                            return AdvanceStepResult::kStepBallFly;
                    }
                    const real_t N_mid = params.gamma_g - F_rolling_mid.z;
                    const Vec2 G_mid = make_Vec2(F_rolling_mid.x, F_rolling_mid.y) -
                                       params.delta * length(rolling_mid.speed) * rolling_mid.speed;
                    const Vec2 a_mid = G_mid + params.alpha * make_Vec2(-M_rolling_mid.y, M_rolling_mid.x);
                    const real_t A_mid = length(a_mid);
                    const real_t b_mid = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N_mid;
                    if (!real_isfinite(A_mid) || !real_isfinite(b_mid))
                        return AdvanceStepResult::kStepNaNOrInf;
                    if (A_mid > b_mid)
                        return AdvanceStepResult::kStepFinerRetry;

                    final_state = state_at_tc + remaining * dot_rolling_mid;
                    project_pure_rolling(final_state, params);
                    if (!normalize_attitude(final_state))
                        return AdvanceStepResult::kStepNumericalUnstable;
                    const Vec3 q_final = vec_cast<real_t>(rotate(final_state.attitude, final_state.q));
                    const auto [h_final, F_final, grad_final] = eval_mag_field_and_force(
                        params, final_state.pos, q_final, final_state.down_disk_angle, cache, LUTptr, nullptr);
                    const Vec3 M_final = -params.gamma_m * E_ * cross(q_final, h_final);
                    const real_t N_final = params.gamma_g - F_final.z;
                    if (!real_isfinite(N_final) || !real_isfinite(F_final.x) || !real_isfinite(F_final.y) ||
                        !real_isfinite(M_final.x) || !real_isfinite(M_final.y))
                        return AdvanceStepResult::kStepNaNOrInf;
                    if (N_final < REAL_LITERAL(0.0))
                        return AdvanceStepResult::kStepBallFly;
                    const Vec2 G_final =
                        make_Vec2(F_final.x, F_final.y) - params.delta * length(final_state.speed) * final_state.speed;
                    const Vec2 a_final = G_final + params.alpha * make_Vec2(-M_final.y, M_final.x);
                    const real_t A_final = length(a_final);
                    const real_t b_final = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * N_final;
                    if (!real_isfinite(A_final) || !real_isfinite(b_final))
                        return AdvanceStepResult::kStepNaNOrInf;
                    if (A_final > b_final)
                        return AdvanceStepResult::kStepFinerRetry;
                    final_field = {h_final, grad_final};
                }

                if (H_cache != nullptr)
                    *H_cache = final_field;
                in = final_state;
                sliding = false;
                sliding_dir = make_Vec2(REAL_LITERAL(0.0), REAL_LITERAL(0.0));
                isHCacheValid = H_cache != nullptr;
                return AdvanceStepResult::kStepOk;
            } else {
                goto NO_PREDICTED_TRANSIT;
            }
        } else {
        NO_PREDICTED_TRANSIT:
            State rk2midpoint = dt * REAL_LITERAL(0.5) * Statedot_start + in;
            Vec2 s_mid = vec_cast<real_t>(rk2midpoint.speed +
                                          params.alpha * make_Vec2(-rk2midpoint.omega.y, rk2midpoint.omega.x));
            const real_t s_mid_norm = length(s_mid);
            if (!real_isfinite(s_mid_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if (s_mid_norm == REAL_LITERAL(0.0))
                return AdvanceStepResult::kStepFinerRetry;
            Vec2 sliddir_mid = s_mid / s_mid_norm;
            Vec3 F_mid;
            Vec3 M_mid;
            State Statedot_mid;
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(rk2midpoint, Statedot_mid, params, sliding, sliddir_mid, nullptr, nullptr, nullptr,
                                    &F_mid, &M_mid))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(rk2midpoint, Statedot_mid, params, sliding, sliddir_mid, cache, LUTptr, nullptr,
                                    &F_mid, &M_mid))
                    return AdvanceStepResult::kStepBallFly;
            }
            const Vec2 Gmid =
                make_Vec2(F_mid.x, F_mid.y) - params.delta * length(rk2midpoint.speed) * rk2midpoint.speed;
            const Vec2 amid = Gmid + params.alpha * make_Vec2(-M_mid.y, M_mid.x);
            real_t Amid = length(amid);
            real_t bmid = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_mid.z);
            if (!real_isfinite(Amid) || !real_isfinite(bmid))
                return AdvanceStepResult::kStepNaNOrInf;
            if (Amid < bmid) {
                real_t D_cmid = (bmid - Amid) * (bmid + Amid);
                real_t Qmid = bmid * length(s_mid) + amid * s_mid;
                if (!real_isfinite(Qmid) || !real_isfinite(D_cmid) || !real_isfinite(dt * REAL_LITERAL(0.5) * D_cmid))
                    return AdvanceStepResult::kStepNaNOrInf;
                if (Qmid <= dt * REAL_LITERAL(0.5) * D_cmid) {
                    return AdvanceStepResult::kStepFinerRetry;
                }
            }
            State final_state = in + dt * Statedot_mid;
            if (!normalize_attitude(final_state))
                return AdvanceStepResult::kStepNumericalUnstable;
            Vec2 sf = vec_cast<real_t>(final_state.speed +
                                       params.alpha * make_Vec2(-final_state.omega.y, final_state.omega.x));
            const real_t sf_norm = length(sf);
            if (!real_isfinite(sf_norm))
                return AdvanceStepResult::kStepNaNOrInf;
            if (sf_norm == REAL_LITERAL(0.0))
                return AdvanceStepResult::kStepFinerRetry;
            const Vec2 final_dir = sf / sf_norm;
            in = final_state;
            sliding_dir = final_dir;
            isHCacheValid = false;
            return AdvanceStepResult::kStepOk;
        }
    } else {
        State start = in;
        project_pure_rolling(start, params);
        Vec3 F_start;
        Vec3 M_start;
        State Statedot_start;
        if (isHCacheValid) {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(start, Statedot_start, params, sliding, sliding_dir, nullptr, nullptr, H_cache,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(start, Statedot_start, params, sliding, sliding_dir, cache, LUTptr, H_cache,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        } else {
            if constexpr (kFieldEvalMode == FieldEvalMode::kExact) {
                if (!eval_rhs<true>(start, Statedot_start, params, sliding, sliding_dir, nullptr, nullptr, nullptr,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            } else {
                if (!eval_rhs<true>(start, Statedot_start, params, sliding, sliding_dir, cache, LUTptr, nullptr,
                                    &F_start, &M_start))
                    return AdvanceStepResult::kStepBallFly;
            }
        }
        const Vec2 G = make_Vec2(F_start.x, F_start.y) - params.delta * length(start.speed) * start.speed;
        const Vec2 a = G + params.alpha * make_Vec2(-M_start.y, M_start.x);
        real_t A = length(a);
        real_t b = (REAL_LITERAL(1.0) + REAL_LITERAL(1.0) / Eta) * Mu * (params.gamma_g - F_start.z);
        if (!real_isfinite(A) || !real_isfinite(b))
            return AdvanceStepResult::kStepNaNOrInf;
        if (A > b)
            return AdvanceStepResult::kStepSolverSelfContradict;
    }
}
