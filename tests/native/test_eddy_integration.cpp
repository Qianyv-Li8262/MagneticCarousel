// SPDX-License-Identifier: AGPL-3.0-only
// Compile with IYPT16_ENABLE_EDDY_CURRENTS=0 and =1; supports C++ and CUDA.
#include <cstdio>
#include "2dmodel-hysteresis.hpp"
#include <iostream>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

DEVICE INLINE bool close_vector(const Vec3 &a, const Vec3 &b)
{
#ifdef USE_FP64
    constexpr real_t tolerance = REAL_LITERAL(2e-12);
#else
    constexpr real_t tolerance = REAL_LITERAL(2e-5);
#endif
    return length(a-b) <= tolerance*(REAL_LITERAL(1.0)+length(b));
}

DEVICE int check_integration()
{
    constexpr int full_size = 15+12*kEddyCurrentModeCount;
    real_t values[full_size]{};
    values[1]=REAL_LITERAL(4.0); values[5]=REAL_LITERAL(1.0);
    values[12]=REAL_LITERAL(0.00001);
    for (int j=15; j<full_size; ++j) values[j]=static_cast<real_t>(j-30)*REAL_LITERAL(0.00001);
    const auto state=read_state_values<true>(values);
    real_t roundtrip[full_size]{};
    write_state_values<true>(state,roundtrip);
    for (int j=0; j<full_size; ++j) if (roundtrip[j]!=values[j]) return 1;
    if (kSerializedStateSize!=(kEnableEddyCurrents?full_size:15)) return 2;

    SolverParams params{REAL_LITERAL(0.3),REAL_LITERAL(0.0),REAL_LITERAL(10.0),
                        REAL_LITERAL(0.072),REAL_LITERAL(2.0),REAL_LITERAL(1.0)};
    const Vec3 qworld=vec_cast<real_t>(rotate(state.attitude,state.q));
    const auto legacy=eval_mag_field_and_force(params,state.pos,qworld,state.down_disk_angle);
    const auto loaded=eval_mag_field_and_force(params,state);
    Vec3 expected_force=legacy.Force;
    Vec3 expected_torque=legacy.magnetic_angular_acceleration;
    if constexpr (kEnableEddyCurrents) {
        const Vec3 dm=eval_eddy_current_moment(state.yx,state.yy,state.yz);
        const auto &g=legacy.hGrad3d.component;
        const real_t scale=params.gamma_m*params.alpha*params.alpha*Eta/REAL_LITERAL(3.0);
        expected_force=expected_force+scale*make_Vec3(g[0]*dm.x+g[1]*dm.y+g[2]*dm.z,
            g[1]*dm.x+g[3]*dm.y+g[4]*dm.z,g[2]*dm.x+g[4]*dm.y-(g[0]+g[3])*dm.z);
        expected_torque=expected_torque+(params.gamma_m/REAL_LITERAL(3.0))*cross(dm,legacy.h);
    }
    if (!close_vector(loaded.Force,expected_force) ||
        !close_vector(loaded.magnetic_angular_acceleration,expected_torque)) return 3;
    auto derivative=state;
    Vec3 hb_dot;
    if (!eval_rhs<false>(state,derivative,hb_dot,params,false,Vec2{})) return 4;
    write_state_values<true>(derivative,roundtrip);
    for (int j=15; j<full_size; ++j) if (roundtrip[j]!=REAL_LITERAL(0.0)) return 5;

    // Check both ordinary stage paths against independently assembled stages.
    params={REAL_LITERAL(1.0),REAL_LITERAL(0.0),REAL_LITERAL(10.0),
            REAL_LITERAL(0.0),REAL_LITERAL(0.0),REAL_LITERAL(1.0)};
    for (int sliding_case=0; sliding_case<2; ++sliding_case) {
        auto start=state;
        start.speed={REAL_LITERAL(0.0),REAL_LITERAL(0.3)};
        start.omega={sliding_case?REAL_LITERAL(0.05):REAL_LITERAL(-0.3),REAL_LITERAL(0.0),REAL_LITERAL(0.0)};
        if (!sliding_case) project_pure_rolling(start,params);
        const real_t dt=REAL_LITERAL(0.00001);
        const Vec2 direction=sliding_case?make_Vec2(REAL_LITERAL(0.0),REAL_LITERAL(1.0)):Vec2{};
        State<kEddyCurrentModeCount> d0{},dm{};
        Vec3 h0,hm;
        if (!eval_rhs<false>(start,d0,h0,params,sliding_case,direction)) return 6;
        auto mid=start+dt*REAL_LITERAL(0.5)*d0;
        if constexpr (kEnableEddyCurrents)
            advance_eddy_current(mid.yx,mid.yy,mid.yz,dt*REAL_LITERAL(0.5),h0,params.eddy_current_scale);
        if (!eval_rhs<false>(mid,dm,hm,params,sliding_case,direction)) return 7;
        auto expected=start;
        if constexpr (kEnableEddyCurrents)
            advance_eddy_current(expected.yx,expected.yy,expected.yz,dt,hm,params.eddy_current_scale);
        auto actual=start;
        bool sliding=sliding_case,valid=false;
        Vec2 actual_direction=direction;
        if (fwd1step(dt,params,actual,sliding,actual_direction,valid)!=AdvanceStepResult::kStepOk) return 8;
        real_t ev[full_size],av[full_size];
        write_state_values<true>(expected,ev); write_state_values<true>(actual,av);
        for (int j=15; j<full_size; ++j)
            if (real_fabs(ev[j]-av[j])>REAL_LITERAL(1e-10)) return 9;
    }

    // A sliding capture must evolve the history before AND after the event.
    auto captured=state;
    captured.speed={REAL_LITERAL(0.0),REAL_LITERAL(0.001)};
    captured.omega={};
    auto fine=captured;
    bool sliding=true,valid=false;
    Vec2 direction{REAL_LITERAL(0.0),REAL_LITERAL(1.0)};
    if (fwd1step(REAL_LITERAL(0.001),params,captured,sliding,direction,valid)!=AdvanceStepResult::kStepOk) return 10;
    if (sliding) return 11;
    bool fine_sliding=true,fine_valid=false;
    Vec2 fine_direction{REAL_LITERAL(0.0),REAL_LITERAL(1.0)};
    for (int i=0; i<100; ++i)
        if (fwd1step(REAL_LITERAL(0.00001),params,fine,fine_sliding,fine_direction,fine_valid)!=AdvanceStepResult::kStepOk) return 12;
    real_t cv[full_size],fv[full_size];
    write_state_values<true>(captured,cv); write_state_values<true>(fine,fv);
    for (int j=15; j<full_size; ++j)
        if (real_fabs(cv[j]-fv[j])>REAL_LITERAL(1e-7)) return 13;

    // A rejected interval must not commit any part of the mode state.
    auto rejected=state;
    params.gamma_g=REAL_LITERAL(-1.0);
    sliding=true;valid=false;direction={REAL_LITERAL(0.0),REAL_LITERAL(1.0)};
    if (fwd1step(REAL_LITERAL(0.001),params,rejected,sliding,direction,valid)==AdvanceStepResult::kStepOk) return 14;
    write_state_values<true>(rejected,roundtrip);
    for (int j=0; j<full_size; ++j) if (roundtrip[j]!=values[j]) return 15;
    return 0;
}

#ifdef __CUDACC__
__global__ void integration_kernel(int *result) { *result=check_integration(); }
#endif
int main()
{
    int result;
#ifdef __CUDACC__
    int *device;
    if (cudaMalloc(&device,sizeof(int))!=cudaSuccess) return 90;
    integration_kernel<<<1,1>>>(device);
    if (cudaGetLastError()!=cudaSuccess || cudaMemcpy(&result,device,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess) return 91;
    cudaFree(device);
#else
    result=check_integration();
#endif
    if (result) std::cerr << "Eddy integration check failed: " << result << '\n';
    else std::cout << "PASS: loads, stage updates, capture, state roundtrip and rollback\n";
    return result;
}
