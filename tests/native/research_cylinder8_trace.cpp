// Selected-point diagnostic only; does not change the production integrator.
#include "2dmodel-hysteresis.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    const double sr=std::atof(argv[1]), sw=std::atof(argv[2]);
    const double dt=std::atof(argv[3]), horizon=std::atof(argv[4]);
    std::vector<LUTNode> lut(kFieldFourierModeCount*kSplineNodes);
    if constexpr (kFieldEvalMode==FieldEvalMode::kFourierSpline) {
        std::ifstream input(argv[5],std::ios::binary|std::ios::ate);
        if (!input || input.tellg()!=static_cast<std::streamoff>(lut.size()*sizeof(LUTNode))) return 3;
        input.seekg(0);input.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(LUTNode));
        if (!input) return 4;
    }
    // Match production parameter arithmetic as well as its mathematical values.
    const double inv_sr=1./sr,inv_sw2=1./(sw*sw);
    const SolverParams p{.33333333*sr,1.7212357e-4*inv_sr,
        2.64965696*inv_sw2,7830.34224*inv_sw2,176182.704*inv_sr*inv_sr*inv_sw2};
    State s{};s.pos.x=4.;s.attitude.real=1.;
    bool sliding=false,valid=false;Vec2 direction{};FieldCache cache{};FieldResult field{};
    std::ofstream output(argv[6]);output<<std::setprecision(17);
    output<<"time,radius,radial_v,v,phase,q_norm,slip,sliding,normal_force,magnetic_radial_force,status\n";
    auto record=[&](int step,int status) {
        const auto pos=vec_cast<real_t>(s.pos), vel=vec_cast<real_t>(s.speed);
        double r=length(pos),crossp=pos.x*vel.y-pos.y*vel.x;
        const Vec3 q=vec_cast<real_t>(rotate(s.attitude,s.q));
        const auto f=eval_mag_field_and_force(p,pos,q,s.down_disk_angle,&cache,lut.data());
        const auto slip=vel+p.alpha*make_Vec2(-static_cast<real_t>(s.omega.y),static_cast<real_t>(s.omega.x));
        output<<step*dt<<','<<r<<','<<(pos.x*vel.x+pos.y*vel.y)/r<<','<<crossp/(r*4.)<<','
            <<1.-crossp/(r*r)-(pos.x*s.omega.x+pos.y*s.omega.y)/(r*4.)<<','<<length(q)<<','
            <<length(slip)/4.<<','<<sliding<<','<<p.gamma_g-f.Force.z<<','
            <<(pos.x*f.Force.x+pos.y*f.Force.y)/r<<','<<status<<'\n';
    };
    record(0,0);int steps=std::llround(horizon/dt),stride=std::max(1,int(std::llround(.1/dt)));
    for (int step=0;step<steps;++step) {
        const auto result=fwd1step(dt,p,s,sliding,direction,valid,&cache,lut.data(),&field);
        if (result!=AdvanceStepResult::kStepOk) {
            record(step,static_cast<int>(result)+1);
            std::cout<<"status="<<static_cast<int>(result)+1<<" last_t="<<step*dt<<" radius="<<length(s.pos)<<'\n';
            return 0;
        }
        if ((step+1)%stride==0)record(step+1,0);
    }
    std::cout<<"status=0 last_t="<<horizon<<" radius="<<length(s.pos)<<'\n';
}
