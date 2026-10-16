// Isolated fitting harness: compile against the audit snapshot, never production.
// LUT CSV sigma alpha_JA c_JA K r_mm vt_mm_s vr_mm_s wz_rad_s hold_turns phase seconds dt mu_r qz eddy_table
#include "2dmodel-hysteresis.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#ifdef IYPT16_RADIAL_DIAGNOSTICS
#include "radial_overshoot_diagnostics.hpp"
#endif
int main(int argc,char** argv) {
    if(argc!=18) return 2;
    Mu_r=std::stod(argv[15]);
    std::ifstream eddy(argv[17],std::ios::binary);
    eddy.read(reinterpret_cast<char*>(eddy_coefficients::aj),12*sizeof(double));
    eddy.read(reinterpret_cast<char*>(eddy_coefficients::inv_tau_ref_star),12*sizeof(double));
    if(!eddy)return 5;
    const double sigma=std::stod(argv[3]);
    Alpha_JA=std::stod(argv[4]); C_JA=std::stod(argv[5]); K=std::stod(argv[6]);
    Xi_JA=1.-1./(1.+Alpha_JA*(Mu_r-1.));
    const double r=std::stod(argv[7]),vt=std::stod(argv[8]),vr=std::stod(argv[9]);
    const double wz=std::stod(argv[10]),turns=std::stod(argv[11]),phase=std::stod(argv[12]);
    const double seconds=std::stod(argv[13]),dt=std::stod(argv[14]);
    const double O=2*hysteresis_physics::omega_ref,vu=17.5*O;
    std::vector<LUTNode> lut(kFieldFourierModeCount*kSplineNodes);
    std::ifstream input(argv[1],std::ios::binary);
    input.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(LUTNode));if(!input)return 3;
    auto p=make_solver_params(magnet_number,1.,2.,Eta);
    p.eddy_current_scale*=sigma/eddy_coefficients::sigma_si;
    State<kEddyCurrentModeCount> state{};state.pos.x=r/17.5;state.attitude.real=1;
    const int held_steps=static_cast<int>(std::round(turns*real_two_pi()/dt));
    const double held_start=phase-held_steps*dt;
    FieldCache cache{};const Vec2 zero{};
    for(int i=0;i<held_steps;++i) {
        state.down_disk_angle=held_start+i*dt;
        State<kEddyCurrentModeCount> first{},second{},mid=state;Vec3 h1{},h2{};
        if(!eval_rhs<false>(state,first,h1,p,false,zero,&cache,lut.data()))return 4;
        mid.q=state.q+first.q*(dt*.5);
        advance_eddy_current_state(state,mid,dt*.5,h1,p);
        mid.down_disk_angle=held_start+(i+.5)*dt;
        if(!eval_rhs<false>(mid,second,h2,p,false,zero,&cache,lut.data()))return 4;
        advance_eddy_current_state(state,state,dt,h2,p);state.q=state.q+second.q*dt;
    }
    state.down_disk_angle=phase;
    state.q.z+=std::stod(argv[16]);
    state.speed={vr/vu,vt/vu};
    state.omega={-state.speed.y/p.alpha,state.speed.x/p.alpha,wz/O};
    bool sliding=false,valid=false;Vec2 direction{};cache=FieldCache{};FieldResult field{};
    std::ofstream out(argv[2]);out<<std::setprecision(17);
    out<<"time_s,speed_mm_s,vt_mm_s,vr_mm_s,radius_mm,wz_rad_s,phi_rad,q_norm,sliding\n";
#ifdef IYPT16_RADIAL_DIAGNOSTICS
    RadialDiagnostics radial_diagnostics(std::string(argv[2])+".radial.csv",p,lut.data());
#endif
    const int steps=static_cast<int>(std::round(seconds*O/dt));
    const int sample=std::max(1,static_cast<int>(std::round(.1/dt)));
    int code=0;double end=0;
    for(int i=0;i<=steps;++i) {
        end=i*dt/O;
        if(i%sample==0||i==steps) {
#ifdef IYPT16_RADIAL_DIAGNOSTICS
            radial_diagnostics.write(end,state,sliding);
#endif
            const double rr=length(state.pos);
            const double tv=(state.pos.x*state.speed.y-state.pos.y*state.speed.x)/rr;
            const double rv=(state.pos*state.speed)/rr;
            out<<end<<','<<length(state.speed)*vu<<','<<tv*vu<<','<<rv*vu<<','<<rr*17.5<<','<<state.omega.z*O<<','<<std::atan2(state.pos.y,state.pos.x)<<','<<length(state.q)<<','<<sliding<<'\n';
        }
        if(i==steps)break;
        state.down_disk_angle=phase+i*dt;
        auto result=fwd1step(dt,p,state,sliding,direction,valid,&cache,lut.data(),&field);
        if(result!=AdvanceStepResult::kStepOk){code=static_cast<int>(result)+1;break;}
    }
    double values[51];write_state_values<true>(state,values);
    std::ofstream final(std::string(argv[2])+".state",std::ios::binary);
    final.write(reinterpret_cast<const char*>(values),sizeof(values));
    std::cout<<"end_s="<<end<<" status="<<code<<'\n';
    return 0;
}
