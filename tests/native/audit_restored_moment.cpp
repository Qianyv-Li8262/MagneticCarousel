// Isolated field-amplitude/conductivity comparisons; production files untouched.
// Args: LUT CSV beta sigma duration [initial_state_51_doubles]
#include "2dmodel-hysteresis.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char **argv){
    if(argc<6)return 2;
    std::vector<LUTNode> lut(kFieldFourierModeCount*kSplineNodes);
    std::ifstream input(argv[1],std::ios::binary);input.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(LUTNode));if(!input)return 3;
    const double beta=std::stod(argv[3]),sigma=std::stod(argv[4]),duration=std::stod(argv[5]);
    const double factor=beta/Beta;
    for(auto &node:lut){node.A*=factor;node.M_A*=factor;node.Z*=factor;node.M_Z*=factor;}
    auto p=make_solver_params(magnet_number,1.,2.,Eta);
    p.eddy_current_scale*=sigma/eddy_coefficients::sigma_si;
    State<kEddyCurrentModeCount> state{};state.down_disk_angle=.58;state.pos.x=4;state.attitude.real=1;
    if(argc==7){double values[51];std::ifstream file(argv[6],std::ios::binary);file.read(reinterpret_cast<char*>(values),sizeof(values));if(!file)return 4;state=read_state_values<true>(values);}
    const double initial_angle=state.down_disk_angle,O=2*hysteresis_physics::omega_ref;
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"time,speed_mm_s,radius_mm,wz_rad_s,JA_drive_mN,eddy_drive_mN,rolling_loss_mN,air_loss_mN,JA_Mz_uNm,eddy_Mz_uNm\n";
    const double mass=hysteresis_physics::rho_b*4*hysteresis_physics::pi/3*std::pow(hysteresis_physics::Rref,3);
    const double fu=mass*.0175*O*O,tu=mass*Eta*std::pow(hysteresis_physics::Rref,2)*O*O;
    bool sliding=false,valid=false;Vec2 direction{};FieldCache cache{};FieldResult field{};
    int code=0;double end=0;
    for(int step=0;step<=static_cast<int>(duration/.0005+.5);++step){
        end=step*.0005;
        if(step%100==0){
            const auto ja=eval_mag_field_and_force(p,state.pos,rotate(state.attitude,state.q),state.down_disk_angle,nullptr,lut.data());
            const auto all=eval_mag_field_and_force(p,state,nullptr,lut.data());
            const auto rr=add_gaussian_rolling_resistance(p,Vec3{},state,all.Force);
            const double vn=length(state.speed);const Vec2 along=vn>0?state.speed/vn:Vec2{};
            const auto qj=make_Vec2(ja.Force.x,ja.Force.y)+Eta*p.alpha*make_Vec2(ja.magnetic_angular_acceleration.y,-ja.magnetic_angular_acceleration.x);
            const auto qe=make_Vec2(all.Force.x-ja.Force.x,all.Force.y-ja.Force.y)+Eta*p.alpha*make_Vec2(all.magnetic_angular_acceleration.y-ja.magnetic_angular_acceleration.y,-all.magnetic_angular_acceleration.x+ja.magnetic_angular_acceleration.x);
            const auto qr=Eta*p.alpha*make_Vec2(rr.y,-rr.x);
            out<<end<<','<<vn*17.5*O<<','<<length(state.pos)*17.5<<','<<state.omega.z*O<<','<<(qj*along)*fu*1000<<','<<(qe*along)*fu*1000<<','<<-(qr*along)*fu*1000<<','<<p.delta*vn*vn*fu*1000<<','<<ja.magnetic_angular_acceleration.z*tu*1e6<<','<<(all.magnetic_angular_acceleration.z-ja.magnetic_angular_acceleration.z)*tu*1e6<<'\n';
        }
        if(end>=duration-1e-10)break;
        state.down_disk_angle=initial_angle+step*.0005;
        const auto result=fwd1step(.0005,p,state,sliding,direction,valid,&cache,lut.data(),&field);
        if(result!=AdvanceStepResult::kStepOk){code=static_cast<int>(result)+1;break;}
    }
    std::cout<<"beta="<<beta<<" sigma="<<sigma<<" end="<<end<<" status="<<code<<'\n';
    return 0; // Domain exit is a comparison result, not a harness failure.
}
