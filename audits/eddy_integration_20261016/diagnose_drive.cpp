// Read binary [t2d, state[51], blow_mN] rows; inspect the actual coupled states.
#include "2dmodel-hysteresis.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 4 || !kEnableEddyCurrents) return 2;
    std::vector<LUTNode> lut(kFieldFourierModeCount*kSplineNodes);
    std::ifstream field_file(argv[2],std::ios::binary);
    field_file.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(LUTNode));
    if (!field_file) return 3;
    std::ifstream input(argv[1],std::ios::binary);
    std::ofstream output(argv[3]);
    auto p=make_solver_params(magnet_number,REAL_LITERAL(1.0),REAL_LITERAL(2.0),Eta);
    const double omega=2*hysteresis_physics::omega_ref;
    const double length_unit=hysteresis_physics::L/(magnet_number/2.0);
    const double mass=hysteresis_physics::rho_b*4*hysteresis_physics::pi/3*
                      std::pow(hysteresis_physics::Rref,3);
    const double force_unit=mass*length_unit*omega*omega;
    const double power_unit=force_unit*length_unit*omega;
    output << "t2d,time_s,speed_mm_s,radius_mm,JA_drive_mN,eddy_drive_mN,roll_loss_mN,"
              "air_loss_mN,push_drive_mN,normal_mN,JA_mechanical_uW,eddy_mechanical_uW,"
              "roll_loss_uW,JA_radial_mN,eddy_radial_mN,JA_torque_drive_mN,eddy_torque_drive_mN,"
              "JA_force_drive_mN,eddy_force_drive_mN,eddy_delta_m_norm,h_norm,"
              "reversible_drive_mN,hysteresis_force_drive_mN,hysteresis_drive_mN,"
              "JA_torque_z_uNm,eddy_torque_z_uNm,h_body_dot_norm\n";
    output << std::setprecision(17);
    double row[2+kSerializedStateSize];
    while (input.read(reinterpret_cast<char*>(row),sizeof(row))) {
        const auto s=read_state_values<true>(row+1);
        const auto ja=eval_mag_field_and_force(p,s.pos,rotate(s.attitude,s.q),s.down_disk_angle,nullptr,lut.data());
        const auto reversible=eval_mag_field_and_force(p,s.pos,Vec3{},s.down_disk_angle,nullptr,lut.data());
        const auto all=eval_mag_field_and_force(p,s,nullptr,lut.data());
        const auto df=all.Force-ja.Force;
        const auto dm=all.magnetic_angular_acceleration-ja.magnetic_angular_acceleration;
        const auto rr=add_gaussian_rolling_resistance(p,Vec3{},s,all.Force);
        const double v=length(s.speed),r=length(s.pos);
        const Vec2 along=v>0?s.speed/v:Vec2{},radial=s.pos/r;
        const auto jf=make_Vec2(ja.Force.x,ja.Force.y);
        const auto rf=make_Vec2(reversible.Force.x,reversible.Force.y);
        const auto ef=make_Vec2(df.x,df.y);
        const auto jt=Eta*p.alpha*make_Vec2(ja.magnetic_angular_acceleration.y,-ja.magnetic_angular_acceleration.x);
        const auto et=Eta*p.alpha*make_Vec2(dm.y,-dm.x);
        const auto qr=Eta*p.alpha*make_Vec2(rr.y,-rr.x);
        const double push=row[1+kSerializedStateSize]*
                          (make_Vec2(s.pos.y/r,-s.pos.x/r)*along);
        const double jp=jf*s.speed+Eta*p.alpha*p.alpha*(ja.magnetic_angular_acceleration*s.omega);
        const double ep=ef*s.speed+Eta*p.alpha*p.alpha*(dm*s.omega);
        output << row[0] << ',' << row[0]/omega << ',' << v*length_unit*omega*1000 << ',' << r*length_unit*1000
               << ',' << ((jf+jt)*along)*force_unit*1000 << ',' << ((ef+et)*along)*force_unit*1000
               << ',' << -(qr*along)*force_unit*1000 << ',' << p.delta*v*v*force_unit*1000 << ',' << push
               << ',' << (p.gamma_g-all.Force.z)*force_unit*1000
               << ',' << jp*power_unit*1e6 << ',' << ep*power_unit*1e6
               << ',' << -Eta*p.alpha*p.alpha*(rr*s.omega)*power_unit*1e6
               << ',' << ((jf+jt)*radial)*force_unit*1000 << ',' << ((ef+et)*radial)*force_unit*1000
               << ',' << (jt*along)*force_unit*1000 << ',' << (et*along)*force_unit*1000
               << ',' << (jf*along)*force_unit*1000 << ',' << (ef*along)*force_unit*1000
               << ',' << length(eval_eddy_current_moment_world(s)) << ',' << length(all.h)
               << ',' << (rf*along)*force_unit*1000
               << ',' << ((jf-rf)*along)*force_unit*1000
               << ',' << ((jf-rf+jt)*along)*force_unit*1000
               << ',' << ja.magnetic_angular_acceleration.z*mass*Eta*std::pow(hysteresis_physics::Rref,2)*omega*omega*1e6
               << ',' << dm.z*mass*Eta*std::pow(hysteresis_physics::Rref,2)*omega*omega*1e6
               << ',' << length(eval_body_field_derivative(s,all.h,all.hGrad3d)) << '\n';
    }
    return input.eof() && output ? 0 : 4;
}
