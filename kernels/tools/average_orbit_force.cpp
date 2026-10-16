// Prescribed circular motion: average the CURRENT production JA magnetic force.
// No mechanical trajectory integration or fitted parameter changes.
#include "2dmodel-hysteresis.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 11) {
        std::cerr << "lut dt field_cycles period_s rpm spin_z_rad_s r_min_mm r_max_mm dr_mm sr\n";
        return 2;
    }
    std::vector<LUTNode> lut(kFieldFourierModeCount * kSplineNodes);
    std::ifstream file(argv[1], std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(lut.data()), lut.size()*sizeof(LUTNode)))
        throw std::runtime_error("Cannot read production LUT");
    const double requested_dt=std::stod(argv[2]), cycles=std::stod(argv[3]);
    const double period=std::stod(argv[4]), rpm=std::stod(argv[5]), wz_si=std::stod(argv[6]);
    const double rmin=std::stod(argv[7]), rmax=std::stod(argv[8]), dr=std::stod(argv[9]);
    const double sr=std::stod(argv[10]);
    const double pi=hysteresis_physics::pi, omega=rpm*2*pi/60, u=-2*pi/(period*omega);
    if (!std::isfinite(sr) || sr <= 0 || !std::isfinite(omega) || omega <= 0)
        throw std::invalid_argument("sr and rpm must be finite and positive");
    const double radius=hysteresis_physics::Rref*sr;
    const double length_unit=hysteresis_physics::L/(magnet_number/2.);
    const double mass=hysteresis_physics::rho_b*4*pi/3*radius*radius*radius;
    const SolverParams params=make_solver_params(magnet_number, sr, omega/hysteresis_physics::omega_ref, Eta);
    const double alpha=params.alpha;
    const double field_period=2*pi/(magnet_number/2.)/(1-u);
    const long steps=static_cast<long>(std::ceil(cycles*field_period/requested_dt));
    const double dt=cycles*field_period/steps;
    constexpr double E=(1-C_JA)/(1-Xi_JA)/(2+Mu_r);
    std::cout << std::setprecision(16);
    std::cout << "radius_mm,period_s,rpm,spin_z_rad_s,mean_inward_magnetic_mN,required_net_inward_mN,mean_rolling_generalized_inward_mN,required_rolling_generalized_mN,mean_tangential_accel_m_s2,mean_normal_load_mN,mean_total_magnetic_power_uW,first_half_inward_mN,second_half_inward_mN\n";
    for (double radius=rmin;radius<=rmax+dr*.1;radius+=dr) {
        State<kEddyCurrentModeCount> state{};state.pos={radius*.001/length_unit,0};state.speed={0,u*state.pos.x};
        state.attitude={1,{0,0,0}};
        state.omega={-u*state.pos.x/alpha,0,wz_si/omega};
        const Vec3 frame_spin=state.omega-make_Vec3(0.,0.,u);
        FieldCache cache{};
        auto derivative=[&](double t, Vec3 q) {
            State<kEddyCurrentModeCount> trial=state,rhs{};trial.down_disk_angle=(1-u)*t;trial.q=q;
            const Vec2 direction{};
            Vec3 hb_dot;
            if (!eval_rhs<false>(trial,rhs, hb_dot,params,false,direction,&cache,lut.data()))
                throw std::runtime_error("Negative normal load on prescribed orbit");
            return rhs.q+cross(frame_spin,q);
        };
        Vec3 q{};
        std::array<double,8> sums{};long samples=0,first_count=0,second_count=0;
        for (long i=0;i<steps;++i) {
            const double t=i*dt;
            const Vec3 k1=derivative(t,q);
            const Vec3 mid=q+.5*dt*k1;
            const Vec3 k2=derivative(t+.5*dt,mid);
            q=q+dt*k2;
            if (i<steps/2)continue;
            State<kEddyCurrentModeCount> sample=state,rhs{};sample.q=mid;sample.down_disk_angle=(1-u)*(t+.5*dt);
            const auto f=eval_mag_field_and_force(params,sample.pos,mid,sample.down_disk_angle,&cache,lut.data());
            const Vec3 torque=-params.gamma_m*E*cross(mid,f.h);
            Vec3 hb_dot;
            eval_rhs<false>(sample,rhs, hb_dot,params,false,Vec2{},&cache,lut.data());
            const double scale=mass*length_unit*omega*omega*1000;
            const double inward=-f.Force.x*scale;
            sums[0]+=inward;
            sums[1]+=-rhs.speed.x*(1+Eta)*scale;
            sums[2]+=rhs.speed.y*length_unit*omega*omega;
            sums[3]+=(params.gamma_g-f.Force.z)*scale;
            sums[4]+=(f.Force.y*sample.speed.y+Eta*alpha*alpha*(torque*sample.omega))*mass*length_unit*length_unit*omega*omega*omega*1e6;
            if (i<3*steps/4){sums[5]+=inward;++first_count;}
            else {sums[6]+=inward;++second_count;}
            ++samples;
        }
        const double required=mass*radius*.001*std::pow(2*pi/period,2)*1000;
        std::cout << radius << ',' << period << ',' << rpm << ',' << wz_si << ','
            << sums[0]/samples << ',' << required << ',' << sums[1]/samples << ','
            << (1+Eta)*required << ',' << sums[2]/samples << ',' << sums[3]/samples << ','
            << sums[4]/samples << ',' << sums[5]/first_count << ',' << sums[6]/second_count << '\n';
    }
}
