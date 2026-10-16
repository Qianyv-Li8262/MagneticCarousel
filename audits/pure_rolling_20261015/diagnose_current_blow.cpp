#include "2dmodel-hysteresis.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
int main(int argc,char **argv) {
    if(argc!=4) return 2;
    std::vector<LUTNode> lut(kFieldFourierModeCount*kSplineNodes);
    std::ifstream lf(argv[2],std::ios::binary); lf.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(LUTNode));
    if(!lf) return 3;
    std::ifstream in(argv[1]); std::ofstream out(argv[3]); std::string line;
    std::getline(in,line);
    out<<"time_s,speed_m_s,radius_mm,Qmag_along_v_mN,Droll_mN,Fpush_along_v_mN,normal_mN,amag_m_s2,arr_m_s2,apush_m_s2,mag_power_uW,roll_power_uW,air_power_uW,Fx_mN,Fy_mN,Fz_mN,taux_uNm,tauy_uNm,tauz_uNm\n"<<std::setprecision(17);
    const double O=100*2*3.14159265358979323846/60*2,k=4/.07;
    const double mass=7850*4*3.14159265358979323846*.006*.006*.006/3;
    SolverParams p{.342857142857143,8.02199442675159e-5,5.111798459304/4,17302.6849037817/4,367982.448040497/4};
    const double fs=mass*O*O/k,ts=mass*Eta*p.alpha*p.alpha*O*O/(k*k);
    const double E=(1-C_JA)/((1-Xi_JA)*(Mu_r+2));
    while(std::getline(in,line)) {
        for(auto &c:line) if(c==',') c=' ';
        std::istringstream row(line); double t,sv[15],mode,blow;
        row>>t;for(auto &x:sv) row>>x;row>>mode>>blow;
        if(!row) return 4;
        State s{sv[0],{sv[1],sv[2]},{sv[3],sv[4]},{sv[5],{sv[6],sv[7],sv[8]}},{sv[9],sv[10],sv[11]},{sv[12],sv[13],sv[14]}};
        FieldCache cache{};
        const auto f=eval_mag_field_and_force(p,s.pos,rotate(s.attitude,s.q),s.down_disk_angle,&cache,lut.data());
        const Vec3 Mmag=-p.gamma_m*E*cross(rotate(s.attitude,s.q),f.h);
        const Vec3 Mrr=add_gaussian_rolling_resistance(p,Vec3{},s,f.Force);
        const double vn=length(s.speed),pn=length(s.pos);
        const Vec2 dir=vn>0?s.speed/vn:Vec2{};
        const Vec2 qmag=make_Vec2(f.Force.x,f.Force.y)+Eta*p.alpha*make_Vec2(Mmag.y,-Mmag.x);
        const Vec2 qrr=Eta*p.alpha*make_Vec2(Mrr.y,-Mrr.x);
        const Vec2 reverse={s.pos.y/pn,-s.pos.x/pn};
        const double speed=vn*O/k,qn=(qmag*dir)*fs,dr=-(qrr*dir)*fs,push=blow*.001*(reverse*dir);
        const double magpower=fs*(f.Force.x*s.speed.x+f.Force.y*s.speed.y)*O/k+ts*(Mmag*s.omega)*O;
        const double rrpower=-ts*(Mrr*s.omega)*O;
        out<<t/O<<','<<speed<<','<<pn/k*1000<<','<<qn*1000<<','<<dr*1000<<','<<push*1000<<','<<(p.gamma_g-f.Force.z)*fs*1000<<','<<qn/(1.4*mass)<<','<<-dr/(1.4*mass)<<','<<push/(1.4*mass)<<','<<magpower*1e6<<','<<rrpower*1e6<<','<<p.delta*vn*vn*vn*fs*O/k*1e6<<','<<f.Force.x*fs*1000<<','<<f.Force.y*fs*1000<<','<<f.Force.z*fs*1000<<','<<Mmag.x*ts*1e6<<','<<Mmag.y*ts*1e6<<','<<Mmag.z*ts*1e6<<'\n';
    }
}
