// Independent scalar/ETDRK4 audit of the enabled rolling capture trajectory.
// INPUT rows: t2d, state[51], blow_mN. Arguments: INPUT LUT OUTPUT_DIRECTORY.
#include "2dmodel-hysteresis.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using A=std::array<double,51>;
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(double x,V a){return {x*a[0],x*a[1],x*a[2]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(V a){return std::sqrt(dot(a,a));}
V cross3(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
V vv(Vec3 v){return {v.x,v.y,v.z};}
V rotate_matrix(const A &x,V v,bool transpose=false){
    const double w=x[5],a=x[6],b=x[7],c=x[8];
    const double R[3][3]={{1-2*(b*b+c*c),2*(a*b-w*c),2*(a*c+w*b)},
        {2*(a*b+w*c),1-2*(a*a+c*c),2*(b*c-w*a)},
        {2*(a*c-w*b),2*(b*c+w*a),1-2*(a*a+b*b)}};
    V out{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=(transpose?R[j][i]:R[i][j])*v[j];
    return out;
}
V jt(const FieldResult &f,V v){auto j=f.hGrad3d.component;return {
    j[0]*v[0]+j[1]*v[1]+j[2]*v[2],j[1]*v[0]+j[3]*v[1]+j[4]*v[2],
    j[2]*v[0]+j[4]*v[1]-(j[0]+j[3])*v[2]};}
void normalize(A &x){double n=0;for(int j=5;j<9;++j)n+=x[j]*x[j];n=std::sqrt(n);for(int j=5;j<9;++j)x[j]/=n;}
void twist(A &x,V axis,double angle){
    const double c=std::cos(angle/2),sn=std::sin(angle/2),w=x[5];
    V q{x[6],x[7],x[8]},v=mul(sn,axis);
    V out=add(add(mul(c,q),mul(w,v)),cross3(v,q));
    x[5]=c*w-dot(v,q);for(int i=0;i<3;++i)x[6+i]=out[i];
}
std::vector<LUTNode> lut;
SolverParams p;
constexpr double E=(1-C_JA)/((1-Xi_JA)*(Mu_r+2));
constexpr double D=(Mu_r-1)/(Mu_r+2);
struct Calc {A rhs{};V hb{},F{},M{},rr{},dm{},qd{};FieldResult field{};double capacity{};};
Calc scalar_rhs(const A &x){
    Calc c; c.field=eval_mag_field({x[1],x[2]},x[0],nullptr,lut.data());
    V h=vv(c.field.h),w{x[9],x[10],x[11]},q{x[12],x[13],x[14]},dm_body{};
    for(int j=0;j<12;++j)for(int k=0;k<3;++k)dm_body[k]+=eddy_coefficients::aj[j]*x[15+12*k+j];
    c.dm=rotate_matrix(x,dm_body);
    V qw=rotate_matrix(x,q);
    V moment=add(sub(mul(D,h),mul(E,qw)),mul(1.0/3,c.dm));
    c.F=mul(p.gamma_m*Eta*p.alpha*p.alpha,jt(c.field,moment));
    c.M=mul(p.gamma_m,cross3(add(mul(-E,qw),mul(1.0/3,c.dm)),h));
    const double wn=std::sqrt(w[0]*w[0]+w[1]*w[1]);
    const double mu=kRollingMuHigh+(kRollingMuLow-kRollingMuHigh)*std::exp(-wn*wn/(kRollingTransitionOmega*kRollingTransitionOmega));
    const double damping=mu*(p.gamma_g-c.F[2])/(Eta*p.alpha*std::sqrt(wn*wn+kRollingOmegaEpsilon*kRollingOmegaEpsilon))+kRollingLambda;
    c.rr={-damping*w[0],-damping*w[1],0};
    V M=add(c.M,c.rr);
    const double speed=std::hypot(x[3],x[4]);
    const double fx=c.F[0]-p.delta*speed*x[3],fy=c.F[1]-p.delta*speed*x[4];
    c.rhs[3]=(fx+Eta*p.alpha*M[1])/(1+Eta);
    c.rhs[4]=(fy-Eta*p.alpha*M[0])/(1+Eta);
    c.rhs[9]=-c.rhs[4]/p.alpha;c.rhs[10]=c.rhs[3]/p.alpha;c.rhs[11]=M[2];
    V lab=add(jt(c.field,{x[3]+x[2],x[4]-x[1],0}),{-h[1],h[0],0});
    c.hb=rotate_matrix(x,sub(lab,cross3(w,h)),true);
    // Independent linear-system form: (a I + b qhat qhat^T) qdot
    // = (c0 I + d qhat qhat^T) h0dot, solved with a rank-one inverse.
    const V h0dot=mul(3.0/(Mu_r+2),c.hb);
    const double inter=Alpha_JA*(1-C_JA*Xi_JA)/Xi_JA-E;
    const double he_scale=Alpha_JA/(Xi_JA*inter);
    c.qd=mul(1/inter,h0dot);
    if(dot(q,mul(he_scale,h0dot))>0){
        const double qn=norm(q),c0=Xi_JA/Alpha_JA,d=-qn/K;
        const double aa=1-C_JA*Xi_JA-E*c0;
        const double bb=-Alpha_JA*(1-C_JA)*qn/K-E*d;
        const V n=mul(1/qn,q);
        const V rhs=add(mul(c0,h0dot),mul(d*dot(n,h0dot),n));
        c.qd=mul(1/aa,sub(rhs,mul(bb*dot(n,rhs)/(aa+bb),n)));
    }
    for(int k=0;k<3;++k){c.rhs[12+k]=c.qd[k];for(int j=0;j<12;++j)c.rhs[15+12*k+j]=-c.hb[k];}
    c.rhs[0]=1;c.rhs[1]=x[3];c.rhs[2]=x[4];
    V qi{x[6],x[7],x[8]},qi_dot=mul(.5,add(mul(x[5],w),cross3(w,qi)));
    c.rhs[5]=-.5*dot(w,qi);for(int k=0;k<3;++k)c.rhs[6+k]=qi_dot[k];
    const double reqx=fx-p.alpha*M[1],reqy=fy+p.alpha*M[0];
    c.capacity=std::hypot(reqx,reqy)/((1+1/Eta)*Mu*(p.gamma_g-c.F[2]));
    return c;
}
double potential(A x){
    const auto c=scalar_rhs(x);V h=vv(c.field.h),q=rotate_matrix(x,{x[12],x[13],x[14]});
    return -p.gamma_m*Eta*p.alpha*p.alpha*(D*dot(h,h)/2-E*dot(q,h)+dot(c.dm,h)/3);
}
struct ETD{
    A rate{},ee{},e2{},Q{},f1{},f2{},f3{};
    explicit ETD(double h){
        for(int j=0;j<51;++j){
            rate[j]=j<15?0:eddy_coefficients::inv_tau_ref_star[(j-15)%12]/p.eddy_current_scale;
            const double z=-h*rate[j];ee[j]=std::exp(z);e2[j]=std::exp(z/2);
            double phi[3]{};
            if(std::abs(z)<.25){
                for(int k=1;k<=3;++k){double term=1;for(int a=2;a<=k;++a)term/=a;double value=term;
                    for(int n=1;n<32;++n){term*=z/(n+k);value+=term;}phi[k-1]=value;}
            }else{phi[0]=std::expm1(z)/z;phi[1]=(phi[0]-1)/z;phi[2]=(phi[1]-.5)/z;}
            Q[j]=z==0?h/2:std::expm1(z/2)/(-rate[j]);
            f1[j]=h*(phi[0]-3*phi[1]+4*phi[2]);f2[j]=h*(phi[1]-2*phi[2]);f3[j]=h*(-phi[1]+4*phi[2]);
        }
    }
    A step(const A &x){
        const A n0=scalar_rhs(x).rhs;A a{},b{},c{},out{};
        for(int j=0;j<51;++j)a[j]=e2[j]*x[j]+Q[j]*n0[j];
        const A na=scalar_rhs(a).rhs;
        for(int j=0;j<51;++j)b[j]=e2[j]*x[j]+Q[j]*na[j];
        const A nb=scalar_rhs(b).rhs;
        for(int j=0;j<51;++j)c[j]=e2[j]*a[j]+Q[j]*(2*nb[j]-n0[j]);
        const A nc=scalar_rhs(c).rhs;
        for(int j=0;j<51;++j)out[j]=ee[j]*x[j]+f1[j]*n0[j]+2*f2[j]*(na[j]+nb[j])+f3[j]*nc[j];
        normalize(out);return out;
    }
};
int main(int argc,char **argv)try{
    if(argc!=4)return 2;
    static_assert(kEnableEddyCurrents && kSerializedStateSize==51);
    p=make_solver_params(magnet_number,1.,2.,Eta);
    lut.resize(kFieldFourierModeCount*kSplineNodes);std::ifstream lf(argv[2],std::ios::binary);
    lf.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(LUTNode));if(!lf)throw std::runtime_error("LUT read");
    std::ifstream in(argv[1],std::ios::binary);std::vector<std::array<double,53>> rows;std::array<double,53> row;
    while(in.read(reinterpret_cast<char*>(row.data()),sizeof(row)))rows.push_back(row);
    if(rows.empty())throw std::runtime_error("empty input");
    double rhs_error=0,hb_error=0,force_fd=0,torque_fd=0,energy_error=0,cache_error=0,capacity=0,quat_error=0;
    for(std::size_t sample=0;sample<rows.size();sample+=37){
        A x{};std::copy_n(rows[sample].data()+1,51,x.begin());double qn=0;for(int j=5;j<9;++j)qn+=x[j]*x[j];quat_error=std::max(quat_error,std::abs(qn-1));normalize(x);
        auto prod=read_state_values<true>(x.data());State<kEddyCurrentModeCount> dx{};Vec3 hb,F,M;
        if(!eval_rhs<true>(prod,dx,hb,p,false,Vec2{},nullptr,lut.data(),nullptr,&F,&M))throw std::runtime_error("contact");
        A pd{};write_state_values<true>(dx,pd.data());const auto c=scalar_rhs(x);capacity=std::max(capacity,c.capacity);
        for(int j=0;j<15;++j)rhs_error=std::max(rhs_error,std::abs(pd[j]-c.rhs[j])/(1+std::abs(pd[j])));
        const double eps=1e-6;A plus=x,minus=x;V omega{x[9],x[10],x[11]};const double wn=norm(omega);
        plus[0]+=eps;minus[0]-=eps;for(int j=1;j<=2;++j){plus[j]+=eps*x[j+2];minus[j]-=eps*x[j+2];}
        if(wn>0){twist(plus,mul(1/wn,omega),eps*wn);twist(minus,mul(1/wn,omega),-eps*wn);}
        const auto fp=eval_mag_field({plus[1],plus[2]},plus[0],nullptr,lut.data());
        const auto fm=eval_mag_field({minus[1],minus[2]},minus[0],nullptr,lut.data());
        V hp=rotate_matrix(plus,vv(fp.h),true),hm=rotate_matrix(minus,vv(fm.h),true);
        hb_error=std::max(hb_error,norm(sub(mul(1/(2*eps),sub(hp,hm)),vv(hb))));
        for(int axis=0;axis<2;++axis){plus=x;minus=x;plus[1+axis]+=eps;minus[1+axis]-=eps;
            force_fd=std::max(force_fd,std::abs(-(potential(plus)-potential(minus))/(2*eps)-c.F[axis]));}
        for(int axis=0;axis<3;++axis){plus=x;minus=x;V unit{};unit[axis]=1;twist(plus,unit,eps);twist(minus,unit,-eps);
            torque_fd=std::max(torque_fd,std::abs(-(potential(plus)-potential(minus))/(2*eps*Eta*p.alpha*p.alpha)-c.M[axis]));}
        FieldCache cache{};const auto cached=eval_mag_field({x[1],x[2]},x[0],&cache,lut.data());
        cache_error=std::max(cache_error,norm(sub(vv(cached.h),vv(c.field.h))));
        constexpr double c0=Xi_JA/Alpha_JA,aa=1-C_JA*Xi_JA-E*c0,A0=3/(Mu_r+2)*c0/aa;
        const double gamma=p.gamma_m*Eta*p.alpha*p.alpha;
        const V h=vv(c.field.h),q{x[12],x[13],x[14]},lab=mul(1/(2*eps),sub(vv(fp.h),vv(fm.h)));
        double Tdot=x[3]*pd[3]+x[4]*pd[4]+Eta*p.alpha*p.alpha*dot(omega,{pd[9],pd[10],pd[11]});
        double Udot=-gamma*D*dot(h,lab)+gamma*E/A0*dot(q,c.qd),eddy_loss=0;
        for(int k=0;k<3;++k)for(int j=0;j<12;++j){double y=x[15+12*k+j],rate=eddy_coefficients::inv_tau_ref_star[j]/p.eddy_current_scale;
            Udot+=gamma/3*eddy_coefficients::aj[j]*y*(-rate*y-c.hb[k]);eddy_loss+=gamma/3*eddy_coefficients::aj[j]*rate*y*y;}
        const double source=x[1]*c.F[1]-x[2]*c.F[0]+Eta*p.alpha*p.alpha*c.M[2];
        const double ja_loss=-gamma*E/A0*dot(q,sub(c.qd,mul(A0,c.hb)));
        const double roll_loss=-Eta*p.alpha*p.alpha*dot(c.rr,omega),air=p.delta*std::pow(std::hypot(x[3],x[4]),3);
        energy_error=std::max(energy_error,std::abs(Tdot+Udot-source+ja_loss+eddy_loss+roll_loss+air)/(1+std::abs(source)));
    }
    std::filesystem::create_directories(argv[3]);std::ofstream summary(std::filesystem::path(argv[3])/"instant_checks.json");
    summary<<std::setprecision(17)<<"{\n\"rhs_relative_error\":"<<rhs_error<<",\n\"body_derivative_abs_error\":"<<hb_error<<",\n\"force_potential_FD_error\":"<<force_fd<<",\n\"torque_potential_FD_error\":"<<torque_fd<<",\n\"energy_identity_error\":"<<energy_error<<",\n\"field_cache_error\":"<<cache_error<<",\n\"max_static_friction_fraction\":"<<capacity<<",\n\"max_quaternion_norm_error\":"<<quat_error<<"\n}\n";
    summary.close();std::cout<<std::ifstream(std::filesystem::path(argv[3])/"instant_checks.json").rdbuf()<<std::flush;
    if(rhs_error>1e-10 || hb_error>1e-7 || force_fd>1e-6 || torque_fd>1e-6 || energy_error>1e-7)throw std::runtime_error("instant audit failed");
    ETD ref(.0005);A x{};std::copy_n(rows[0].data()+1,51,x.begin());auto nc=read_state_values<true>(x.data());bool sliding=false,valid=false;Vec2 direction{};
    std::ofstream out(std::filesystem::path(argv[3])/"independent_trajectory.csv");out<<"time,ref_speed,ref_wz,ref_radius,nocache_speed,nocache_wz,nocache_radius\n"<<std::setprecision(17);
    const double angle=x[0],O=2*hysteresis_physics::omega_ref,L=.0175;int mode_changes=0;
    for(int step=0;step<=2000000;++step){
        if(step%100==0)out<<step*.0005<<','<<std::hypot(x[3],x[4])*L*O*1000<<','<<x[11]*O<<','<<std::hypot(x[1],x[2])*L*1000<<','<<length(nc.speed)*L*O*1000<<','<<nc.omega.z*O<<','<<length(nc.pos)*L*1000<<'\n';
        if(step==2000000)break;
        x=ref.step(x);x[0]=angle+(step+1)*.0005;
        nc.down_disk_angle=angle+step*.0005;
        if(fwd1step(.0005,p,nc,sliding,direction,valid,nullptr,lut.data(),nullptr)!=AdvanceStepResult::kStepOk)throw std::runtime_error("cache-off advance failed");
        mode_changes+=sliding;
        if(step%400000==0)std::cout<<"ETDRK4 and cache-off t="<<step*.0005<<'\n'<<std::flush;
    }
    std::cout<<"PASS scalar ETDRK4, cache-off trajectory; sliding endpoints="<<mode_changes<<'\n';
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
