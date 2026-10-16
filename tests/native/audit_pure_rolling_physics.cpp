// Standalone theoretical audit of the current exact-field FP64 rolling RHS.
// Compile with -std=c++17 -O2 -DUSE_FP64 -Ikernels/headers.
#include "2dmodel-hysteresis.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>

double residual(double x, double y) { return std::abs(x-y)/(1+std::abs(x)+std::abs(y)); }
Vec3 grad_times(const TracelessSymTensor &g, Vec3 v) {
    const auto *j=g.component;
    return {j[0]*v.x+j[1]*v.y+j[2]*v.z,
            j[1]*v.x+j[3]*v.y+j[4]*v.z,
            j[2]*v.x+j[4]*v.y-(j[0]+j[3])*v.z};
}
int main() {
    static_assert(!kEnableInclinedDisk, "Energy audit below is for a horizontal fixed support");
    static_assert(kFieldEvalMode == FieldEvalMode::kExact);
    const double alpha=0.342857142857143, gm=367982.448040497/4;
    SolverParams p{alpha,8.02199442675159e-5,5.111798459304/4,Eta*alpha*alpha*gm,gm};
    constexpr double E=(1-C_JA)/((1-Xi_JA)*(Mu_r+2));
    constexpr double D=(Mu_r-1)/(Mu_r+2), c=Xi_JA/Alpha_JA;
    constexpr double a=1-C_JA*Xi_JA-E*c;
    constexpr double L=Alpha_JA*(1-C_JA)-E;
    constexpr double s0=3/(Mu_r+2), A0=s0*c/a;
    assert(a>0 && L<0 && A0>0);
    double mech=0, energy=0, passive_max=0, field_deriv=0, projection=0;
    int active=0, inactive=0;
    for(int i=0;i<400;++i) {
        const double t=0.137*i;
        State<kEddyCurrentModeCount> x{};
        x.pos={4+0.35*std::sin(t),0.45*std::cos(1.3*t)};
        x.speed={0.13*std::sin(0.8*t),0.22*std::cos(t)};
        x.omega={-x.speed.y/alpha,x.speed.x/alpha,2*std::sin(1.7*t)};
        x.attitude={std::cos(t/2),{std::sin(t/2)/3,2*std::sin(t/2)/3,2*std::sin(t/2)/3}};
        x.down_disk_angle=0.31*t;
        x.q={0.35*std::sin(2*t),0.4*std::cos(1.1*t),0.3*std::sin(0.7*t+1)};
        State<kEddyCurrentModeCount> dx{}; Vec3 F{},M{};
        Vec3 hb_dot_rhs;
        assert(eval_rhs<true>(x,dx, hb_dot_rhs,p,false,Vec2{},nullptr,nullptr,nullptr,&F,&M));
        const Vec2 G={F.x-p.delta*length(x.speed)*x.speed.x,F.y-p.delta*length(x.speed)*x.speed.y};
        const Vec2 friction=dx.speed-G;
        const Vec3 reconstructed=M+make_Vec3(friction.y,-friction.x,0)/(Eta*alpha);
        mech=std::max({mech,length(dx.omega-reconstructed)/(1+length(dx.omega)),
                       std::abs(dx.speed.x-alpha*dx.omega.y),std::abs(dx.speed.y+alpha*dx.omega.x)});
        const auto field=eval_mag_field(x.pos,x.down_disk_angle);
        const Vec3 qw=rotate(x.attitude,x.q);
        const Vec3 mag=-gm*E*cross(qw,field.h);
        const Vec3 hb_dot=eval_body_field_derivative(x,field.h,field.hGrad3d);
        const double diss=-p.gamma*E/A0*(x.q*(dx.q-A0*hb_dot));
        if(x.q*hb_dot>0) ++active; else ++inactive;
        passive_max=std::max(passive_max,-diss);
        // Independent lab-frame derivative of the rotating source field.
        const Vec3 ht=cross(make_Vec3(0.,0.,1.),field.h)-
                      grad_times(field.hGrad3d,make_Vec3(-x.pos.y,x.pos.x,0));
        const double eps=1e-6;
        const Vec3 ht_fd=(eval_mag_field(x.pos,x.down_disk_angle+eps).h-
                          eval_mag_field(x.pos,x.down_disk_angle-eps).h)/(2*eps);
        field_deriv=std::max(field_deriv,length(ht-ht_fd)/(1+length(ht)));
        const Vec3 hdot=ht+grad_times(field.hGrad3d,make_Vec3(x.speed.x,x.speed.y,0));
        const double Tdot=x.speed*dx.speed+Eta*alpha*alpha*(x.omega*dx.omega);
        const double Udot=-p.gamma*D*(field.h*hdot)+p.gamma*E/A0*(x.q*dx.q);
        const double source=x.pos.x*F.y-x.pos.y*F.x+Eta*alpha*alpha*mag.z;
        const double loss_air=p.delta*std::pow(length(x.speed),3);
        const double loss_roll=-Eta*alpha*alpha*((M-mag)*x.omega);
        energy=std::max(energy,residual(Tdot+Udot,source-diss-loss_air-loss_roll));
        // Contact-impulse projection must remove precisely this slip energy.
        State<kEddyCurrentModeCount> z=x; z.speed.x+=0.017; z.speed.y-=0.023;
        const Vec2 slip=z.speed+alpha*make_Vec2(-z.omega.y,z.omega.x);
        const double before=(z.speed*z.speed+Eta*alpha*alpha*(z.omega*z.omega))/2;
        project_pure_rolling(z,p);
        const double after=(z.speed*z.speed+Eta*alpha*alpha*(z.omega*z.omega))/2;
        projection=std::max(projection,residual(before-after,Eta/(2*(1+Eta))*(slip*slip)));
    }
    std::printf("samples=400 active=%d inactive=%d\nNewton/constraint=%.12g\nenergy_balance=%.12g\nnegative_dissipation=%.12g\nsource_field_derivative=%.12g\nprojection_energy=%.12g\n",active,inactive,mech,energy,passive_max,field_deriv,projection);
    assert(mech<1e-10 && energy<1e-10 && passive_max<1e-10 && field_deriv<1e-9 && projection<1e-12);
    // Independent measured-definition check: torque mu*N*R -> a=mu*g/(1+Eta).
    p.gamma_m=0; p.gamma=0; p.delta=0;
    State<kEddyCurrentModeCount> x{}; x.pos.x=4; x.speed.x=0.2; x.omega.y=x.speed.x/alpha; x.attitude.real=1;
    State<kEddyCurrentModeCount> dx{};
    Vec3 hb_dot_rhs;
    assert(eval_rhs<false>(x,dx, hb_dot_rhs,p,false,Vec2{}));
    const double reg=x.omega.y/std::sqrt(x.omega.y*x.omega.y+kRollingOmegaEpsilon*kRollingOmegaEpsilon);
    assert(residual(dx.speed.x,-kRollingMuLow*p.gamma_g/(1+Eta)*reg)<1e-12);
    std::puts("rolling_resistance_definition=PASS");
    // Actual accepted rolling advancement against a separately implemented RK4.
    p.gamma_m=gm; p.gamma=Eta*alpha*alpha*gm; p.gamma_g=20;
    x={}; x.pos={4,0.1}; x.speed={0.02,-0.2};
    x.omega={-x.speed.y/alpha,x.speed.x/alpha,0.7};
    x.attitude.real=1; x.q={0.01,0.02,-0.03}; x.down_disk_angle=0.58;
    auto rhs=[&](const State<kEddyCurrentModeCount> &s) { State<kEddyCurrentModeCount> d{}; Vec3 hb_dot_rhs; assert(eval_rhs<false>(s,d, hb_dot_rhs,p,false,Vec2{})); return d; };
    auto error=[](const State<kEddyCurrentModeCount> &u,const State<kEddyCurrentModeCount> &v) {
        return length(u.pos-v.pos)+length(u.speed-v.speed)+length(u.q-v.q)+length(u.omega-v.omega)+
               std::abs(u.attitude.real-v.attitude.real)+length(u.attitude.imagine-v.attitude.imagine);
    };
    double previous=0;
    for(double dt : {0.004,0.002,0.001,0.0005}) {
        State<kEddyCurrentModeCount> ref=x;
        const double ds=dt/256;
        for(int j=0;j<256;++j) {
            const State<kEddyCurrentModeCount> k1=rhs(ref),k2=rhs(ref+ds/2*k1),k3=rhs(ref+ds/2*k2),k4=rhs(ref+ds*k3);
            ref=ref+ds/6*(k1+2*k2+2*k3+k4);
            assert(normalize_attitude(ref));
        }
        State<kEddyCurrentModeCount> actual=x; bool sliding=false, valid=false; Vec2 dir{};
        assert(fwd1step(dt,p,actual,sliding,dir,valid)==AdvanceStepResult::kStepOk);
        assert(!sliding);
        const double err=error(actual,ref);
        std::printf("rolling_step dt=%.7g local_error=%.12g ratio=%.8g\n",dt,err,previous/err);
        if(previous>0) assert(previous/err>6 && previous/err<10);
        previous=err;
    }
}
