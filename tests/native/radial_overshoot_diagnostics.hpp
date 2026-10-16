#pragma once
// Physical generalized radial loads for a sphere in pure rolling.
// (1+Eta)*m*r_ddot = Q_r + (1+Eta)*m*v_phi^2/r.
struct RadialDiagnostics {
    std::ofstream out;
    SolverParams p;
    const LUTNode *lut;
    double O=2*hysteresis_physics::omega_ref;
    double mass=hysteresis_physics::rho_b*4*hysteresis_physics::pi/3*std::pow(hysteresis_physics::Rref,3);
    double fu=mass*.0175*O*O;
    RadialDiagnostics(const std::string &path,const SolverParams &params,const LUTNode *nodes):out(path),p(params),lut(nodes) {
        out<<std::setprecision(17)<<"time_s,r_mm,vt_mm_s,vr_mm_s,wz_rad_s,rev_Fr_mN,memory_Fr_mN,memory_Tr_mN,eddy_Fr_mN,eddy_Tr_mN,roll_Qr_mN,air_Qr_mN,Qr_mN,centrifugal_mN,net_r_mN,rddot_mm_s2,rev_Qclock_mN,memory_Qclock_mN,eddy_Qclock_mN,roll_Qclock_mN,memory_Mz_uNm,eddy_Mz_uNm,rolling,closure_mN\n";
    }
    void write(double time,const State<kEddyCurrentModeCount>&s,bool sliding) {
        const auto ja=eval_mag_field_and_force(p,s.pos,rotate(s.attitude,s.q),s.down_disk_angle,nullptr,lut);
        const auto rev=eval_mag_field_and_force(p,s.pos,Vec3{},s.down_disk_angle,nullptr,lut);
        const auto all=eval_mag_field_and_force(p,s,nullptr,lut);
        const auto rr=add_gaussian_rolling_resistance(p,Vec3{},s,all.Force);
        const Vec2 er=s.pos/length(s.pos),clock={er.y,-er.x};
        const Vec2 jf={ja.Force.x,ja.Force.y},rf={rev.Force.x,rev.Force.y};
        const Vec2 ef={all.Force.x-ja.Force.x,all.Force.y-ja.Force.y};
        const Vec2 jt=Eta*p.alpha*make_Vec2(ja.magnetic_angular_acceleration.y,-ja.magnetic_angular_acceleration.x);
        const Vec3 em=all.magnetic_angular_acceleration-ja.magnetic_angular_acceleration;
        const Vec2 et=Eta*p.alpha*make_Vec2(em.y,-em.x);
        const Vec2 roll=Eta*p.alpha*make_Vec2(rr.y,-rr.x);
        const Vec2 air=-p.delta*length(s.speed)*s.speed;
        const Vec2 total=jf+ef+jt+et+roll+air;
        const double vt=-(s.speed*clock),vr=s.speed*er,r=length(s.pos);
        const double cf=(1+Eta)*vt*vt/r;
        State<kEddyCurrentModeCount> rhs{};Vec3 hb{};Vec2 zero{};
        eval_rhs<false>(s,rhs,hb,p,sliding,zero,nullptr,lut);
        const double rddot=rhs.speed*er+vt*vt/r;
        const double tu=mass*Eta*std::pow(hysteresis_physics::Rref,2)*O*O*1e6;
        out<<time<<','<<r*17.5<<','<<vt*17.5*O<<','<<vr*17.5*O<<','<<s.omega.z*O
           <<','<<(rf*er)*fu*1000<<','<<((jf-rf)*er)*fu*1000<<','<<(jt*er)*fu*1000
           <<','<<(ef*er)*fu*1000<<','<<(et*er)*fu*1000<<','<<(roll*er)*fu*1000<<','<<(air*er)*fu*1000
           <<','<<(total*er)*fu*1000<<','<<cf*fu*1000<<','<<((total*er)+cf)*fu*1000
           <<','<<rddot*17.5*O*O<<','<<(rf*clock)*fu*1000<<','<<((jf-rf+jt)*clock)*fu*1000
           <<','<<((ef+et)*clock)*fu*1000<<','<<(roll*clock)*fu*1000
           <<','<<ja.magnetic_angular_acceleration.z*tu<<','<<em.z*tu<<','<<!sliding
           <<','<<((1+Eta)*rddot-(total*er)-cf)*fu*1000<<'\n';
    }
};
