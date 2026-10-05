"""Enable parameter-only runtime scans in the isolated copy of the solver.

The production RHS/contact equations remain unchanged. Added helpers hold the
ball for magnetic initialization and choose the existing zero-slip contact mode.
Candidate parameters must finally be replayed with a normal constexpr build.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parent
source = root/'scratch'/'legacy-release-source'
header = source/'kernels'/'headers'/'hysteresis_config.hpp'
base = (root/'runs'/'legacy-release-capture'/'legacy_base'/'hysteresis_config.hpp').read_text(encoding='utf-8')
for key in ['Mu_r','Alpha_JA','C_JA','K','Xi_JA','kTiltAxisAzimuthRad','kTiltAngleRad']:
    base, count = re.subn(rf'CONFIG_CONST real_t {key}\b',rf'inline real_t {key}',base)
    assert count == 1, key
header.write_text(base,encoding='utf-8')
model_header=source/'kernels'/'headers'/'2dmodel-hysteresis.hpp'
text=(root/'kernels'/'headers'/'2dmodel-hysteresis.hpp').read_text(encoding='utf-8')
text=text.replace('constexpr real_t E_', 'const real_t E_').replace('constexpr real_t E ', 'const real_t E ')
text=text.replace('constexpr real_t D_', 'const real_t D_').replace('constexpr real_t c = Xi_JA', 'const real_t c = Xi_JA')
text=text.replace('constexpr real_t a = REAL_LITERAL(1.0) - C_JA', 'const real_t a = REAL_LITERAL(1.0) - C_JA')
model_header.write_text(text,encoding='utf-8')
writer=source/'write_build_config.py'
text=(root/'write_build_config.py').read_text(encoding='utf-8')
text=text.replace('CONFIG_CONST\\s+{kind}', '(?:CONFIG_CONST|inline)\\s+{kind}')
writer.write_text(text,encoding='utf-8')
cpp=source/'kernels'/'2dmodel-hysteresis.cpp'
text=(root/'kernels'/'2dmodel-hysteresis.cpp').read_text(encoding='utf-8')
text=text.replace('constexpr real_t E =', 'const real_t E =')
text+='''

// Parameter-only diagnostic API. Use sequentially: model parameters are globals.
void configure_release_model(real_t mur, real_t aja, real_t cja, real_t khys,
                             real_t tilt, real_t azimuth) {
    if (!(mur > 1 && aja > 0 && cja > 0 && cja < 1 && khys > 0) ||
        !real_isfinite(mur+aja+cja+khys+tilt+azimuth))
        throw std::invalid_argument("invalid diagnostic parameters");
    Mu_r=mur; Alpha_JA=aja; C_JA=cja; K=khys;
    Xi_JA=REAL_LITERAL(1.0)-REAL_LITERAL(1.0)/(REAL_LITERAL(1.0)+aja*(mur-REAL_LITERAL(1.0)));
    kTiltAngleRad=tilt; kTiltAxisAzimuthRad=azimuth;
}

State magnetize_held_ball(const State &initial, real_t dt, int steps, const SolverParams &params) {
    validate_params(params);
    if (!(dt>0) || steps<0) throw std::invalid_argument("invalid magnetic preparation duration");
    State state=initial;
    state.speed=make_Vec2(REAL_LITERAL(0.0),REAL_LITERAL(0.0));
    state.omega=make_Vec3(REAL_LITERAL(0.0),REAL_LITERAL(0.0),REAL_LITERAL(0.0));
    const real_t start_angle=static_cast<real_t>(state.down_disk_angle);
    const Vec2 zero=make_Vec2(REAL_LITERAL(0.0),REAL_LITERAL(0.0));
    FieldCache cache{};
    for (int i=0;i<steps;++i) {
        state.down_disk_angle=start_angle+static_cast<real_t>(i)*dt;
        State first{},second{};
        if (!eval_rhs<false>(state,first,params,false,zero,&cache,trajectory_lut()))
            throw std::runtime_error("magnetic preparation lost contact");
        State mid=state;
        mid.q=state.q+first.q*(dt*REAL_LITERAL(0.5));
        mid.down_disk_angle=start_angle+(static_cast<real_t>(i)+REAL_LITERAL(0.5))*dt;
        if (!eval_rhs<false>(mid,second,params,false,zero,&cache,trajectory_lut()))
            throw std::runtime_error("magnetic preparation lost contact");
        state.q=state.q+second.q*dt;
        if (!real_isfinite(length(state.q))) throw std::runtime_error("magnetic preparation diverged");
    }
    state.down_disk_angle=start_angle+static_cast<real_t>(steps)*dt;
    return state;
}

std::pair<bool,Vec2> release_contact_mode(const State &state,const SolverParams &params) {
    bool sliding=false; Vec2 direction{}; FieldCache cache{};
    const auto result=resolve_zero_slip_mode(state,params,sliding,direction,&cache,trajectory_lut());
    if (result!=AdvanceStepResult::kStepOk) throw std::runtime_error("invalid initial contact mode");
    return {sliding,direction};
}
'''
cpp.write_text(text,encoding='utf-8')
binding=source/'kernels'/'bindings'/'pybind.cpp'
text=(root/'kernels'/'bindings'/'pybind.cpp').read_text(encoding='utf-8')
text=text.replace('namespace py = pybind11;', '''namespace py = pybind11;
void configure_release_model(real_t,real_t,real_t,real_t,real_t,real_t);
State magnetize_held_ball(const State&,real_t,int,const SolverParams&);
std::pair<bool,Vec2> release_contact_mode(const State&,const SolverParams&);
''')
insert='''
    m.def("configure_release_model", &configure_release_model);
    m.def("magnetize_held_ball", [](const py::array_t<real_t,py::array::c_style|py::array::forcecast>& a,
                                    real_t dt,int steps,const SolverParams& p) {
        return state_to_numpy(magnetize_held_ball(state_from_numpy(a),dt,steps,p));
    });
    m.def("release_contact_mode", [](const py::array_t<real_t,py::array::c_style|py::array::forcecast>& a,
                                     const SolverParams& p) {
        const auto mode=release_contact_mode(state_from_numpy(a),p);
        py::array_t<real_t> direction(2);
        direction.mutable_at(0)=mode.second.x; direction.mutable_at(1)=mode.second.y;
        return py::make_tuple(mode.first,direction);
    });
'''
text=text.replace('    m.doc() = ',insert+'    m.doc() = ',1)
binding.write_text(text,encoding='utf-8')
print(source)
