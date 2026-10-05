"""Add a real sustained COM force to the isolated CPU solver, not velocity kicks.

The force enters every RHS stage and every contact-mode force evaluation.
All production sources, geometry and magnetic/mechanical constants stay intact.
"""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch'/'legacy-release-source'


def main():
    header=SOURCE/'kernels'/'headers'/'2dmodel-hysteresis.hpp'
    text=(ROOT/'kernels'/'headers'/'2dmodel-hysteresis.hpp').read_text(encoding='utf-8')
    replacements=[('field_force.Force','in.pos'),('field.Force','state.pos'),
       ('field_force.Force','state_at_tc.pos'),('field_force_final.Force','final_state.pos'),
       ('field_force_final.Force','trial_final.pos'),('field_force_event.Force','state_at_tr.pos'),
       ('field_force_sliding_mid.Force','sliding_mid.pos')]
    calls=list(re.finditer(r'apply_inclined_disk_force\(params, ([^)]*)\)',text))
    assert len(calls)==len(replacements),(len(calls),len(replacements))
    for match,(expected,pos) in reversed(list(zip(calls,replacements))):
        assert match.group(1)==expected,(match.group(1),expected)
        text=text[:match.start()]+f'apply_inclined_disk_force(params, {expected}, {pos})'+text[match.end():]
    text=text.replace('DEVICE INLINE Vec3 apply_inclined_disk_force(const SolverParams &params, Vec3 force)',
       '''// External air-push diagnostic, CPU only. Acceleration is in solver units.
inline real_t gAirPushAcceleration = REAL_LITERAL(0.0);
inline bool gAirPushTangential = true;
inline Vec2 gAirPushFixedDirection{};

DEVICE INLINE Vec3 apply_inclined_disk_force(const SolverParams &params, Vec3 force, const Vec2 &position)''')
    marker='''    return force;
}

DEVICE INLINE auto eval_body_field_derivative'''
    assert marker in text
    text=text.replace(marker,'''    if (gAirPushAcceleration != REAL_LITERAL(0.0)) {
        const real_t radius=length(position);
        const Vec2 direction=gAirPushTangential
            ? make_Vec2(position.y/radius,-position.x/radius) : gAirPushFixedDirection;
        force.x += gAirPushAcceleration*direction.x;
        force.y += gAirPushAcceleration*direction.y;
    }
    return force;
}

DEVICE INLINE auto eval_body_field_derivative''',1)
    header.write_text(text,encoding='utf-8')
    cpp=SOURCE/'kernels'/'2dmodel-hysteresis.cpp'
    old=cpp.read_text(encoding='utf-8')
    old=old.split('// External push setter',1)[0]
    old+='''
// External push setter. A COM force adds no direct torque and no normal load.
void configure_air_push(real_t acceleration,bool tangential,real_t dx,real_t dy) {
    if (!real_isfinite(acceleration+dx+dy) || acceleration<REAL_LITERAL(0.0))
        throw std::invalid_argument("invalid air-push acceleration");
    Vec2 direction=make_Vec2(dx,dy);
    if (!tangential && acceleration>REAL_LITERAL(0.0)) {
        const real_t magnitude=length(direction);
        if (!(magnitude>REAL_LITERAL(0.0))) throw std::invalid_argument("zero push direction");
        direction=direction/magnitude;
    }
    gAirPushAcceleration=acceleration;
    gAirPushTangential=tangential;
    gAirPushFixedDirection=direction;
}
'''
    cpp.write_text(old,encoding='utf-8')
    binding=SOURCE/'kernels'/'bindings'/'pybind.cpp'
    text=binding.read_text(encoding='utf-8')
    if 'void configure_air_push' not in text:
        text=text.replace('namespace py = pybind11;',
                          'namespace py = pybind11;\nvoid configure_air_push(real_t,bool,real_t,real_t);')
        text=text.replace('    m.doc() = ',
             '    m.def("configure_air_push", &configure_air_push, py::arg("acceleration"), '
             'py::arg("tangential")=true, py::arg("dx")=0., py::arg("dy")=0.);\n    m.doc() = ',1)
    binding.write_text(text,encoding='utf-8')
    print(SOURCE)


if __name__=='__main__':main()
