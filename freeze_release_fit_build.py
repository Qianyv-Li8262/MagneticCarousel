"""Freeze one diagnostic candidate into a normal constexpr solver build."""
import json
import argparse
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch'/'legacy-release-source'
OUT=ROOT/'runs'/'prepared-release-fit'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path)
    args=parser.parse_args()
    if args.candidate is None:
        rows=json.loads((OUT/'capture_escape.json').read_text(encoding='utf-8'))
        selected=next(r for r in rows if r['kind']=='capture' and r['candidate']==2)
    else:selected=json.loads(args.candidate.read_text(encoding='utf-8'))
    (OUT/'fixed_candidate.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    names=['Mu_r','Alpha_JA','C_JA','K','kTiltAngleRad','kTiltAxisAzimuthRad']
    header=(ROOT/'kernels'/'headers'/'hysteresis_config.hpp').read_text(encoding='utf-8')
    for name,value in zip(names,selected['settings']):
        header,count=re.subn(rf'(CONFIG_CONST real_t {name}\s*=\s*)REAL_LITERAL\([^)]*\)',
                             rf'\g<1>REAL_LITERAL({value:.17g})',header)
        assert count==1,name
    header=re.sub(r'(CONFIG_CONST real_t kTiltAxisAzimuthRad[^\n]*;)\s*//[^\n]*',r'\1 // radians',header)
    header=re.sub(r'(CONFIG_CONST real_t kTiltAngleRad[^\n]*;)\s*//[^\n]*',r'\1 // radians',header)
    (SOURCE/'kernels'/'headers'/'hysteresis_config.hpp').write_text(header,encoding='utf-8')
    (OUT/'candidate_config.hpp').write_text(header,encoding='utf-8')
    model=ROOT/'kernels'/'headers'/'2dmodel-hysteresis.hpp'
    (SOURCE/'kernels'/'headers'/model.name).write_text(model.read_text(encoding='utf-8'),encoding='utf-8')
    cpp=SOURCE/'kernels'/'2dmodel-hysteresis.cpp'
    held='State magnetize_held_ball'+cpp.read_text(encoding='utf-8').split('State magnetize_held_ball',1)[1]
    cpp.write_text((ROOT/'kernels'/'2dmodel-hysteresis.cpp').read_text(encoding='utf-8')+'\n'+held,encoding='utf-8')
    binding=SOURCE/'kernels'/'bindings'/'pybind.cpp'
    text=binding.read_text(encoding='utf-8')
    text=text.replace('void configure_release_model(real_t,real_t,real_t,real_t,real_t,real_t);','')
    text=text.replace('    m.def("configure_release_model", &configure_release_model);','')
    binding.write_text(text,encoding='utf-8')
    (SOURCE/'write_build_config.py').write_text((ROOT/'write_build_config.py').read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(dict(frozen_settings=selected['settings'],phase=selected['phase']),indent=2))


if __name__=='__main__':main()
