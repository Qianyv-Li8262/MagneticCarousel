from pathlib import Path
import sys,json
sys.path.insert(0,'src2d/audits/eddy_integration_20261016');import fit_video56c6 as f
OUT=f.ROOT/'runs/video56c6-rest-release-20261017';OUT.mkdir(parents=True,exist_ok=True)
B=json.loads((f.OUT/'startup_candidate.json').read_text())['parameters'];f.OUT=OUT
base=dict(B,wz=0,hold=0,vt=0,vr=0,qz=0)
cases=[('rest_r'+str(r),dict(base,r=r)) for r in [66.5,67.4,68.5,70]]
cases += [('remove_hold_only',dict(B,hold=0)),('remove_spin_only',dict(B,wz=0))]
rs=f.batch(cases);(OUT/'initial_checks.json').write_text(json.dumps(rs,indent=2))
(OUT/'constraints.json').write_text(json.dumps(dict(initial_velocity_mm_s=[0,0],initial_omega_rad_s=[0,0,0],initial_q=[0,0,0],initial_eddy_modes=0,held_magnetization_turns=0,height_mm=32,priority='startup curve',original_B=B),indent=2))
