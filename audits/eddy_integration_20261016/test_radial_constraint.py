"""Physical diagnostic: an ideal fixed-radius guide after t2D=500.

Project COM velocity tangentially and restore pure rolling after each short
interval. This adds an ideal radial constraint; it is not a production change.
Check interval sensitivity because finite projection removes radial kinetic
energy, which must tend to zero with the interval.
"""
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import animate_one_trajectory as a
OUT=ROOT/"scratch/eddy_integration_20261016/sigma12_start_rest_no_blow_long"
d=np.load(OUT/"trace.npz")
index=np.argmin(abs(d["time_2d"]-500));initial=d["state"][index].copy()
module=a.load_hysteresis();params=module.SolverParams(sr=1,somega=2)
O=module.physical_units(sr=1,somega=2)["disk_omega_rad_s"]
radius=np.linalg.norm(initial[1:3]);report={}
for interval_steps in (20,10,2):
    state=initial.copy();sliding=False;direction=np.zeros(2)
    rows=[];removed=0
    span=interval_steps*.0005
    for n in range(round(500/span)+1):
        time=500+n*span
        if n % round(.05/span)==0:
            rows.append([time,np.linalg.norm(state[3:5])*17.5*O,state[11]*O,
                         np.linalg.norm(state[1:3])*17.5])
        if time>=1000-1e-8:break
        state,sliding,direction=module.evolve_trajectory(state,.0005,interval_steps,params,sliding,direction,n*interval_steps)
        radial=state[1:3]/np.linalg.norm(state[1:3])
        vr=float(np.dot(state[3:5],radial))
        removed+=.5*(1+.4)*vr*vr
        state[1:3]=radius*radial
        state[3:5]-=vr*radial
        state[9]=-state[4]/params.alpha;state[10]=state[3]/params.alpha
        if sliding:raise RuntimeError("guide diagnostic left rolling regime")
    table=np.asarray(rows)
    np.savetxt(OUT/f"radial_guide_{interval_steps}.csv",table,delimiter=",",
        header="time_2d,speed_mm_s,wz_rad_s,radius_mm",comments="")
    report[str(interval_steps)]={"radius_mm":float(radius*17.5),"final_speed_mm_s":float(table[-1,1]),
        "final_wz_rad_s":float(table[-1,2]),"removed_radial_kinetic_dimensionless":removed}
(OUT/"radial_guide_summary.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
