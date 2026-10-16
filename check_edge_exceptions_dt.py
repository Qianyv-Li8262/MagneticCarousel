"""Half-step validation of the four edge exceptions at the scan horizon."""
import importlib.util
import json
import numpy as np
from project_paths import RUNS_DIR,SCRATCH_DIR,extension_dir

scan=np.load(RUNS_DIR/"phase-behavior"/"initial_2d_cpu_fp64.npz")
directory=extension_dir("fp64")
binary=directory/json.loads((directory/"build_config.json").read_text())["binary"]
spec=importlib.util.spec_from_file_location("hysteresis",binary)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
params=module.SolverParams()
out=SCRATCH_DIR/"phase-behavior"/"edge-exceptions"/"dt_check.csv"
with out.open("w",encoding="utf-8") as file:
    file.write("row,s0,T,dt,status,mean_v,mean_phase,delta_v,delta_phase,mean_radius,quaternion_norm,finite\n")
    for row in (1,2,60,63):
        v0=np.asarray([scan["v0"][-1]],dtype=np.float64)
        s0=np.asarray([scan["s0"][row]],dtype=np.float64)
        for T,dt in ((7000,.001),(7000,.0005),(24000,.001),(24000,.0005)):
            status,metrics,states=module.scan_initial_grid(v0,s0,dt,
                round(T/dt),round(500/dt),round(.05/dt),
                float(scan["sr"]),float(scan["somega"]),1)
            x=metrics[0,0]; state=states[0,0]
            record=(row,float(s0[0]),T,dt,int(status[0,0]),x[0],x[3],x[12],x[13],x[19],
                np.linalg.norm(state[5:9]),bool(np.isfinite(x).all() and np.isfinite(state).all()))
            file.write(",".join(map(str,record))+"\n");file.flush()
            print(record,flush=True)
