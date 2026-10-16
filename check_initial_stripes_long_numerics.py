"""Check long-time scan states and time-step sensitivity."""
import importlib.util
import json
import numpy as np
from project_paths import RUNS_DIR, SCRATCH_DIR, extension_dir

scan = np.load(RUNS_DIR / "phase-behavior" / "initial_2d_cpu_fp64.npz")
directory = extension_dir("fp64")
manifest = json.loads((directory / "build_config.json").read_text())
spec = importlib.util.spec_from_file_location("hysteresis", directory / manifest["binary"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
params = module.SolverParams()
out = SCRATCH_DIR / "phase-behavior" / "stripe-exploration" / "long_numerics.csv"
with out.open("w",encoding="utf-8") as file:
    file.write("row,s0,T,dt,status,mean_v,mean_phase,delta_v,delta_phase,final_r,quaternion_norm,final_q_norm,finite\n")
    for row in (38, 43, 51, 61):
        v0 = np.asarray([scan["v0"][36]],dtype=np.float64)
        s0 = np.asarray([scan["s0"][row]],dtype=np.float64)
        for dt in (0.001,0.0005):
            T=24000
            status,metrics,states = module.scan_initial_grid(v0,s0,dt,
                round(T/dt),round(500/dt),round(0.05/dt),
                float(scan["sr"]),float(scan["somega"]),1)
            m=metrics[0,0]; state=states[0,0]
            values=(row,float(s0[0]),T,dt,int(status[0,0]),m[0],m[3],m[12],m[13],
                np.linalg.norm(state[1:3]),np.linalg.norm(state[5:9]),
                np.linalg.norm(state[12:15]),bool(np.isfinite(state).all() and np.isfinite(m).all()))
            file.write(",".join(map(str,values))+"\n"); file.flush()
            print(values,flush=True)
