"""Long-horizon, exact-window rechecks of representative initial-state stripes."""
import importlib.util
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project_paths import RUNS_DIR, SCRATCH_DIR, extension_dir

source = np.load(RUNS_DIR / "phase-behavior" / "initial_2d_cpu_fp64.npz")
directory = extension_dir("fp64")
manifest = json.loads((directory / "build_config.json").read_text())
spec = importlib.util.spec_from_file_location("hysteresis", directory / manifest["binary"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
params = module.SolverParams()
rows = (38, 39, 43, 44, 48, 51, 56, 61)
times = (1500, 3000, 6000, 12000, 24000)
dt = float(source["dt"])
out = SCRATCH_DIR / "phase-behavior" / "stripe-exploration"
out.mkdir(parents=True, exist_ok=True)
result = []
start = time.perf_counter()
with (out / "long_horizon.csv").open("w", encoding="utf-8") as file:
    file.write("row,v0,s0,T,window,mean_v,mean_phase,delta_v,delta_phase,std_v,std_phase,status\n")
    for row in rows:
        v0 = np.asarray([source["v0"][36]], dtype=np.float64)
        s0 = np.asarray([source["s0"][row]], dtype=np.float64)
        for T in times:
            status, metrics, _ = module.scan_initial_grid(v0, s0, dt,
                round(T/dt), round(500/dt), round(0.05/dt),
                float(source["sr"]), float(source["somega"]),  1)
            m = metrics[0,0]
            record = (row,float(v0[0]),float(s0[0]),T,500,m[0],m[3],m[12],m[13],m[1],m[4],status[0,0])
            result.append(record)
            file.write(",".join(map(str,record))+"\n")
            file.flush()
            print(f"row={row} T={T} v={m[0]:.6f} phase={m[3]:.6f} dv={m[12]:.6f} elapsed={time.perf_counter()-start:.1f}s",flush=True)

fig, axes = plt.subplots(2,1,figsize=(9,7),sharex=True,layout="constrained")
for row in rows:
    records = [r for r in result if r[0]==row]
    label = f"S0={records[0][2]:.2f}"
    axes[0].plot(times,[r[5] for r in records],"o-",label=label)
    axes[1].plot(times,[r[6] for r in records],"o-",label=label)
axes[0].set_ylabel("mean tangential speed / m")
axes[1].set_ylabel("mean phase rate")
axes[1].set_xlabel("total integration time")
axes[0].legend(ncol=4,fontsize=8)
for ax in axes: ax.grid(alpha=.25)
fig.savefig(out / "long_horizon.png",dpi=180)
plt.close(fig)
