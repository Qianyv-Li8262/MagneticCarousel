"""Recompute selected initial-scan pixels at half time step."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from project_paths import extension_dir, RUNS_DIR, SCRATCH_DIR

source = np.load(RUNS_DIR / "phase-behavior" / "initial_2d_cpu_fp64.npz")
directory = extension_dir("fp64")
binary = directory / json.loads((directory / "build_config.json").read_text())["binary"]
spec = importlib.util.spec_from_file_location("hysteresis", binary)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
params = module.SolverParams()
rows = (38, 39, 43, 44, 48, 51, 56, 61)
out = SCRATCH_DIR / "phase-behavior" / "stripe-exploration" / "dt_check.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as file:
    file.write("row,s0,dt,mean_v,mean_phase,delta_v,delta_phase,status\n")
    for row in rows:
        v0 = np.asarray([source["v0"][36]], dtype=np.float64)
        s0 = np.asarray([source["s0"][row]], dtype=np.float64)
        for dt in (0.001, 0.0005):
            status, metrics, _ = module.scan_initial_grid(v0, s0, dt,
                round(1500/dt), round(500/dt), round(0.05/dt),
                float(source["sr"]), float(source["somega"]),  1)
            metric = metrics[0,0]
            line = (row, float(s0[0]), dt, metric[0], metric[3], metric[12], metric[13], status[0,0])
            file.write(",".join(map(str,line))+"\n")
            file.flush()
            print(line, flush=True)
