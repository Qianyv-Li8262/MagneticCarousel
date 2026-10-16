"""Factorial moment/conductivity comparisons and local perturbation checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"scratch/eddy_integration_20261016"
OUT=DATA/"restored_moment_sigma9_h32"
config=json.loads((OUT/"summary.json").read_text())["config"]
lut=config["lut"]
trace=np.load(OUT/"trace.npz");t=trace["time_2d"];s=trace["state"]
rows=np.column_stack((t,s,trace["blow_mN"]))
rows.astype("<f8").tofile(OUT/"drive_input.bin")
subprocess.run([str(DATA/"diagnose_drive.exe"),str(OUT/"drive_input.bin"),lut,str(OUT/"drive.csv")],check=True)
a=np.genfromtxt(OUT/"drive.csv",names=True,delimiter=",")
report={"current_windows":[]}
for lo,hi in [(100,150),(250,300),(350,400),(450,500),(550,600),(700,800),(1500,2000),(4500,5000)]:
    mask=(t>=lo-1e-7)&(t<hi)
    rec={key:float(a[key][mask].mean()) for key in a.dtype.names if key not in ("t2d","time_s")}
    rec["interval_t2d"]=[lo,hi]
    rec["wz_rad_s"]=float(s[mask,11].mean()*200*2*np.pi/60)
    rec["net_Mz_uNm"]=rec["JA_torque_z_uNm"]+rec["eddy_torque_z_uNm"]
    rec["net_drive_mN"]=rec["JA_drive_mN"]+rec["eddy_drive_mN"]-rec["roll_loss_mN"]-rec["air_loss_mN"]
    report["current_windows"].append(rec)

cases=[("moment_high_sigma12",.0854985,1.2e7,2000,None),
       ("moment_high_sigma9",.0854985,9e6,2000,None),
       ("moment_restored_sigma12",.0569993,1.2e7,2000,None),
       ("moment_restored_sigma9",.0569993,9e6,2000,None)]
state=s[np.argmin(abs(t-2000))].copy()
for name in ("baseline","speed_plus5pct","wz_plus5pct","radius_plus1mm"):
    initial=state.copy()
    if name=="speed_plus5pct":initial[3:5]*=1.05;initial[9:11]*=1.05
    if name=="wz_plus5pct":initial[11]*=1.05
    if name=="radius_plus1mm":initial[1:3]*=(np.linalg.norm(initial[1:3])+1/17.5)/np.linalg.norm(initial[1:3])
    path=OUT/f"kick_{name}.bin";initial.astype("<f8").tofile(path)
    cases.append(("kick_"+name,.0569993,9e6,1500,path))
def run(case):
    name,beta,sigma,duration,initial=case
    command=[str(DATA/"audit_restored_moment.exe"),lut,str(OUT/f"{name}.csv"),str(beta),str(sigma),str(duration)]
    if initial is not None:command.append(str(initial))
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    table=np.genfromtxt(OUT/f"{name}.csv",delimiter=",",names=True)
    tail=table[table["time"]>=table["time"][-1]-100]
    return name,{"beta":beta,"sigma_S_m":sigma,"run":result.stdout.strip(),"last_sample_time":float(table["time"][-1]),
        "peak_speed_mm_s":float(table["speed_mm_s"].max()),
        "tail":{key:float(tail[key].mean()) for key in table.dtype.names if key!="time"}}
with ThreadPoolExecutor(max_workers=4) as pool:
    report["isolated_comparisons"]=dict(pool.map(run,cases))
baseline=np.genfromtxt(OUT/"moment_restored_sigma9.csv",names=True,delimiter=",")
original=np.genfromtxt(OUT/"observables.csv",names=True,delimiter=",")[:len(baseline)]
report["native_replay_speed_max_difference_mm_s"]=float(np.max(abs(baseline["speed_mm_s"]-original["speed_mm_s"])))
(OUT/"restored_analysis.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
