"""Frozen-orbit conductivity estimate, calibrated to the sigma=6.67 MS/m replay.

This does not change production parameters or integrate a new coupled orbit.
Uses the exact conducting-sphere response, all five cylinder-field harmonics,
and rolling/normal-spin frequency sidebands.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"scratch/eddy_integration_20261016"
OUT=DATA/"conductivity_estimate"
OUT.mkdir(exist_ok=True)
BASE=DATA/"eddy_on_blow_200_300_long"
summary=json.loads((BASE/"summary.json").read_text())
budget=json.loads((BASE/"coast_force_summary.json").read_text())
config=summary["config"]
lut_path=Path(config["lut"])
assert hashlib.sha256(lut_path.read_bytes()).hexdigest()==config["lut_sha256"]
nodes=np.fromfile(lut_path,dtype="<f8").reshape(5,204,4)
trace=np.load(BASE/"trace.npz")
selected=(trace["time_2d"]>=300-1e-8)&(trace["time_2d"]<340)
states=trace["state"][selected]
pi=np.pi; mu0=4*pi*1e-7; R=.006; V=4*pi*R**3/3; Mref=1.1e6
k=4/.07; Om=200*2*pi/60; sigma0=1/(.15e-6); muroll=.0049
literal=lambda key:float(config["model_parameters"][key].removeprefix("REAL_LITERAL(").removesuffix(")"))
mu=literal("Mu_r"); beta=literal("Beta")
radius=float(np.mean(np.linalg.norm(states[:,1:3],axis=1)))/k
speed=float(np.mean(np.linalg.norm(states[:,3:5],axis=1)))*Om/k
wz=float(np.mean(states[:,11]))*Om

def spline(left,right,u,dr):
    a=left[0]; b=right[0]-left[0]-dr*dr*(2*left[1]+right[1])/6
    c=dr*dr*left[1]/2; d=dr*dr*(right[1]-left[1])/6
    return a+u*(b+u*(c+u*d)),(b+u*(2*c+3*d*u))/dr,(2*c+6*d*u)/dr**2

def harmonics():
    rho=radius*k; dr=4/203; x=(rho-2)/dr; j=min(202,max(0,int(np.floor(x)))); u=x-j
    result=[]
    for index in range(5):
        A,Ap,App=spline(nodes[index,j,:2],nodes[index,j+1,:2],u,dr)
        Z,Zp,_=spline(nodes[index,j,2:],nodes[index,j+1,2:],u,dr)
        m=4*(2*index+1)
        b=np.array([-Ap,1j*m*A/rho,-Z])*beta*Mref*mu0
        J=np.array([[-App,1j*m*(Ap/rho-A/rho**2),-Zp],
             [1j*m*(Ap/rho-A/rho**2),m*m*A/rho**2-Ap/rho,1j*m*Z/rho],
             [-Zp,1j*m*Z/rho,App-(m*m*A/rho**2-Ap/rho)]])*beta*Mref*mu0*k
        result.append((m,b,J))
    return result

def polarizability(f,sigma):
    if abs(f)<1e-10 or sigma==0:return 3*V*(mu-1)/(mu+2)
    x=R*np.sqrt(-1j*f*mu0*mu*sigma)
    return 1.5*V*((2*mu+1)*spherical_jn(1,x)-x*spherical_jn(0,x))/((mu-1)*spherical_jn(1,x)+x*spherical_jn(0,x))

orbit=-speed/radius
w=np.array([speed/R,0,wz-orbit]); W=np.linalg.norm(w); u=w/W
C=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
P0=np.outer(u,u); Pp=(np.eye(3)-P0+1j*C)/2; Pm=(np.eye(3)-P0-1j*C)/2
tones=harmonics()
def response(sigma):
    F=np.zeros(3); T=np.zeros(3)
    for m,b,J in tones:
        freq=m*(Om-orbit); dp=np.zeros(3,dtype=complex)
        for P,f in [(P0,freq),(Pp,freq+W),(Pm,freq-W)]:
            dp+=(polarizability(f,sigma)-polarizability(0,0))*(P@b)/mu0
        F+=.5*np.real(J.T@np.conj(dp)); T+=.5*np.real(np.cross(dp,np.conj(b)))
    return np.array([(-F[1]+T[0]/R)*1000,-muroll*F[2]*1000])

raw0=response(sigma0)
# Correct the 0.44% mean-orbit approximation against the actual sampled drive.
anchor=budget["eddy_drive_mN"]/raw0[0]
def evaluate(sigma):
    raw=response(sigma)
    q=float(anchor*raw[0])
    loss=budget["roll_loss_mN"]+budget["air_loss_mN"]+anchor*(raw[1]-raw0[1])
    return {"sigma_S_m":sigma,"resistivity_microohm_m":1e6/sigma,"eddy_drive_mN":q,
        "loss_mN":loss,"net_with_original_model_mN":budget["JA_drive_mN"]+q-loss,
        "net_with_memory_only_mN":budget["hysteresis_drive_mN"]+q-loss}

local=brentq(lambda s:evaluate(s)["net_with_original_model_mN"],sigma0,4e7)
local_margin=brentq(lambda s:evaluate(s)["net_with_original_model_mN"]-.02,sigma0,4e7)
memory=brentq(lambda s:evaluate(s)["net_with_memory_only_mN"],sigma0,4e7)
memory_margin=brentq(lambda s:evaluate(s)["net_with_memory_only_mN"]-.02,sigma0,4e7)
report={"assumptions":"Fixed mean radius, COM speed and normal spin from the old sigma=6.67 MS/m coupled replay. Original JA or memory contribution held fixed. Includes change of rolling loss from eddy normal force. Does not prove a self-sustained coupled orbit.",
    "radius_mm":radius*1000,"speed_mm_s":speed*1000,"normal_spin_rad_s":wz,
    "baseline_sigma_S_m":sigma0,"raw_model_baseline_drive_mN":float(raw0[0]),"calibration_factor":anchor,
    "local_balance_sigma_S_m":local,"local_plus_0p02_mN_sigma_S_m":local_margin,
    "memory_only_balance_sigma_S_m":memory,"memory_only_plus_0p02_mN_sigma_S_m":memory_margin,
    "linear_extrapolation_sigma_S_m":sigma0*(budget["roll_loss_mN"]+budget["air_loss_mN"]-budget["JA_drive_mN"])/budget["eddy_drive_mN"],
    "table":[evaluate(s) for s in [sigma0,1e7,1.1e7,1.2e7,1.6e7,1.7e7,1.8e7,2e7]]}
(OUT/"estimate.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

scan=np.linspace(1e5,2.2e7,240)
q=np.array([evaluate(s)["eddy_drive_mN"] for s in scan])
np.savetxt(OUT/"scan.csv",np.column_stack((scan,q)),delimiter=",",header="sigma_S_m,eddy_drive_mN",comments="")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],"font.size":11,
    "axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(9,5.4),layout="constrained")
ax.plot(scan/1e7,q,color="#2563eb",linewidth=2,label="导电球频率响应估算（按原轨迹校准）")
ax.plot(scan/1e7,budget["eddy_drive_mN"]*scan/sigma0,color="#94a3b8",linestyle="--",label="从旧基准直接线性外推")
ax.axhline(budget["roll_loss_mN"]+budget["air_loss_mN"]-budget["JA_drive_mN"],color="#ea580c",linestyle=":",label="原模型其余贡献不变：所需涡流驱动")
ax.axhline(budget["roll_loss_mN"]+budget["air_loss_mN"]-budget["hysteresis_drive_mN"],color="#059669",linestyle=":",label="仅计涡流与磁滞记忆：所需涡流驱动")
ax.scatter([local/1e7,memory/1e7],[evaluate(local)["eddy_drive_mN"],evaluate(memory)["eddy_drive_mN"]],color=["#ea580c","#059669"],zorder=5)
ax.set(title="涡流广义驱动随电导率增加，但此区间已偏离线性",xlabel="电导率 σ (10⁷ S/m)",ylabel="涡流广义驱动 (mN)",xlim=(0,2.2),ylim=(0,.66))
ax.legend(loc="upper left",fontsize=9,frameon=False)
ax.grid(alpha=.2)
fig.savefig(OUT/"conductivity_drive.png",dpi=160)
