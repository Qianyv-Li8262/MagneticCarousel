"""Resolve axial hysteresis torque into circular field harmonics and phase."""
from pathlib import Path
import json
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import animate_one_trajectory as animation
parser=argparse.ArgumentParser()
parser.add_argument('--label',default='sigma12_start_rest_no_blow_long')
args=parser.parse_args()
OUT=ROOT/"scratch/eddy_integration_20261016"/args.label
d=np.load(OUT/"trace.npz");t=d["time_2d"];s=d["state"]
a=np.genfromtxt(OUT/"drive.csv",delimiter=",",names=True)
meta=json.loads((OUT/"summary.json").read_text())["config"]
nodes=np.fromfile(meta["lut"],dtype="<f8").reshape(5,204,4)
r=np.linalg.norm(s[:,1:3],axis=1);phi=np.arctan2(s[:,2],s[:,1]);disk=s[:,0]
dr=4/203;x=(r-2)/dr;j=np.clip(np.floor(x).astype(int),0,202);u=x-j
beta=float(meta['model_parameters']['Beta'].removeprefix('REAL_LITERAL(').removesuffix(')'))
def spline(k,offset):
    left=nodes[k,j,offset]; Ml=nodes[k,j,offset+1]; right=nodes[k,j+1,offset]; Mr=nodes[k,j+1,offset+1]
    b=right-left-dr*dr*(2*Ml+Mr)/6;c=dr*dr*Ml/2;dd=dr*dr*(Mr-Ml)/6
    return left+u*(b+u*(c+u*dd)),(b+u*(2*c+3*dd*u))/dr
q=s[:,5:9];v=s[:,12:15];im=q[:,1:]
qworld=v+2*q[:,0,None]*np.cross(im,v)+2*np.cross(im,np.cross(im,v))
moment=-(qworld[:,0]+1j*qworld[:,1])
hxy=np.zeros(len(t),complex);hz=np.zeros(len(t))
positive=[];negative=[]
for k in range(5):
    A,Ap=spline(k,0);Z,_=spline(k,2);m=4*(2*k+1)
    plus=beta*(-Ap-m*A/r)/2*np.exp(1j*(m*disk-(m-1)*phi))
    minus=beta*(-Ap+m*A/r)/2*np.exp(1j*(-m*disk+(m+1)*phi))
    positive.append(plus);negative.append(minus);hxy+=plus+minus
    hz+=-beta*Z*np.cos(m*(disk-phi))
np.testing.assert_allclose(np.sqrt(abs(hxy)**2+hz**2),a["h_norm"],rtol=1e-11,atol=1e-12)
mod=animation.load_hysteresis();p=mod.SolverParams(sr=1,somega=2)
mu=605.8974337495617;alpha=.0021323083849413885;c=.22415154978863677
xi=alpha*(mu-1)/(1+alpha*(mu-1));E=(1-c)/((1-xi)*(mu+2))
O=200*2*np.pi/60;I=.4*(7850*4*np.pi*.006**3/3)*.006**2
scale=I*O*O*p.gamma_m*E*1e6
torque=lambda h:scale*np.imag(np.conj(moment)*h)
np.testing.assert_allclose(torque(hxy),a["JA_torque_z_uNm"],rtol=1e-8,atol=1e-8)
records=[]
for lo,hi in [(250,300),(350,400),(400,450),(500,550),(600,650),(700,750),(750,775),(775,800),(800,810),(810,820),(820,830),(830,840),(840,850),(850,860),(860,875),(875,900),(1500,2000),(4500,5000)]:
    w=(t>=lo-1e-8)&(t<hi)
    if not np.any(w):continue
    pos=positive[0][w];pm=moment[w]
    demod=pm*np.conj(pos)/abs(pos)
    mean_demod=np.mean(demod)
    weighted=np.mean(pm*np.conj(pos))
    weighted_negative=np.mean(pm*np.conj(negative[0][w]))
    rec={"t_interval":[lo,hi],"radius_mm":float(r[w].mean()*17.5),
        "fund_positive_h_amp":float(abs(pos).mean()),"fund_negative_h_amp":float(abs(negative[0][w]).mean()),
        "moment_xy_amp":float(abs(pm).mean()),"coherent_moment_amp":float(abs(mean_demod)),
        "coherent_phase_deg":float(np.angle(mean_demod)*180/np.pi),
        "weighted_phase_deg":float(np.angle(weighted)*180/np.pi),
        "coherent_torque_scale_uNm":float(scale*abs(weighted)),
        "negative_coherent_phase_deg":float(np.angle(weighted_negative)*180/np.pi),
        "negative_coherent_torque_scale_uNm":float(scale*abs(weighted_negative)),
        "plus_fund_torque_uNm":float(torque(positive[0])[w].mean()),
        "minus_fund_torque_uNm":float(torque(negative[0])[w].mean()),
        "higher_harmonic_torque_uNm":float((torque(hxy)-torque(positive[0])-torque(negative[0]))[w].mean()),
        "torque_angle_efficiency":float(np.mean(torque(positive[0])[w])/(scale*np.mean(abs(pm)*abs(pos)))),
        "wz_std_detrended_rad_s":float(np.std((s[w,11]-np.polyval(np.polyfit(t[w],s[w,11],1),t[w]))*O))}
    records.append(rec)
np.savez_compressed(OUT/"capture_phase.npz",time_2d=t,positive_h=positive[0],negative_h=negative[0],
    demodulated_memory=moment*np.conj(positive[0])/abs(positive[0]),
    plus_torque=torque(positive[0]),minus_torque=torque(negative[0]),scale=scale)
(OUT/"capture_loss_phase.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
