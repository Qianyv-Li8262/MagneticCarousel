"""Prescribed fixed-centre, fixed-attitude field history at CURRENT point-field Beta.
No mechanical trajectory is integrated; no production parameter is modified.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

MS=1200019.0724085916; AA=1514.216550463362; KP=5913.52056037147
C=.22415154978863677; AL=.0021323083849413885; D=1/3-AL; S=MS/(3*AA)
BETA=.0854985; MREF=1.1e6; KSPACE=4/.07; V=4*np.pi*.006**3/3; MU0=4*np.pi*1e-7

def lang(x):
    if abs(x)<.05: return x/3-x**3/45+2*x**5/945-x**7/4725+2*x**9/93555
    return 1/np.tanh(x)-1/x

def prime(x):
    if abs(x)<.05: return 1/3-x*x/15+2*x**4/189-x**6/675+2*x**8/10395
    return 1/x**2-(1/np.sinh(x)**2 if abs(x)<350 else 0)

def constitutive(H,Q,full):
    b=H+D*(1-C)*Q; bn=np.linalg.norm(b); u=b/bn if bn else np.array([1.,0,0])
    he=bn/(1+D*S)
    if full and bn:
        lo=0.; hi=bn
        for _ in range(40):
            value=he+D*MS*lang(he/AA)-bn
            if abs(value)<1e-9+1e-12*bn: break
            if value>0: hi=he
            else: lo=he
            new=he-value/(1+D*MS/AA*prime(he/AA))
            he=new if lo<new<hi else (lo+hi)/2
        else: raise RuntimeError('constitutive convergence')
    man=(MS*lang(he/AA) if full else S*he)*u
    jp=MS/AA*prime(he/AA) if full else S
    jt=MS*lang(he/AA)/he if full and he else S
    return man-(1-C)*Q,u,jp,jt,he/AA

def field(phase):
    theta=np.arange(8)*np.pi/4+phase/4
    d=np.c_[4-4*np.cos(theta),-4*np.sin(theta),np.full(8,2.264)]
    rr=np.sum(d*d,axis=1); sg=(-1.)**np.arange(8)
    h=np.sum((sg/rr**2.5)[:,None]*(3*d[:,2,None]*d-rr[:,None]*np.array([0,0,1])),axis=0)*BETA*MREF
    J=np.zeros((3,3)); z=np.array([0.,0,1.])
    for v,r,s in zip(d,rr,sg):
        J+=3*s/r**3.5*(r*(np.outer(v,z)+np.outer(z,v)+v[2]*np.eye(3))-5*v[2]*np.outer(v,v))
    return h,J*BETA*MREF*KSPACE

phase=np.linspace(0,2*np.pi,513); fields=np.array([field(t)[0] for t in phase]); fields[-1]=fields[0]
spline=CubicSpline(phase,fields,bc_type='periodic')

def rhs(t,Q,full):
    H=spline(t%(2*np.pi)); Hd=spline(t%(2*np.pi),1)
    _,u,jp,jt,_=constitutive(H,Q,full)
    def inv(v):
        par=(u@v)*u
        return (v-par)/(1+D*C*jt)+par/(1+D*C*jp)
    hd=inv(Hd); qn=np.linalg.norm(Q); active=Q@hd>0
    if active:
        n=Q/qn; w=inv(n); lam=D*(1-C)*qn/KP
        hd-=lam*w*(n@hd)/(1+lam*(n@w))
    qd=jt*hd+(jp-jt)*(u@hd)*u
    if active: qd-=qn/KP*n*(n@hd)
    return qd

records=[]; series=[]; end=128*2*np.pi
for full in [False,True]:
    sol=solve_ivp(lambda t,Q:rhs(t,Q,full),[0,end],np.zeros(3),rtol=2e-9,atol=2e-5,max_step=.06,dense_output=True)
    assert sol.success
    t=np.linspace(end-2*np.pi,end,1025); Q=sol.sol(t).T
    force=[]; torque=[]; xs=[]; mags=[]
    for tt,q in zip(t,Q):
        h,J=field(tt%(2*np.pi)); m,_,_,_,xx=constitutive(h,q,full)
        force.append(MU0*V*J@m); torque.append(MU0*V*np.cross(m,h)); xs.append(xx); mags.append(m)
    force=np.array(force); torque=np.array(torque)
    rms=lambda a:float(np.sqrt(np.mean(np.sum(a*a,axis=1))))
    rec=dict(model='Langevin' if full else 'linear',max_He_over_a=max(xs),max_M_over_Ms=float(np.max(np.linalg.norm(mags,axis=1))/MS),force_rms_N=rms(force),torque_rms_Nm=rms(torque),q_cycle_mismatch_A_m=float(np.linalg.norm(Q[-1]-Q[0])))
    records.append(rec); series.append((force,torque)); print(json.dumps(rec),flush=True)

out=dict(protocol='Current Beta=0.0854985, exact point dipoles, H=2.264, radius=70mm, ball diameter=12mm; prescribed rotating source, fixed centre and attitude, 128 field periods; no mechanical integration',records=records,
         max_external_H_A_m=float(np.max(np.linalg.norm(fields,axis=1))),
         force_rms_change=records[1]['force_rms_N']/records[0]['force_rms_N']-1,
         torque_rms_change=records[1]['torque_rms_Nm']/records[0]['torque_rms_Nm']-1,
         relative_force_waveform_error=rms(series[1][0]-series[0][0])/rms(series[0][0]),
         relative_torque_waveform_error=rms(series[1][1]-series[0][1])/rms(series[0][1]))
x=records[1]['max_He_over_a']; out['anhysteretic_curve_relative_suppression_at_max_x']=1-lang(x)/(x/3)
Path(__file__).with_name('current_langevin_scale.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
