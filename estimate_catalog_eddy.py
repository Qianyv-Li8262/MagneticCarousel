"""Postprocess saved catalog states; no trajectories are rerun.

Frequencies are dimensionless (disk angular time). Physical examples assume
reference radius 1 mm and conductivity 1e6 S/m, not measured properties.
"""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CAT = ROOT / 'scratch/phase-behavior/cpu-region-probe/catalog/cases.json'
OUT = ROOT / 'results/eddy-estimate'
MU_R, AJ, C, K = 2292.6667, .0032, .38, .0010909091
XI = 1 - 1 / (1 + AJ * (MU_R - 1))
D = (MU_R - 1) / (MU_R + 2)
E = (1 - C) / ((1 - XI) * (MU_R + 2))

def rotate(q, v, inverse=False):
    q = q / np.linalg.norm(q, axis=1)[:, None]
    u = q[:, 1:] * (-1 if inverse else 1)
    t = 2 * np.cross(u, v)
    return v + q[:, :1] * t + np.cross(u, t)

def evaluate(s):
    h = np.zeros((len(s), 3)); jac = np.zeros((len(s), 3, 3))
    for j in range(8):
        ang = s[:, 0] + j * np.pi / 4
        d = np.column_stack((s[:, 1]-4*np.cos(ang), s[:, 2]-4*np.sin(ang), np.full(len(s),1.610066235)))
        r2 = np.sum(d*d, axis=1); z = d[:, 2]; coef = .079370775450601 * (-1)**j
        h += coef * (3*z[:,None]*d - r2[:,None]*np.array([0,0,1])) / r2[:,None]**2.5
        for a in range(3):
            for b in range(3):
                numerator = (d[:,a]*(b==2)+d[:,b]*(a==2)+z*(a==b))*r2-5*z*d[:,a]*d[:,b]
                jac[:,a,b] += 3*coef*numerator/r2**3.5
    rel = np.column_stack((s[:,3]+s[:,2],s[:,4]-s[:,1],np.zeros(len(s))))
    dhw = np.einsum('nij,nj->ni',jac,rel)+np.cross(np.array([0,0,1])-s[:,9:12],h)
    hb = rotate(s[:,5:9],h,True); dh = rotate(s[:,5:9],dhw,True)
    q=s[:,12:15]; h0=3*dh/(MU_R+2)
    i0=AJ*(1-C*XI)/XI-E
    dq=h0/i0
    active=np.sum(q*(AJ/(XI*i0)*h0),axis=1)>0
    norm=np.linalg.norm(q[active],axis=1); qn=q[active]/norm[:,None]
    c=XI/AJ; dd=-norm/K; aa=1-C*XI-E*c
    bb=-AJ*(1-C)*norm/K-E*dd
    bf=dd/aa-bb*(c+dd)/(aa*(aa+bb))
    dq[active]=c/aa*h0[active]+bf[:,None]*np.sum(qn*h0[active],axis=1)[:,None]*qn
    # B_internal/(mu0 Ms) = h_body + 2 m/3.
    bi=(1+2*D)*hb-2*E*q
    db=(1+2*D)*dh-2*E*dq
    return hb,dh,bi,db

def metrics(s,t,w=None):
    h,dh,b,db=evaluate(s)
    if w is not None:
        pass
    elif len(s)>1:
        w=np.empty(len(s)); w[0]=(t[1]-t[0])/2; w[-1]=(t[-1]-t[-2])/2
        w[1:-1]=(t[2:]-t[:-2])/2
    else: w=np.ones(1)
    def rms(v): return np.sqrt(np.average(np.sum(v*v,axis=1),weights=w))
    return dict(W_H=rms(dh)/rms(h),W_B=rms(db)/rms(b),b_rms=rms(b),db_rms=rms(db),samples=len(s))

def main():
    rows=[]; shape_counts={}
    for case in json.loads(CAT.read_text(encoding='utf-8')):
        with np.load(case['data_file']) as data:
            full='states' in data
            shape_counts['full' if full else 'trace_only']=shape_counts.get('full' if full else 'trace_only',0)+1
            if full:
                s=data['states']; t=data['t']; rho=np.linalg.norm(s[:,1:3],axis=1)
                intervals=[('all_saved',np.ones(len(s),dtype=bool)),('near_ring', (rho>=2)&(rho<=6)),('tail',t>=max(t[0],t[-1]-500))]
            else:
                s=data['final_state'][None,:]; t=s[:,0]; intervals=[('endpoint',np.ones(1,dtype=bool))]
            weights=np.ones(len(s))
            if len(s)>1:
                weights[0]=(t[1]-t[0])/2; weights[-1]=(t[-1]-t[-2])/2
                weights[1:-1]=(t[2:]-t[:-2])/2
            for scope,mask in intervals:
                if not mask.any(): continue
                row=dict(id=case['id'],point=case['point'],sr=case['sr'],somega=case['somega'],scope=scope,**metrics(s[mask],t[mask],weights[mask]))
                # Example uses permeability in current static config; not a measured dynamic permeability.
                omega_ref=np.sqrt(9.81*(.33333333/.001)/2.64965696)
                row['eps_example']=4*np.pi*1e-7*MU_R*1e6*.001**2*case['sr']**2*omega_ref*case['somega']*row['W_B']
                rows.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'catalog_estimates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('coverage',shape_counts)
    for scope in ('near_ring','endpoint','tail'):
        subset=[r for r in rows if r['scope']==scope]
        print(scope,'count',len(subset),'W_B range', min(r['W_B'] for r in subset),max(r['W_B'] for r in subset))
    for point in 'ABCDEFGHIJ':
        subset=[r for r in rows if r['point']==point and r['scope'] in ('near_ring','endpoint')]
        print(point, 'W_B', [round(min(r['W_B'] for r in subset),4),round(max(r['W_B'] for r in subset),4)], 'eps example',[round(min(r['eps_example'] for r in subset),4),round(max(r['eps_example'] for r in subset),4)])
    for id in ('B-zero3','J-zero3','C-nearzero1-finest','G-old1','G-old2','F-low1'):
        print(id,[r for r in rows if r['id']==id])

if __name__=='__main__': main()
