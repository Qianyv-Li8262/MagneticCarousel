"""Physical dipole moment decomposition of the actual production coast replay.
Eddy increment is evaluated separately on a prescribed circular rolling orbit.
"""
from pathlib import Path
import runpy,contextlib,io,json
import numpy as np
HERE=Path(__file__).parent
with contextlib.redirect_stdout(io.StringIO()):
    e=runpy.run_path(str(HERE/'eddy_actual_field.py'))
V=e['V'];R=e['R'];mu0=e['mu0'];Mref=e['Mref'];Om=e['Om']
ur=605.8974337495617;aj=.0021323083849413885;c=.22415154978863677
xi=aj*(ur-1)/(1+aj*(ur-1));D=(ur-1)/(ur+2);E=(1-c)/((1-xi)*(ur+2))
trace=np.load(HERE/'current_blow/trace.npz')['table']
rms=lambda x:float(np.sqrt(np.mean(np.sum(np.asarray(x)**2,axis=-1))))
def rotate(q,v):
    u=q[1:];return v+2*q[0]*np.cross(u,v)+2*np.cross(u,np.cross(u,v))
def budget(t0,t1):
    rows=trace[(trace[:,0]/Om>=t0)&(trace[:,0]/Om<t1)]
    direct=[];hist=[];total=[];torque=[];perp=[];parallel=[];fields=[]
    for row in rows:
        state=row[1:16];rad=np.linalg.norm(state[1:3])/e['k'];phi=np.arctan2(state[2],state[1])
        B=sum(np.real(b*np.exp(1j*m*(state[0]-phi))) for m,b,J in e['harmonics'](rad))
        cs=np.cos(phi);sn=np.sin(phi);B=np.array([cs*B[0]-sn*B[1],sn*B[0]+cs*B[1],B[2]])
        qw=rotate(state[5:9],state[12:15]);p0=3*V*D*B/mu0;ph=-3*V*Mref*E*qw
        direction=B/np.linalg.norm(B);pp=np.dot(ph,direction)*direction
        direct.append(p0);hist.append(ph);total.append(p0+ph);fields.append(B)
        parallel.append(pp);perp.append(ph-pp);torque.append(np.cross(ph,B))
    return dict(window_s=[t0,t1],radius_mean_mm=float(np.mean(np.linalg.norm(rows[:,2:4],axis=1)/e['k']*1000)),
                speed_mean_m_s=float(np.mean(np.linalg.norm(rows[:,4:6],axis=1)*Om/e['k'])),
                B_rms_mT=rms(fields)*1000,direct_p_rms_Am2=rms(direct),hysteretic_delta_p_rms_Am2=rms(hist),
                total_p_rms_Am2=rms(total),hysteretic_parallel_rms_Am2=rms(parallel),
                hysteretic_perpendicular_rms_Am2=rms(perp),hysteretic_torque_rms_uNm=rms(torque)*1e6,
                hysteretic_relative_vector_change=rms(hist)/rms(direct),
                total_magnitude_rms_relative_change=rms(total)/rms(direct)-1)

def eddy_budget(radius,speed,permeability,sigma,wz=0):
    orbit=-speed/radius;w=np.array([speed/R,0,wz-orbit]);W=np.linalg.norm(w);u=w/W
    C=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    P0=np.outer(u,u);Pp=(np.eye(3)-P0+1j*C)/2;Pm=(np.eye(3)-P0-1j*C)/2
    p0sq=dpsq=psq=0.;crosssq=0.;Tc=np.zeros(3);alpha0=e['alpha'](0,permeability,0)
    for m,b,J in e['harmonics'](radius):
        freq=m*(Om-orbit);p=np.zeros(3,dtype=complex)
        for P,f in [(P0,freq),(Pp,freq+W),(Pm,freq-W)]:p+=e['alpha'](f,permeability,sigma)*(P@b)/mu0
        p0=alpha0*b/mu0;dp=p-p0
        p0sq+=np.vdot(p0,p0).real/2;dpsq+=np.vdot(dp,dp).real/2;psq+=np.vdot(p,p).real/2
        Tc+=np.real(np.cross(dp,np.conj(b)))/2
    return dict(radius_mm=radius*1000,speed_m_s=speed,mu_r=permeability,sigma_S_m=sigma,wz_rad_s=wz,
                direct_linear_p_rms_Am2=np.sqrt(p0sq),eddy_delta_p_rms_Am2=np.sqrt(dpsq),
                eddy_relative_vector_change=np.sqrt(dpsq/p0sq),total_rms_relative_change=np.sqrt(psq/p0sq)-1,
                mean_eddy_torque_local_uNm=(Tc*1e6).tolist())

result=dict(JA_replay=[budget(4,12.5),budget(14,16)],
            eddy_prescribed=[eddy_budget(.07,.18,mu,1/(.15e-6)) for mu in [100,600]]+
                            [eddy_budget(.07,.18,600,1/(.15e-6),10.5644)],
            note='JA terms from actual baseline coast trajectory; eddy terms from a separate prescribed circular orbit. Not a coupled JA+eddy trajectory.')
(HERE/'eddy_actual_field/moment_budget.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
