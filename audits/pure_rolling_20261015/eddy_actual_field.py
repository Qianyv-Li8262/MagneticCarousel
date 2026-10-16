"""Linear conducting-sphere estimate using the production cylinder LUT.
Includes rolling rotation by analytic frequency sidebands in the orbit frame.
Subtracts sigma=0 polarizability; this is NOT a coupled JA+eddy solution.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import spherical_jn

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent/'eddy_actual_field';OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'build/release/extensions/fp64/build_config.json').read_text())
LUT=Path(manifest['lut']);assert hashlib.sha256(LUT.read_bytes()).hexdigest()==manifest['lut_sha256']
nodes=np.fromfile(LUT,dtype='<f8').reshape(5,204,4)
mu0=4*np.pi*1e-7;R=.006;V=4*np.pi*R**3/3;k=4/.07;Om=200*2*np.pi/60
beta=.0854985;Mref=1.1e6;muroll=.0049

def spline(l,r,u,dr):
    left,M,right,Mr=l[0],l[1],r[0],r[1]
    a=left;b=right-left-dr*dr*(2*M+Mr)/6;c=dr*dr*M/2;d=dr*dr*(Mr-M)/6
    return np.array([a+u*(b+u*(c+u*d)),(b+u*(2*c+3*d*u))/dr,(2*c+6*d*u)/dr**2])

def harmonics(radius):
    rho=radius*k;dr=4/203;x=(rho-2)/dr;j=min(202,max(0,int(np.floor(x))));u=x-j
    result=[]
    for i in range(5):
        A,Ap,App=spline(nodes[i,j,:2],nodes[i,j+1,:2],u,dr)
        Z,Zp,_=spline(nodes[i,j,2:],nodes[i,j+1,2:],u,dr)
        m=4*(2*i+1)
        b=np.array([-Ap,1j*m*A/rho,-Z])*beta*Mref*mu0
        J=np.array([[-App,1j*m*(Ap/rho-A/rho**2),-Zp],
                    [1j*m*(Ap/rho-A/rho**2),m*m*A/rho**2-Ap/rho,1j*m*Z/rho],
                    [-Zp,1j*m*Z/rho,App-(m*m*A/rho**2-Ap/rho)]])*beta*Mref*mu0*k
        result.append((m,b,J))
    return result

def alpha(freq,ur,sigma):
    if abs(freq)<1e-9 or sigma==0:return 3*V*(ur-1)/(ur+2)
    x=R*np.sqrt(-1j*freq*mu0*ur*sigma)
    return 1.5*V*((2*ur+1)*spherical_jn(1,x)-x*spherical_jn(0,x))/((ur-1)*spherical_jn(1,x)+x*spherical_jn(0,x))

def evaluate(radius,speed,ur,sigma,wz=0):
    orbit=-speed/radius
    rel=np.array([speed/R,0,wz-orbit]);W=np.linalg.norm(rel)
    if W<1e-12:u=np.array([0.,0,1.])
    else:u=rel/W
    C=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    P0=np.outer(u,u);Pp=(np.eye(3)-P0+1j*C)/2;Pm=(np.eye(3)-P0-1j*C)/2
    F=np.zeros(3);T=np.zeros(3);heat=0;energy=0;brms=0;firstB=None
    for m,b,J in harmonics(radius):
        freq=m*(Om-orbit)
        response=np.zeros(3,dtype=complex);zero=np.zeros(3,dtype=complex)
        for P,f in [(P0,freq),(Pp,freq+W),(Pm,freq-W)]:
            bb=P@b;aa=alpha(f,ur,sigma)
            response+=aa*bb/mu0;zero+=alpha(0,ur,0)*bb/mu0
            heat+=-f*aa.imag*np.vdot(bb,bb).real/(2*mu0)
        dp=response-zero
        F+=.5*np.real(J.T@np.conj(dp));T+=.5*np.real(np.cross(dp,np.conj(b)))
        brms+=np.vdot(b,b).real/2
        if firstB is None:firstB=np.abs(b)
    mech=-speed*F[1]+np.dot(T,np.array([speed/R,0,wz]))
    source=Om*(radius*F[1]+T[2])
    energy=abs(source-mech-heat)/(1e-15+abs(source)+abs(mech)+abs(heat))
    assert energy<1e-7,(energy,source,mech,heat)
    return dict(radius_mm=radius*1000,speed_m_s=speed,mu_r_incremental=ur,sigma_S_m=sigma,wz_rad_s=wz,
                field_rms_mT=np.sqrt(brms)*1000,first_harmonic_peak_components_mT=(firstB*1000).tolist(),
                skin_depth_main_mm=1000/np.sqrt(.5*m0*(Om-orbit)*mu0*ur*sigma) if sigma else None,
                delta_F_mN=(F*1000).tolist(),delta_torque_uNm=(T*1e6).tolist(),
                direct_reverse_force_mN=-F[1]*1000,rolling_torque_drive_mN=T[0]/R*1000,
                net_reverse_generalized_drive_mN=(-F[1]+T[0]/R)*1000,
                inward_generalized_correction_mN=(-F[0]-T[1]/R)*1000,
                extra_rolling_loss_mN=-muroll*F[2]*1000,
                useful_orbit_power_uW=(-F[1]+T[0]/R)*speed*1e6,
                joule_heat_uW=heat*1e6,source_power_uW=source*1e6,energy_residual=energy)

m0=4
records=[]
for radius in [.07,.0776]:
    for speed in [0,.1,.18,.25]:
        for ur in [10,100,600]:
            for sigma in [1e6,3e6,6e6]:
                records.append(evaluate(radius,speed,ur,sigma))
for wz in [-50,-20,20,50]:
    records.append(evaluate(.07,.18,600,6e6,wz))
result=dict(protocol='Actual production finite-cylinder Fourier LUT, Beta=0.0854985, height=39.62mm, 200rpm. Prescribed circular pure rolling, optional normal spin; all five field harmonics and rolling frequency sidebands. Linear real permeability conducting sphere, sigma=0 response subtracted. No mechanical integration or nonlinear JA/eddy feedback.',records=records)
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for q in records:
    if q['radius_mm']==70 and q['speed_m_s']==.18:
        print(json.dumps({key:q[key] for key in ['mu_r_incremental','sigma_S_m','wz_rad_s','field_rms_mT','skin_depth_main_mm','direct_reverse_force_mN','rolling_torque_drive_mN','net_reverse_generalized_drive_mN','joule_heat_uW','energy_residual']}))
