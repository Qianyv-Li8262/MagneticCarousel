"""Infer illustrative magnet models from two manufacturer surface readings.

SI units; B outside an axially magnetized cylinder from its bound-current
solenoid, integrated using exact circular-loop elliptic-integral fields.
Sharp corners are evaluated at positive probe gaps, never at the singular edge.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ellipk,ellipe
from scipy.optimize import brentq,least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/laser-tilt-20261009/manufacturer_magnet_models';OUT.mkdir(exist_ok=True)
MU0=4*np.pi*1e-7;R=.014;L=.015;V=np.pi*R*R*L
BC=.210;BE=.290

def axial_unit(radius,gap):
    return .5*((gap+L)/np.hypot(radius,gap+L)-gap/np.hypot(radius,gap))

def cylinder_unit(radius,rho,gap):
    """(Brho,Bz)/(mu0 M); above top face. Valid only gap>0 off axis."""
    if abs(rho)<1e-14:return np.array([0.,axial_unit(radius,gap)])
    def integrand(z):
        q=(radius+rho)**2+z*z;d=(radius-rho)**2+z*z
        parameter=4*radius*rho/q
        k=ellipk(parameter);e=ellipe(parameter)
        return np.array([z/rho*(-k+(radius*radius+rho*rho+z*z)/d*e), k+(radius*radius-rho*rho-z*z)/d*e])/(2*np.pi*np.sqrt(q))
    return quad_vec(integrand,gap,gap+L,epsabs=1e-10,epsrel=1e-9)[0]

def model_record(label,inner_radius,br_inner,br_outer,gap,component):
    # A radial two-zone profile is the difference of two uniform cylinders.
    moment=(br_outer*V+(br_inner-br_outer)*np.pi*inner_radius**2*L)/MU0
    def field(rho,g):return br_outer*cylinder_unit(R,rho,g)+(br_inner-br_outer)*cylinder_unit(inner_radius,rho,g)
    edge=field(R,gap);center=field(0,gap)
    ball=field(0,.03962-L/2)
    return {'label':label,'model':'uniform' if br_inner==br_outer else 'two-zone axial magnetization','probe_gap_mm':gap*1000,'edge_readout_assumption':component,'inner_radius_mm':inner_radius*1000,'mu0M_inner_T':float(br_inner),'mu0M_outer_T':float(br_outer),'magnetic_moment_A_m2':float(moment),'reference_moment_scale':float(moment/4.222662956003016),'center_reading_G':float(center[1]*1e4),'edge_Bz_G':float(edge[1]*1e4),'edge_Brho_G':float(edge[0]*1e4),'edge_magnitude_G':float(np.linalg.norm(edge)*1e4),'single_magnet_Bz_at_ball_height_G':float(ball[1]*1e4)}

uniform=[]
for gap_mm in [0,.1,.3,.5,1,2,3,5]:
    gap=gap_mm/1000;br=BC/axial_unit(R,gap);m=br*V/MU0
    edge=cylinder_unit(R,R,gap)*br if gap>0 else None
    uniform.append({'probe_gap_mm':gap_mm,'mu0M_T':br,'moment_A_m2':m,'center_G':2100,'edge_Bz_G':None if edge is None else float(edge[1]*1e4),'edge_magnitude_G':None if edge is None else float(np.linalg.norm(edge)*1e4),'edge_is_ideal_corner_singular':gap==0})

def edge_ratio(g):return np.linalg.norm(cylinder_unit(R,R,g))/axial_unit(R,g)
gap_fit=brentq(lambda g:edge_ratio(g)-BE/BC,.00001,.006,xtol=1e-12)
br_fit=BC/axial_unit(R,gap_fit)
joint=model_record('Uniform: simultaneous center/edge magnitude fit',R,br_fit,br_fit,gap_fit,'magnitude')

profiles=[]
for gap_mm in [.3,.5,1,2]:
    gap=gap_mm/1000
    for component in ['magnitude','axial']:
        for inner_fraction in [.7,.8,.9]:
            inner=R*inner_fraction
            full_c=cylinder_unit(R,0,gap);inner_c=cylinder_unit(inner,0,gap)
            full_e=cylinder_unit(R,R,gap);inner_e=cylinder_unit(inner,R,gap)
            def residual(x):
                bi,bo=x;c=bo*full_c+(bi-bo)*inner_c;e=bo*full_e+(bi-bo)*inner_e
                e_read=np.linalg.norm(e) if component=='magnitude' else e[1]
                return np.array([(c[1]-BC)/BC,(e_read-BE)/BE])
            fit=least_squares(residual,[.55,.65],bounds=(0,np.inf),xtol=1e-12,ftol=1e-12,gtol=1e-12)
            if np.max(abs(residual(fit.x)))<1e-7:
                profiles.append(model_record(f'Two-zone: inner R={inner_fraction:g}R, gap={gap_mm:g}mm, edge={component}',inner,*fit.x,gap,component))

# A third field shape: smooth axial magnetization M(r)=Mc+(Me-Mc)(r/R)^2.
# Represent it as a weighted superposition of concentric uniform cylinders.
nodes,weights=np.polynomial.legendre.leggauss(48)
radii=(nodes+1)*R/2;weights=weights*R/2
def quadratic_bases(rho,gap):
    full=cylinder_unit(R,rho,gap)
    integrated=sum(w*(2*r/R**2)*cylinder_unit(r,rho,gap) for r,w in zip(radii,weights))
    return full,full-integrated
smooth=[]
for gap_mm,component in [(.3,'magnitude'),(.5,'magnitude'),(1,'magnitude'),(.5,'axial')]:
    gap=gap_mm/1000;c0,cq=quadratic_bases(0,gap);e0,eq=quadratic_bases(R,gap)
    def smooth_residual(x):
        mc,me=x;c=mc*c0+(me-mc)*cq;e=mc*e0+(me-mc)*eq
        return [(c[1]-BC)/BC,((np.linalg.norm(e) if component=='magnitude' else e[1])-BE)/BE]
    fit=least_squares(smooth_residual,[.55,.65],bounds=(0,np.inf),xtol=1e-12,gtol=1e-12,ftol=1e-12)
    if max(abs(np.array(smooth_residual(fit.x))))<1e-7:
        mc,me=fit.x;b0,bq=quadratic_bases(0,.03962-L/2);edge=mc*e0+(me-mc)*eq
        smooth.append({'model':'smooth quadratic axial magnetization','probe_gap_mm':gap_mm,'edge_readout_assumption':component,'mu0M_center_T':float(mc),'mu0M_edge_T':float(me),'magnetic_moment_A_m2':float((mc+me)/2*V/MU0),'center_reading_G':2100.,'edge_Bz_G':float(edge[1]*1e4),'edge_magnitude_G':float(np.linalg.norm(edge)*1e4),'single_magnet_Bz_at_ball_height_G':float((mc*b0+(me-mc)*bq)[1]*1e4)})

current_m=12.836895386249168
current_br=current_m*MU0/V
current_gap=brentq(lambda g:current_br*axial_unit(R,g)-BC,0,.05)
current={'moment_A_m2':current_m,'mu0M_T':current_br,'predicted_face_center_G':current_br*axial_unit(R,0)*1e4,'gap_needed_for_center_2100G_mm':current_gap*1000,'predicted_edge_magnitude_at_same_gap_G':np.linalg.norm(cylinder_unit(R,R,current_gap))*current_br*1e4}
checks=[]
for gap in [.0001,.001,.03]:
    numerical=cylinder_unit(R,1e-7,gap)[1];exact=axial_unit(R,gap)
    checks.append({'gap_mm':gap*1000,'near_axis_relative_error':float(abs(numerical/exact-1))})
assert max(c['near_axis_relative_error'] for c in checks)<1e-7
assert abs(joint['center_reading_G']-2100)<1e-5 and abs(joint['edge_magnitude_G']-2900)<1e-4
data={'manufacturer_input_G':{'center':2100,'edge':2900},'geometry_mm':{'diameter':28,'thickness':15},'surface_model_note':'Assume bare, axially magnetized cylinder. Probe standoff and measured component are unknown. mu0M is effective magnetization, not a direct measured remanence/grade.','uniform_center_fits':uniform,'uniform_joint_magnitude_fit':joint,'two_zone_models':profiles,'smooth_quadratic_models':smooth,'current_trial_comparison':current,'numerical_checks':checks,'sources':['https://web.mit.edu/6.013_book/www/chapter9/9.3.html','https://arxiv.org/abs/1802.00384']}
(OUT/'models.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
ax=axes[0];ax.plot([u['probe_gap_mm'] for u in uniform],[u['moment_A_m2'] for u in uniform],'o-',label='Uniform cylinder, center=2100 G')
ax.scatter(gap_fit*1000,joint['magnetic_moment_A_m2'],marker='*',s=120,label='Also fits edge magnitude=2900 G')
ax.axhline(current_m,color='red',ls='--',label='Trajectory trial: 12.84 A m^2');ax.set(xlabel='Probe-to-face gap (mm)',ylabel='Inferred dipole moment (A m^2)',title='Center reading strongly constrains moment');ax.grid(alpha=.2);ax.legend(fontsize=8)
rho=np.linspace(0,R,100);gap=.0005
illustrations=[('Uniform, common fitted gap',R,br_fit,br_fit,gap_fit)]
illustrations.extend([(p['label'],p['inner_radius_mm']/1000,p['mu0M_inner_T'],p['mu0M_outer_T'],p['probe_gap_mm']/1000) for p in profiles if abs(p['probe_gap_mm']-.5)<1e-8 and abs(p['inner_radius_mm']-.8*R*1000)<1e-8])
ax=axes[1]
for label,inner,bi,bo,g in illustrations:
    fields=np.array([bo*cylinder_unit(R,r,g)+(bi-bo)*cylinder_unit(inner,r,g) for r in rho])
    ax.plot(rho*1000,np.linalg.norm(fields,axis=1)*1e4,label=label.replace('Two-zone: ','').replace('Uniform, ','')+' |B|')
q=next(p for p in smooth if p['probe_gap_mm']==.5 and p['edge_readout_assumption']=='magnitude')
curve=[]
for r in rho:
    f0,fq=quadratic_bases(r,.0005)
    curve.append(np.linalg.norm(q['mu0M_center_T']*f0+(q['mu0M_edge_T']-q['mu0M_center_T'])*fq)*1e4)
ax.plot(rho*1000,curve,ls='--',label='Smooth radial magnetization, gap=0.5mm |B|')
ax.scatter([0,14],[2100,2900],c='black',marker='x',s=60,label='Manufacturer readings (component unknown)')
ax.set(xlabel='Radius on top face (mm)',ylabel='External |B| (G)',title='Different profiles can share two readings');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(OUT/'surface_field_models.png',dpi=160);plt.close(fig)
print(json.dumps({'uniform_surface':uniform[0],'uniform_joint':joint,'smooth_quadratic_models':smooth,'current':current,'out':str(OUT)},indent=2))
