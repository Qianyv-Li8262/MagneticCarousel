"""Variational extrema of axial radial magnetization with whole-face |B| bounds.

The boundary is the exterior trace (gap=0) or a specified probe plane.
At zero gap an edge-tapered Ritz basis avoids the sharp-edge singularity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from math import comb
import numpy as np
from scipy.special import beta
from scipy.optimize import minimize, nnls, least_squares
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R, L, MU0 = .014, .015, 4e-7*np.pi
V = np.pi*R*R*L
ROOT = Path(__file__).resolve().parent


def gauss(n, lo, hi):
    x,w=np.polynomial.legendre.leggauss(n)
    return lo+(hi-lo)*(x+1)/2,w*(hi-lo)/2


def basis(t, degree, alpha):
    t=np.clip(np.asarray(t),0,1)
    return edge_basis(1-t,degree,alpha)


def edge_basis(y,degree,alpha):
    y=np.clip(np.asarray(y),0,1)
    return np.stack([y**alpha*comb(degree,j)*(1-y)**j*y**(degree-j)
                     for j in range(degree+1)],axis=-1)


def offset_operator(radii,degree,alpha,gap,order):
    """Smooth finite-gap charge integrals in observation-centered polar rays."""
    u,wu=gauss(order,0,1)
    result=[]
    for rho in radii:
        theta,wt=gauss(2*order,np.pi/2 if rho==R else 0,np.pi)
        ct=np.cos(theta)
        root=np.sqrt(np.maximum(0,R*R-rho*rho*np.sin(theta)**2))
        smax=root-rho*ct
        positive=ct>=0
        smax[positive]=(R*R-rho*rho)/(root[positive]+rho*ct[positive])
        other=root+rho*ct
        other[~positive]=(R*R-rho*rho)/smax[~positive]
        s=smax[:,None]*u
        y=smax[:,None]*(1-u)*(s+other[:,None])/R**2
        values=edge_basis(y,degree,alpha)
        radial=np.zeros(degree+1);axial=np.zeros(degree+1)
        for sign,z in [(1,gap),(-1,gap+L)]:
            inv=(s*s+z*z)**-1.5
            wr=-sign*wt[:,None]*ct[:,None]*smax[:,None]*s*s*inv*wu/(2*np.pi)
            wz=sign*wt[:,None]*smax[:,None]*z*s*inv*wu/(2*np.pi)
            radial+=np.einsum('ij,ijk->k',wr,values)
            axial+=np.einsum('ij,ijk->k',wz,values)
        result.append(np.array([radial,axial]))
    return np.array(result)


def operator(radii, degree, alpha, order=100, gap=0):
    """Surface field by magnetic charges, with singularity subtraction on top.

    Top radial PV integral uses polar rays centered on each observation point.
    A power-transformed ray coordinate removes edge-taper endpoint singularities.
    Bottom surface is nonsingular and integrated independently in disk coordinates.
    Returns B components per control coefficient q=mu0*M (T).
    """
    if np.any(np.asarray(radii)<0) or np.any(np.asarray(radii)>R):
        raise ValueError('Observation radii must lie in the projected face, 0<=rho<=R')
    if gap>0:return offset_operator(radii,degree,alpha,gap,order)
    if alpha<=0:raise ValueError('Zero-gap sharp-edge model requires alpha>0')
    v,wv=gauss(order,0,1)
    power=1/alpha
    denominator=v**power+(1-v)**power
    u=v**power/denominator
    jac=power*v**(power-1)*(1-v)**(power-1)/denominator**2
    a,wa=gauss(order,0,R)
    phi,wp=gauss(2*order,0,np.pi)
    cp=np.cos(phi)
    source_basis=basis((a/R)**2,degree,alpha)
    result=[]
    for rho in radii:
        obs=basis((rho/R)**2,degree,alpha)
        theta,wt=gauss(2*order,np.pi/2 if rho==R else 0,np.pi)
        ct=np.cos(theta)
        root=np.sqrt(np.maximum(0,R*R-rho*rho*np.sin(theta)**2))
        smax=root-rho*ct
        positive=ct>=0
        smax[positive]=(R*R-rho*rho)/(root[positive]+rho*ct[positive])
        s=smax[:,None]*u[None,:]
        other=root+rho*ct
        other[~positive]=(R*R-rho*rho)/smax[~positive]
        remaining=smax[:,None]*((1-v)**power/denominator)[None,:]
        edge_distance=remaining*(s+other[:,None])/R**2
        delta=edge_basis(edge_distance,degree,alpha)-obs
        radial=np.einsum('i,ij,ijk,j->k',-wt*ct/(2*np.pi),np.broadcast_to(jac/u,s.shape),delta,wv)
        if rho<R:
            radial-=obs*np.sum(wt*ct*np.log(smax/R))/(2*np.pi)
        axial=obs/2
        dx=rho-a[:,None]*cp[None,:]
        dist2=rho*rho+a[:,None]**2-2*rho*a[:,None]*cp[None,:]+L*L
        inv=dist2**-1.5
        kr=np.sum(dx*inv*wp,axis=1)/(2*np.pi)
        kz=np.sum(L*inv*wp,axis=1)/(2*np.pi)
        radial-=np.einsum('i,ij->j',wa*a*kr,source_basis)
        axial-=np.einsum('i,ij->j',wa*a*kz,source_basis)
        result.append(np.array([radial,axial]))
    return np.array(result)


def moment_weights(degree, alpha):
    return V/MU0*np.array([comb(degree,j)*beta(j+1,degree-j+alpha+1)
                           for j in range(degree+1)])


def normal_field_relaxation(order):
    """Variational upper relaxation: retain Bz<=0.290, discard Br constraints.

    On the top face Bz=q/2-Kq, with positive bottom-disk operator K and
    ||2K||_inf <= 1-L/sqrt(L^2+R^2)<1. Its inverse is positive; hence
    q <= (I/2-K)^(-1)0.290. This bound covers every admissible radial profile,
    not only the finite Ritz basis. Nystrom quadrature approximates its value.
    """
    t,w=gauss(order,0,1)
    a=R*np.sqrt(t)
    phi,wp=gauss(order,0,np.pi)
    K=np.empty((order,order))
    for i,rho in enumerate(a):
        d2=rho*rho+a[:,None]**2-2*rho*a[:,None]*np.cos(phi)+L*L
        K[i]=R*R*L/(4*np.pi)*w*np.sum(d2**-1.5*wp,axis=1)
    A=.5*np.eye(order)-K
    q=np.linalg.solve(A,np.full(order,.290))
    dual=np.linalg.solve(A.T,V/MU0*w)
    return dict(order=order,upper_relaxation_A_m2=float(V/MU0*w@q),
                dual_value_A_m2=float(.290*np.sum(dual)),positive_dual=bool(np.all(dual>0)),
                algebraic_residual_T=float(np.max(abs(A@q-.290))),
                analytic_supremum_bound_A_m2=float(V/MU0*.580*np.sqrt(L*L+R*R)/L))


def solve(A, weights, seed, maximize=False, center_minimum=True):
    # Squared norms preserve the exact pointwise magnitude constraints.
    def norm2(c):
        f=A@c
        return np.einsum('ij,ij->i',f,f)
    def jac(c):
        return 2*np.einsum('ri,rij->rj',A@c,A)
    lower=slice(1,-1) if center_minimum else slice(None,-1)
    cons=[{'type':'ineq','fun':lambda c:(norm2(c)[lower]-.210**2)/.210**2,
           'jac':lambda c:jac(c)[lower]/.210**2},
          {'type':'ineq','fun':lambda c:(.290**2-norm2(c)[1:-1])/.290**2,
           'jac':lambda c:-jac(c)[1:-1]/.290**2},
          {'type':'eq','fun':lambda c:(norm2(c)[-1]-.290**2)/.290**2,
           'jac':lambda c:jac(c)[-1]/.290**2}]
    if center_minimum:
        cons.append({'type':'eq','fun':lambda c:((A@c)[0,1]-.210)/.210,
                     'jac':lambda c:A[0,1]/.210})
    sign=-1 if maximize else 1
    return minimize(lambda c:sign*(weights@c),seed,jac=lambda c:sign*weights,
                    method='SLSQP',bounds=[(0,None)]*len(seed),constraints=cons,
                    options={'maxiter':1800,'ftol':2e-11})


def solve_movable_minimum(A, radii, weights, seed, minimum_radius, maximize):
    """Ritz variation of both the profile and the point attaining the minimum."""
    spline=CubicSpline(radii/R,A,axis=0)
    def norm2(x):
        f=A@x[:-1]
        return np.einsum('ij,ij->i',f,f)
    def jac(x):
        return np.column_stack([2*np.einsum('ri,rij->rj',A@x[:-1],A),np.zeros(len(A))])
    def minimum_eq(x):
        f=spline(x[-1])@x[:-1]
        return (f@f-.210**2)/.210**2
    def minimum_jac(x):
        mat=spline(x[-1]);f=mat@x[:-1]
        return np.r_[2*f@mat,2*f@(spline(x[-1],1)@x[:-1])]/.210**2
    # A tiny slack avoids duplicate active constraints when the movable minimum
    # coincides with a mesh node; it is included in dense-grid verification.
    eps=1e-11
    cons=[{'type':'ineq','fun':lambda x:(norm2(x)[:-1]-(.210-eps)**2)/.210**2,
           'jac':lambda x:jac(x)[:-1]/.210**2},
          {'type':'ineq','fun':lambda x:((.290+eps)**2-norm2(x)[:-1])/.290**2,
           'jac':lambda x:-jac(x)[:-1]/.290**2},
          {'type':'eq','fun':lambda x:(norm2(x)[-1]-.290**2)/.290**2,
           'jac':lambda x:jac(x)[-1]/.290**2},
          {'type':'eq','fun':minimum_eq,'jac':minimum_jac}]
    sign=-1 if maximize else 1
    opt=minimize(lambda x:sign*(weights@x[:-1]),np.r_[seed,minimum_radius],
                 jac=lambda x:np.r_[sign*weights,0.],method='SLSQP',
                 bounds=[(0,None)]*len(seed)+[(0,.999999)],constraints=cons,
                 options={'maxiter':1800,'ftol':2e-11})
    opt.minimum_radius=float(opt.x[-1]);opt.x=opt.x[:-1]
    return opt


def feasible_seed(A,seed):
    """Find a near-feasible center-minimum profile before constrained variation."""
    def residual(c):
        f=A@c;n=np.einsum('ij,ij->i',f,f)
        return np.r_[np.minimum(0,n-.210**2)/.210**2,
                     np.maximum(0,n-.290**2)/.290**2,
                     10*((f[0,1]-.210)/.210),10*(n[-1]-.290**2)/.290**2]
    def derivative(c):
        f=A@c;n=np.einsum('ij,ij->i',f,f)
        j=2*np.einsum('ri,rij->rj',f,A)
        return np.vstack([j*(n<.210**2)[:,None]/.210**2,
                          j*(n>.290**2)[:,None]/.290**2,
                          10*A[0,1]/.210,10*j[-1]/.290**2])
    return least_squares(residual,np.maximum(seed,1e-12),jac=derivative,bounds=(0,np.inf),
                         ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=3000).x


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--degree',type=int,default=8)
    ap.add_argument('--alpha',type=float,default=None)
    ap.add_argument('--gap-mm',type=float,default=0.)
    ap.add_argument('--starts',type=int,default=6)
    ap.add_argument('--order',type=int,default=100)
    ap.add_argument('--free-minimum',action='store_true',help='Allow minimum away from center')
    ap.add_argument('--warm',type=Path)
    ap.add_argument('--out',type=Path,default=ROOT/'runs/variational-surface-magnet-20261015')
    args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    if args.gap_mm<0:ap.error('gap must be nonnegative')
    degree,alpha=args.degree,(args.alpha if args.alpha is not None else (0. if args.gap_mm>0 else .5))
    evaluate=lambda radii,deg,exponent,order:operator(radii,deg,exponent,order,gap=args.gap_mm/1000)
    # Dense near the physical edge, where trace fields change fastest.
    r=R*np.sin(np.linspace(0,np.pi/2,321))
    r[-1]=R
    A=evaluate(r,degree,alpha,args.order)
    weights=moment_weights(degree,alpha)
    rng=np.random.default_rng(20261015)
    candidates=[];attempts=[]
    warm=[]
    if args.warm:
        previous=json.loads(args.warm.read_text())
        old=previous['degree']
        if old>degree:raise ValueError('Warm start must use no higher degree')
        for rec in previous['candidates']:
            c=np.array(rec['controls_T'])
            if previous['alpha']==alpha:
                elevated=np.array([sum(c[i]*comb(old,i)*comb(degree-old,j-i)/comb(degree,j)
                                       for i in range(max(0,j-(degree-old)),min(old,j)+1)) for j in range(degree+1)])
            else:
                t=np.sin(np.linspace(0,np.pi/2,501))**2
                elevated=nnls(basis(t,degree,alpha),basis(t,old,previous['alpha'])@c)[0]
            warm.append((rec['extremum'],elevated))
    for maximize in [False,True]:
        best=None
        for start in range(args.starts):
            seed=np.full(degree+1,.8) if start==0 else rng.uniform(.1,2,degree+1)
            for label,c in warm:
                if start==0 and label==('maximum' if maximize else 'minimum'):seed=c.copy()
            if args.gap_mm:seed=feasible_seed(A,seed)
            opt=(solve_movable_minimum(A,r,weights,seed,[0,.25,.5,.75,.9,.97][start%6],maximize)
                 if args.free_minimum else solve(A,weights,seed,maximize,True))
            fields=A@opt.x;norm=np.linalg.norm(fields,axis=1)
            violation=max(0,.210-float(norm.min()),float(norm.max())-.290,
                          abs(float(norm[-1])-.290))
            if not args.free_minimum:violation=max(violation,abs(float(norm[0])-.210))
            else:
                atmin=evaluate([opt.minimum_radius*R],degree,alpha,args.order)@opt.x
                violation=max(violation,abs(float(np.linalg.norm(atmin))-.210))
            attempts.append(dict(maximize=maximize,start=start,success=bool(opt.success),message=str(opt.message),
                                 moment_A_m2=float(weights@opt.x),violation_T=violation))
            if violation<2e-7 and (best is None or opt.fun<best.fun):best=opt
        if best is not None:
            candidates.append(dict(extremum='maximum' if maximize else 'minimum',controls_T=best.x.tolist(),
                                   moment_A_m2=float(weights@best.x),optimizer_success=bool(best.success),
                                   minimum_radius_fraction=getattr(best,'minimum_radius',0.)))
        print(json.dumps(attempts[-args.starts:]),flush=True)
    tag=f'n{degree}_a{alpha:g}'+('_free' if args.free_minimum else '')
    if args.gap_mm:tag+=f'_gap{args.gap_mm:g}mm'
    data=dict(degree=degree,alpha=alpha,gap_mm=args.gap_mm,whole_face_bounds_T=[.210,.290],edge_magnitude_T=.290,
              center_minimum=not args.free_minimum,attempts=attempts,candidates=candidates)
    if candidates:
        fine=R*np.sin(np.linspace(0,np.pi/2,1001));fine[-1]=R
        Afine=evaluate(fine,degree,alpha,args.order)
        Acheck=evaluate(fine[::20],degree,alpha,2*args.order)
        for rec in candidates:
            c=np.array(rec['controls_T'])
            refined=(solve_movable_minimum(Afine,fine,weights,c,rec['minimum_radius_fraction'],rec['extremum']=='maximum')
                     if args.free_minimum else solve(Afine,weights,c,rec['extremum']=='maximum',True))
            if refined.success:
                c=refined.x
                rec['controls_T']=c.tolist();rec['moment_A_m2']=float(weights@c)
                rec['minimum_radius_fraction']=getattr(refined,'minimum_radius',0.)
            fields=Afine@c;mag=np.linalg.norm(fields,axis=1)
            rec['dense_min_G']=float(mag.min()*1e4);rec['dense_max_G']=float(mag.max()*1e4)
            rec['edge_components_G']=(fields[-1]*1e4).tolist()
            rec['quadrature_max_vector_error_G']=float(np.max(np.linalg.norm(Acheck@c-fields[::20],axis=1))*1e4)
            rec['radius_mm']=(fine*1000).tolist();rec['field_G']=(fields*1e4).tolist()
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for rec in candidates:
            c=np.array(rec['controls_T']);fields=np.array(rec['field_G'])
            label=f"{rec['extremum']}: m={rec['moment_A_m2']:.5f} A m^2"
            axes[0].plot(fine*1000,basis((fine/R)**2,degree,alpha)@c,label=label)
            axes[1].plot(fine*1000,np.linalg.norm(fields,axis=1),label=label)
        axes[0].set(xlabel='Radius (mm)',ylabel='mu0 M_z (T)',title='Variational radial magnetization')
        axes[1].set(xlabel='Radius (mm)',ylabel='External |B| (G)',title=f'Constraint plane: {args.gap_mm:g} mm above face')
        axes[1].axhline(2100,color='gray',ls=':');axes[1].axhline(2900,color='gray',ls=':')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(args.out/f'{tag}.png',dpi=170);plt.close(fig)
    (args.out/f'{tag}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='attempts' and k!='candidates'}),flush=True)
    print(json.dumps([{k:v for k,v in c.items() if k not in ('radius_mm','field_G')} for c in candidates]),flush=True)


if __name__=='__main__':main()
