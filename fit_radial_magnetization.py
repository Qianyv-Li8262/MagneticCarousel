"""Fit radially varying axial magnetization to two surface-field readings.

This isolated inverse problem does not edit production parameters or field LUTs.
All fitted profiles describe q(r)=mu0*M_z(r), in tesla, uniform along thickness.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad_vec
from scipy.optimize import least_squares, minimize
from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi
R, L = .014, .015
V = np.pi * R**2 * L
REFERENCE = 4.222662956003016
TARGET = np.array([.210, .290])
ROOT = Path(__file__).resolve().parent


def cylinder(radius, rho, gap):
    """B/(mu0*M) for a uniform cylinder, used only as a superposition kernel."""
    if rho == 0:
        return np.array([0., .5 * ((gap + L) / np.hypot(radius, gap + L)
                                  - gap / np.hypot(radius, gap))])

    def loop(z):
        q = (radius + rho)**2 + z*z
        d = (radius - rho)**2 + z*z
        parameter = 4 * radius * rho / q
        k, e = ellipk(parameter), ellipe(parameter)
        return np.array([z / rho * (-k + (radius*radius + rho*rho + z*z) / d * e),
                         k + (radius*radius - rho*rho - z*z) / d * e]) / (2*np.pi*np.sqrt(q))

    return quad_vec(loop, gap, gap + L, epsabs=2e-11, epsrel=2e-10)[0]


def basis_fields(rho, gap, degree=4, order=96):
    """Fields of Bernstein profiles in t=(r/R)^2; includes volume bound current.

    For any smooth q, B=q(R)*C(R)-integral(q'(a)*C(a), da).
    Retaining this integral is essential when magnetization varies with radius.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    radii, weights = (nodes+1)*R/2, weights*R/2
    kernels = np.array([cylinder(a, rho, gap) for a in radii])
    full = cylinder(R, rho, gap)
    fields = []
    for j in range(degree+1):
        poly = comb(degree, j) * Polynomial([0, 1])**j * Polynomial([1, -1])**(degree-j)
        derivative = poly.deriv()(radii**2/R**2) * 2*radii/R**2
        fields.append(poly(1)*full - np.einsum('i,ij->j', weights*derivative, kernels))
    return np.array(fields).T


def profile(controls, radius):
    t = (np.asarray(radius)/R)**2
    n = len(controls)-1
    return sum(c*comb(n,j)*t**j*(1-t)**(n-j) for j,c in enumerate(controls))


def moment(controls):
    # Each Bernstein basis has integral 1/(degree+1) in area coordinate t.
    return float(V/MU0 * np.mean(controls))


def readings(controls, center, edge, component):
    b = edge @ controls
    return np.array([(center @ controls)[1], np.linalg.norm(b) if component == 'magnitude' else b[1]])


def charge_field(controls, rho, gap, nr=180, nphi=2048):
    """Independent top/bottom magnetic-charge quadrature, no cylinder kernels."""
    nodes, weights = np.polynomial.legendre.leggauss(nr)
    a, w = (nodes+1)*R/2, weights*R/2
    phi = (np.arange(nphi)+.5)*2*np.pi/nphi
    dx = rho-a[:,None]*np.cos(phi)
    dxy = rho*rho+a[:,None]**2-2*rho*a[:,None]*np.cos(phi)
    radial = np.zeros(len(a))
    axial = np.zeros(len(a))
    for sign, z in [(1, gap), (-1, gap+L)]:
        inv = (dxy+z*z)**-1.5
        radial += sign*np.mean(dx*inv, axis=1)
        axial += sign*np.mean(z*inv, axis=1)
    return .5*np.array([np.sum(w*a*profile(controls,a)*radial),
                         np.sum(w*a*profile(controls,a)*axial)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'runs/radial-magnetization-20261015')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    records, candidates, kernels = [], [], {}
    max_quadrature_error = 0.
    for gap_mm in [.1, .3, .5, 1., 2.]:
        gap = gap_mm/1000
        center, edge = basis_fields(0, gap), basis_fields(R, gap)
        check = basis_fields(R, gap, order=192)
        max_quadrature_error = max(max_quadrature_error, float(np.max(abs(check-edge))))
        kernels[gap_mm] = (center, edge)
        for component in ['magnitude', 'axial']:
            for power in [2, 4, 8]:
                # Degree-four Bernstein representation of q=qc+(qe-qc)*t^(p/2).
                exponent = power//2
                fraction = np.array([comb(j,exponent)/comb(4,exponent) if j>=exponent else 0.
                                     for j in range(5)])
                def controls(x):
                    return x[0]+(x[1]-x[0])*fraction
                fit = least_squares(lambda x: (readings(controls(x),center,edge,component)-TARGET)/TARGET,
                                    [.6,.7], bounds=(0,np.inf), xtol=1e-13,ftol=1e-13,gtol=1e-13)
                c = controls(fit.x)
                if np.max(abs(readings(c,center,edge,component)-TARGET)) < 1e-7:
                    rec = dict(family=f'power_{power}', gap_mm=gap_mm, component=component,
                               controls_T=c.tolist(), moment_A_m2=moment(c),
                               center_G=float((center@c)[1]*1e4), edge_G=float(readings(c,center,edge,component)[1]*1e4),
                               q_center_T=float(c[0]), q_edge_T=float(c[-1]))
                    records.append(rec)
            # Positivity of Bernstein controls guarantees q(r)>=0 everywhere.
            # Upper caps are declared sensitivity assumptions, not material measurements.
            for cap in [1., 1.5, 2.]:
                for desired in [None, 8., 12.836895386249168]:
                    best = None
                    for _ in range(8):
                        initial = rng.uniform(.2,cap,size=5)
                        constraints = [{'type':'eq','fun':lambda c: (readings(c,center,edge,component)-TARGET)/TARGET}]
                        if desired is not None:
                            constraints.append({'type':'eq','fun':lambda c,m=desired: (moment(c)-m)/m})
                        objective = (lambda c: -moment(c)) if desired is None else (lambda c: float(np.sum(np.diff(c,2)**2)))
                        fit = minimize(objective,initial,method='SLSQP',bounds=[(0,cap)]*5,
                                       constraints=constraints,options={'maxiter':500,'ftol':1e-11})
                        feasible = np.max(abs(readings(fit.x,center,edge,component)-TARGET))<1e-7
                        feasible &= desired is None or abs(moment(fit.x)-desired)<1e-5
                        if fit.success and feasible and (best is None or fit.fun<best.fun): best=fit
                    if best is not None:
                        c=best.x
                        rec=dict(family='bernstein4',gap_mm=gap_mm,component=component,
                                 cap_T=cap, requested_moment_A_m2=desired,controls_T=c.tolist(),
                                 moment_A_m2=moment(c),center_G=float((center@c)[1]*1e4),
                                 edge_G=float(readings(c,center,edge,component)[1]*1e4),
                                 edge_Brho_G=float((edge@c)[0]*1e4),edge_Bz_G=float((edge@c)[1]*1e4))
                        candidates.append(rec)
    # Check genuinely independent field formulations on near-face and remote points.
    examples=[r for r in candidates if r['gap_mm']==.5 and r['cap_T']==2. and
              r['requested_moment_A_m2'] is None]
    validations=[]
    for rec in examples:
        c=np.array(rec['controls_T'])
        for rho,gap in [(0,.0005),(R,.0005),(0,.03212)]:
            fast=basis_fields(rho,gap,order=192)@c
            direct=charge_field(c,rho,gap)
            err=float(np.linalg.norm(fast-direct)/max(np.linalg.norm(direct),1e-20))
            validations.append(dict(component=rec['component'],rho_mm=rho*1000,gap_mm=gap*1000,
                                    relative_error=err,kernel_G=(fast*1e4).tolist(),charge_G=(direct*1e4).tolist()))
    assert max_quadrature_error<1e-6, max_quadrature_error
    assert validations and max(v['relative_error'] for v in validations)<1e-6
    data=dict(geometry_mm={'radius':14,'thickness':15},readings_G={'center':2100,'edge':2900},
              assumptions={'edge_position':'rho=14 mm above top face; unknown in manufacturer data',
                           'common_probe_gap_mm':[.1,.3,.5,1,2],
                           'components':['magnitude','axial Bz'],
                           'cap_mu0M_T':[1,1.5,2],
                           'cap_note':'Sensitivity assumptions; neither measured magnetization nor remanence.',
                           'profile':'M_z(r), independent of axial coordinate; nonnegative'},
              power_fits=records,bernstein_fits=candidates,
              max_96_vs_192_kernel_absolute_difference=max_quadrature_error,independent_checks=validations)
    with (args.out/'power_fits.csv').open('w',newline='',encoding='utf-8') as f:
        keys=[k for k in records[0] if k!='controls_T']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in records)
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    radius=np.linspace(0,R,121)
    curve_kernels=np.array([basis_fields(r,.0005,order=64) for r in radius])
    selected=[r for r in records if r['family']=='power_2' and r['gap_mm']==.5]
    selected+=examples
    for rec in selected:
        c=np.array(rec['controls_T']);component=rec['component']
        label=f"{rec['family']}, {component}, m={moment(c):.2f}"
        axes[0].plot(radius*1000,profile(c,radius),label=label)
        fields=np.einsum('rij,j->ri',curve_kernels,c)*1e4
        axes[1].plot(radius*1000,np.linalg.norm(fields,axis=1) if component=='magnitude' else fields[:,1],label=label)
        rec['predicted_surface_line_peak_magnitude_G']=float(np.max(np.linalg.norm(fields,axis=1)))
        rec['predicted_axis_Bz_G']={str(g):float((basis_fields(0,g/1000)@c)[1]*1e4)
                                   for g in [5,10,20,32.12]}
    axes[0].set(xlabel='Radius (mm)',ylabel='mu0 M_z (T)',title='Nonuniform profiles; probe gap 0.5 mm')
    axes[1].scatter([0,14],[2100,2900],color='black',marker='x',zorder=10)
    axes[1].set(xlabel='Probe radius (mm)',ylabel='Readout (G)',title='Both endpoint readings fitted')
    for component,style in [('magnitude','-'),('axial','--')]:
        for cap in [1.,1.5,2.]:
            rows=sorted([r for r in candidates if r['component']==component and r['cap_T']==cap and r['requested_moment_A_m2'] is None],key=lambda r:r['gap_mm'])
            axes[2].plot([r['gap_mm'] for r in rows],[r['moment_A_m2'] for r in rows],style,marker='o',label=f'{component}, cap {cap} T')
    axes[2].axhline(REFERENCE,color='gray',ls=':',label='Existing reference')
    axes[2].axhline(12.836895386249168,color='red',ls=':',label='3.04x reference')
    axes[2].set(xlabel='Assumed common probe gap (mm)',ylabel='Largest moment found (A m^2)',title='Degree-4 profile sensitivity; not global bounds')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.savefig(args.out/'radial_fits.png',dpi=180);plt.close(fig)
    data['plotted_profiles']=selected
    (args.out/'fits.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({'power_fits':records,'maximum_profiles':examples,'validation':validations,
                      'feasible_12_84':sum(r['requested_moment_A_m2']==12.836895386249168 for r in candidates),
                      'out':str(args.out)},indent=2))


if __name__=='__main__':
    main()
