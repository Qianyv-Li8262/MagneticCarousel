"""Float64 finite-cylinder Hankel quadrature, checked against panel refinement."""
from pathlib import Path
import json,time
import numpy as np
from scipy.special import jv,jvp
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/height-response-20261013';OUT.mkdir(exist_ok=True)
HEIGHTS=[39.62,37.5,35,32.5,30,27.5,25]
R=4.;A=.8;L=15*4/70;NR=600;RAD=np.linspace(.8,10,NR)
def quadrature(width):
 xs,ws=np.polynomial.legendre.leggauss(32);edges=np.arange(0,80,width);q=(edges[:,None]+width*(xs+1)/2).ravel();w=np.tile(ws*width/2,len(edges));return q,w
def at_nodes(rad,q,w,height):
 gap=height*4/70-L/2;common=16*np.exp(-q*gap)*(-np.expm1(-q*L))/L*2*jv(1,q*A)/(q*A)*w;out=[]
 for m in [4,12,20,28,36]:
  weights=common*jv(m,q*R);b=jv(m,rad[:,None]*q);bp=jvp(m,rad[:,None]*q)*q
  out.append(np.c_[b@weights,bp@weights,-b@(q*weights),-bp@(q*weights)])
 return np.array(out)
def build():
 q,w=quadrature(1.);qf,wf=quadrature(.5);checks=[];start=time.perf_counter();bases=[]
 for m in [4,12,20,28,36]:
  b=jv(m,RAD[:,None]*q);bp=jvp(m,RAD[:,None]*q)*q;bases.append((b,bp,jv(m,q*R)));print('basis',m,'seconds',time.perf_counter()-start,flush=True)
 for height in HEIGHTS:
  gap=height*4/70-L/2;common=16*np.exp(-q*gap)*(-np.expm1(-q*L))/L*2*jv(1,q*A)/(q*A)*w;table=[];values=[]
  for b,bp,source in bases:
   weights=common*source;av=b@weights;ap=bp@weights;zv=-b@(q*weights);zp=-bp@(q*weights)
   ac=CubicSpline(RAD,av,bc_type=((1,ap[0]),(1,ap[-1])));zc=CubicSpline(RAD,zv,bc_type=((1,zp[0]),(1,zp[-1])));table.append(np.c_[av,ac(RAD,2),zv,zc(RAD,2)]);values.append(np.c_[av,ap,zv,zp])
  idx=np.array([0,100,250,350,450,599]);fine=at_nodes(RAD[idx],qf,wf,height);coarse=np.array(values)[:,idx,:];err=np.max(abs(fine-coarse));rel=err/np.max(abs(fine))
  path=OUT/f'height_{height:g}_fp64.bin';np.array(table,dtype='<f8').tofile(path);checks.append(dict(height_mm=height,max_panel_refinement_absolute_error=float(err),normalized_panel_refinement_error=float(rel),file=str(path),bytes=path.stat().st_size));print('height',height,'relative error',rel,flush=True)
 # Same baseline height compared to established high precision 600-node LUT.
 ref=np.fromfile(ROOT/'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin',dtype='<f8').reshape(5,600,4);generated=np.fromfile(OUT/'height_39.62_fp64.bin',dtype='<f8').reshape(5,600,4)
 info=dict(checks=checks,baseline_table_relative_norm_error=float(np.linalg.norm(ref-generated)/np.linalg.norm(ref)),method='Finite-cylinder Hankel integral; composite Gauss 32 points/panel, kappa range 0..80. Clamped cubic splines for A and Z. Independent doubled panel density checks at six radii; baseline compared with established high-precision table.',wall_s=time.perf_counter()-start)
 (OUT/'table_generation.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
if __name__=='__main__':build()
