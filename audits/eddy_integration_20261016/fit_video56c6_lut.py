from pathlib import Path
import numpy as np
from scipy.special import jv,jvp
from scipy.interpolate import CubicSpline
from numpy.polynomial.legendre import leggauss
out=Path('src2d/scratch/video56c6-fit-20261016-source')
def generate(h):
 a=.8;L=.857142857142857;gap=h-L/2;R=4;rho=np.linspace(2,6,204);cutoff=80/gap
 panels=int(np.ceil(cutoff/(16/(6+R+a+gap))));x,w=leggauss(32)
 k=((np.arange(panels)[:,None]+.5)+(x[None,:]/2))*cutoff/panels;k=k.ravel();w=np.tile(w,panels)*cutoff/panels/2
 weight=w*16*np.exp(-k*gap)*(-np.expm1(-k*L))/L*2*jv(1,k*a)/(k*a)
 nodes=np.zeros((5,204,4))
 for j,m in enumerate([4,12,20,28,36]):
  wk=weight*jv(m,k*R);b=jv(m,rho[:,None]*k);bd=jvp(m,rho[:,None]*k)*k
  A=b@wk;Z=b@(-k*wk);Ap=bd@wk;Zp=bd@(-k*wk)
  nodes[j,:,0]=A;nodes[j,:,1]=CubicSpline(rho,A,bc_type=((1,Ap[0]),(1,Ap[-1])))(rho,2)
  nodes[j,:,2]=Z;nodes[j,:,3]=CubicSpline(rho,Z,bc_type=((1,Zp[0]),(1,Zp[-1])))(rho,2)
 return nodes
old=generate(2.264);ref=np.fromfile('src2d/scratch/eddy_integration_20261016/height35_backup/field_lut_h39p62_fp64.bin',dtype='<f8').reshape(5,204,4)
print('old verification abs',abs(old-ref).max(),'relative norm',np.linalg.norm(old-ref)/np.linalg.norm(ref))
a=generate(32/17.5);a.astype('<f8').tofile(out/'field_lut_double_fp64.bin');print('saved double LUT')
