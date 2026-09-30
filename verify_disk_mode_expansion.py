"""Independent disk quadrature checks for the Fourier-Bessel coefficients."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import jv, jvp
from numpy.polynomial.legendre import leggauss
from estimate_catalog_eddy import evaluate

def hankel(m,rho,z,a,derivative_r=0,derivative_z=0):
    def fun(k):
        form=1 if a==0 or k==0 else 2*jv(1,k*a)/(k*a)
        return 16*k**(1+derivative_r+derivative_z)*(-1)**derivative_z*np.exp(-k*z)*form*jvp(m,k*rho,derivative_r)*jv(m,k*4)
    return quad(fun,0,np.inf,epsabs=1e-12,epsrel=1e-10,limit=500)[0]

def direct(m,rho,z,a,nphi=512):
    phi=2*np.pi*np.arange(nphi)/nphi
    x,w=leggauss(24); u=(x+1)/2; w=w/2
    theta=2*np.pi*np.arange(96)/96
    sx=(a*np.sqrt(u[:,None])*np.cos(theta)).ravel()
    sy=(a*np.sqrt(u[:,None])*np.sin(theta)).ravel()
    weight=np.repeat(w,96)/96
    potential=np.zeros(nphi)
    for j in range(8):
        dx=rho*np.cos(phi[:,None])-4*np.cos(j*np.pi/4)-sx
        dy=rho*np.sin(phi[:,None])-4*np.sin(j*np.pi/4)-sy
        potential+=(-1)**j*np.sum(weight*z/(dx*dx+dy*dy+z*z)**1.5,axis=1)
    return 2*np.mean(potential*np.cos(m*phi))

def main():
    checks=[]
    for m,rho,a in [(4,4,0),(4,4,.5),(12,4,.5),(4,2.5,.8)]:
        z=1.610066235; direct_a=direct(m,rho,z,a); h=hankel(m,rho,z,a)
        checks.append(dict(m=m,rho=rho,a=a,direct=direct_a,hankel=h,relative_error=abs(h-direct_a)/abs(direct_a)))
    # Check body derivative against rigid advection, including quaternion rotation.
    s=np.array([[.37,3.2,1.4,.4,-.8,1,0,0,0,.6,-.3,.2,0,0,0]],float)
    h,dh,_,_=evaluate(s); dt=1e-6
    def shifted(sign):
        t=s.copy(); t[:,0]+=sign*dt; t[:,1:3]+=sign*dt*s[:,3:5]
        t[:,6:9]+=sign*dt*.5*s[:,9:12]
        return evaluate(t)[0]
    fd=(shifted(1)-shifted(-1))/(2*dt)
    derivative_error=float(np.linalg.norm(fd-dh)/np.linalg.norm(dh))
    result=dict(disk_checks=checks,body_derivative_relative_error=derivative_error)
    out=Path(__file__).resolve().parent/'results/eddy-estimate/expansion_validation.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    assert max(c['relative_error'] for c in checks)<1e-8
    assert derivative_error<1e-7

if __name__=='__main__': main()
