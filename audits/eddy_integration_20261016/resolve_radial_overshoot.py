import sys,json,numpy as np
from scipy.optimize import brentq
from pathlib import Path
sys.path.insert(0,'src2d/audits/eddy_integration_20261016');import fit_video56c6 as f
out=f.ROOT/'runs/video56c6-rest-release-20261017/overshoot';a=np.genfromtxt(out/'trajectory.csv.radial.csv',names=True,delimiter=',');t=a['time_s']+4.9
nodes=np.fromfile(f.SOURCE/'field_lut_fp64.bin',dtype='<f8').reshape(5,204,4)
def spline(rho,k,o):
 dr=4/203;x=(rho-2)/dr;j=np.clip(np.floor(x).astype(int),0,202);u=x-j
 y,ml=nodes[k,j,o],nodes[k,j,o+1];yr,mr=nodes[k,j+1,o],nodes[k,j+1,o+1]
 b=yr-y-dr*dr*(2*ml+mr)/6;c=dr*dr*ml/2;d=dr*dr*(mr-ml)/6
 return y+u*(b+u*(c+u*d)),(b+u*(2*c+3*d*u))/dr,(2*c+6*d*u)/dr**2
cfg=json.loads((out.parent/'best_rest.json').read_text())['config'];mu=cfg['mu'];D=(mu-1)/(mu+2);O=200*np.pi/30;mass=7850*4*np.pi*.006**3/3;meff=1.4*mass;gamma=3*(4*np.pi*1e-7)*1.1e6**2*(4/.07)**2/(7850*O**2);fu=mass*.0175*O**2

def revforce(rmm):
 rho=np.asarray(rmm)/17.5;grad=np.zeros_like(rho)
 for k in range(5):
  A,Ap,App=spline(rho,k,0);Z,Zp,_=spline(rho,k,2);m=4*(2*k+1)
  grad+=Ap*App+m*m*A/rho*(Ap/rho-A/rho**2)+Z*Zp
 return fu*gamma*D*.0569993**2/2*grad*1000
for rr in [74.63,75,78,80,82,83,85,90]:print('radius/mean_rev',rr,revforce(rr))
windows={}
for lo,hi in [(26,28),(28,29),(29,30),(30,31),(55,65)]:
 w=(t>=lo)&(t<hi);rec={k:float(a[k][w].mean()) for k in a.dtype.names if k not in ['time_s','rolling','closure_mN']};rec['other_Qr_mN']=rec['Qr_mN']-rec['rev_Fr_mN'];windows[str(lo)+'-'+str(hi)]=rec
rgrid=np.linspace(68,100,1000)
def root(v,other):
 y=revforce(rgrid)+other+meff*(v*.001)**2/(rgrid*.001)*1000
 ix=np.where(y[:-1]*y[1:]<0)[0]
 return [float(brentq(lambda r:revforce(r)+other+meff*(v*.001)**2/(r*.001)*1000,rgrid[j],rgrid[j+1])) for j in ix]
late=windows['55-65'];peak=windows['29-30']
print('roots steady',root(abs(late['vt_mm_s']),late['other_Qr_mN']))
print('roots fast/late orientation',root(abs(peak['vt_mm_s']),late['other_Qr_mN']))
print('roots fast/peak orientation',root(abs(peak['vt_mm_s']),peak['other_Qr_mN']))
n=round(1/np.median(np.diff(t)));sm=lambda v:np.convolve(v,np.ones(n)/n,mode='valid');tt=sm(t)
rsm=sm(a['r_mm']);vsm=sm(abs(a['vt_mm_s']));wsm=sm(a['wz_rad_s']);i=np.argmax(rsm);iv=np.argmax(vsm);zero=np.where((wsm[:-1]>0)&(wsm[1:]<=0))[0][0]
print('smoothed peaks',tt[i],rsm[i],vsm[i],wsm[i], 'speedpeak',tt[iv],vsm[iv], 'spinzero',tt[zero])
report=dict(windows=windows,mean_radius_peak=dict(pts_s=float(tt[i]),radius_mm=float(rsm[i]),vt_mm_s=float(vsm[i]),wz_rad_s=float(wsm[i])),speed_peak=dict(pts_s=float(tt[iv]),speed_mm_s=float(vsm[iv])),spin_zero_pts_s=float(tt[zero]),frozen_mean_balance=dict(note='Prescribed-speed scalar balance with nonreversible generalized radial load held at its window mean; diagnostic approximation, not a coupled rerun or stability proof.',steady_roots_mm=root(abs(late['vt_mm_s']),late['other_Qr_mN']),fast_with_late_memory_roots_mm=root(abs(peak['vt_mm_s']),late['other_Qr_mN']),fast_with_peak_memory_roots_mm=root(abs(peak['vt_mm_s']),peak['other_Qr_mN'])),max_closure_error_mN=float(abs(a['closure_mN']).max()),rolling_fraction=float(a['rolling'].mean()))
(out/'analysis.json').write_text(json.dumps(report,indent=2))
np.savez(out/'field_radial.npz',radius_mm=rgrid,rev_force_mN=revforce(rgrid))
