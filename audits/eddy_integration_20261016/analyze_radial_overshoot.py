from pathlib import Path
import sys,json,subprocess,numpy as np
sys.path.insert(0,'src2d/audits/eddy_integration_20261016');import fit_video56c6 as f
out=f.ROOT/'runs/video56c6-rest-release-20261017/overshoot';out.mkdir(exist_ok=True)
rec=json.loads((out.parent/'best_rest.json').read_text());cfg=rec['config'];path=out/'trajectory.csv'
subprocess.run([str(f.SOURCE/'audit_radial.exe'),str(f.SOURCE/'field_lut_fp64.bin'),str(path)]+[str(cfg[k]) for k in f.KEYS]+[str(f.eddy_table(cfg['mu']))],check=True)
a=np.genfromtxt(str(path)+'.radial.csv',names=True,delimiter=',');t=a['time_s']+4.9
original=np.genfromtxt(out.parent/(rec['key']+'.csv'),names=True,delimiter=',');new=np.genfromtxt(path,names=True,delimiter=',');print('replaydiff',abs(new['speed_mm_s']-original['speed_mm_s']).max(),'closure',abs(a['closure_mN']).max(),'rolling',np.mean(a['rolling']))
for lo,hi in [(15,17),(19,21),(23,25),(26,27),(27,28),(28,29),(29,30),(30,31),(32,34),(36,38),(40,42),(55,65)]:
 w=(t>=lo)&(t<hi)
 print(lo,hi,{k:round(float(np.mean(a[k][w])),4) for k in ['r_mm','vt_mm_s','vr_mm_s','wz_rad_s','rev_Fr_mN','memory_Fr_mN','memory_Tr_mN','eddy_Fr_mN','eddy_Tr_mN','Qr_mN','centrifugal_mN','net_r_mN','memory_Qclock_mN','eddy_Qclock_mN','roll_Qclock_mN']})
