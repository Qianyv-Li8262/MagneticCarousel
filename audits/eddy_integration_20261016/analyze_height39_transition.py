from pathlib import Path
import numpy as np,subprocess,json
root=Path('scratch/eddy_integration_20261016');out=root/'restored_moment_sigma9_h39p62'
d=np.load(out/'trace.npz');t=d['time_2d'];s=d['state']
np.column_stack((t,s,d['blow_mN'])).astype('<f8').tofile(out/'drive_input.bin')
subprocess.run([str(root/'diagnose_drive.exe'),str(out/'drive_input.bin'),'build/release/field-lut/field_lut_fp64.bin',str(out/'drive.csv')],check=True)
a=np.genfromtxt(out/'drive.csv',names=True,delimiter=',')
r=np.linalg.norm(s[:,1:3],axis=1);phi=np.unwrap(np.arctan2(s[:,2],s[:,1]));vt=(s[:,1]*s[:,4]-s[:,2]*s[:,3])/r
vr=np.sum(s[:,1:3]*s[:,3:5],axis=1)/r
phidot=vt/r;W=np.linalg.norm(s[:,9:12]-np.column_stack((np.zeros(len(s)),np.zeros(len(s)),phidot)),axis=1);H=4*(1-phidot)
report=[]
for lo,hi in [(1400,1450),(1500,1550),(1575,1600),(1600,1625),(1625,1650),(1650,1700),(1800,1850),(4500,5000)]:
 w=(t>=lo)&(t<hi);rec={'interval':[lo,hi],'radius_mm':float(r[w].mean()*17.5),'wz_rad_s':float(s[w,11].mean()*20.943951),'W_minus_H':float(np.mean(W[w]-H[w])),'signed_vt_mm_s':float(vt[w].mean()*366.519),'vt_rms_mm_s':float(np.sqrt(np.mean(vt[w]**2))*366.519),'vr_rms_mm_s':float(np.sqrt(np.mean(vr[w]**2))*366.519),'phi_change_rad':float(phi[w][-1]-phi[w][0])}
 for k in ['JA_drive_mN','eddy_drive_mN','roll_loss_mN','JA_torque_z_uNm','eddy_torque_z_uNm','h_body_dot_norm','JA_mechanical_uW','eddy_mechanical_uW','roll_loss_uW']:
  rec[k]=float(a[k][w].mean())
 report.append(rec)
(out/'transition1600.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
np.savez(out/'kinematics.npz',W=W,H=H,vt=vt,vr=vr,phi=phi)
