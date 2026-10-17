import sys,json,pandas as pd,numpy as np
sys.path.insert(0,'src2d/audits/eddy_integration_20261016');import fit_video56c6 as f
PREV=f.ROOT/'runs/video56c6-rest-release-20261017';OUT=f.ROOT/'runs/video56c6-release-radius-20261017';OUT.mkdir(exist_ok=True)
data=pd.read_csv(f.VIDEO/'kinematics.csv');row=data.loc[(data.position_valid==1)&(data.source_pts_s>=f.RELEASE)].iloc[0];r0=float(row.r_mm)
base=json.loads((PREV/'candidate.json').read_text());f.OUT=OUT
cases=[('startup_measured_r',dict(base,r=r0)),('startup_release_next_frame',dict(base,r=float(data.loc[(data.position_valid==1)&(data.source_pts_s>f.RELEASE+.01)].iloc[0].r_mm)))]
for c,k in [(.4,.2),(.35,.3),(.25,1)]:cases.append(('smallover_measured_r_c'+str(c),dict(f.BASE,sigma=12e6,c=c,K=f.BASE['K']*k,r=r0,wz=0,hold=0,vt=0,vr=0,qz=0)))
rs=f.batch(cases);(OUT/'runs.json').write_text(json.dumps(rs,indent=2))
meta=dict(source_pts_s=float(row.source_pts_s),radius_mm=r0,x_mm=float(row.x_mm),y_mm=float(row.y_mm),position_valid=int(row.position_valid),derivative_valid=int(row.derivative_valid),previous_radius_mm=67.4,previous_radius_source_pts_s=5.066667,note='First accepted position at release, not first accepted derivative. Initial velocity remains zero by user constraint; the following measured radial oscillation is not fitted via an inferred kick.')
(OUT/'release_measurement.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
