"""Show actual ball-center motion for the calibrated 15 s Z damping."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/spin-damping-20261013'
K=4/.07

def main():
 cases=[]
 for name in ['parked_fresh','parked_high_spin']:
  for p in OUT.glob('*.json'):
   try:q=json.loads(p.read_text())
   except Exception:continue
   if isinstance(q,dict) and q.get('seed_branch')==name and q.get('tau_s')==15 and q.get('seconds')==600:
    rows=np.load(p.with_suffix('.npz'))['rows'];xy=rows[:,2:4]/K*1000
    tail=rows[:,0]>=rows[-1,0]-60;center=xy[tail].mean(axis=0)
    metric=dict(case=name,mean_center_mm=center.tolist(),mean_radius_mm=float(np.linalg.norm(xy[tail],axis=1).mean()),angle_deg=float(np.degrees(np.arctan2(center[1],center[0]))),last_60s_x_range_mm=np.ptp(xy[tail,0]),last_60s_y_range_mm=np.ptp(xy[tail,1]),last_60s_max_distance_from_mean_mm=float(np.linalg.norm(xy[tail]-center,axis=1).max()),start_xy_mm=xy[0].tolist())
    cases.append((name,rows,xy,tail,center,metric));break
  else:raise RuntimeError(name)
 colors=['#1479b8','#dc7b1c'];labels=['Release from rest','Start from high spin']
 fig=plt.figure(figsize=(13,4.7),layout='constrained');axes=fig.subplots(1,3)
 ax=axes[0];ax.add_patch(Circle((0,0),70,fill=False,ls='--',color='.5',lw=1,label='Magnet-center ring R=70 mm'))
 ax.scatter(0,0,c='k',marker='+',s=50)
 for i,(name,rows,xy,tail,center,metric) in enumerate(cases):
  ax.plot(xy[:,0],xy[:,1],color=colors[i],lw=1,alpha=.8,label=labels[i])
  ax.scatter(*xy[0],c=colors[i],s=40,marker='s',zorder=5)
  ax.scatter(*center,c=colors[i],s=60,marker='*',zorder=6)
 ax.annotate('Downhill (+x)',xy=(-40,80),xytext=(-85,80),arrowprops=dict(arrowstyle='->',color='.3'),ha='center',fontsize=9)
 ax.set(xlabel='x (mm)',ylabel='y (mm)',title='Ball-center trajectory, 0–600 s',xlim=(-100,100),ylim=(-100,100));ax.set_aspect('equal');ax.legend(loc='lower left',fontsize=7);ax.grid(alpha=.2)
 ax=axes[1]
 for i,(name,rows,xy,tail,center,metric) in enumerate(cases):
  delta=(xy[tail]-center)*1000
  ax.plot(delta[:,0],delta[:,1],c=colors[i],lw=.7,label=labels[i])
 ax.set(xlabel='x − final mean (µm)',ylabel='y − final mean (µm)',title='Residual motion, final 60 s');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=8)
 ax=axes[2]
 for i,(name,rows,xy,tail,center,metric) in enumerate(cases):
  ax.plot(rows[:,0],np.linalg.norm(xy-center,axis=1),c=colors[i],lw=1,label=labels[i])
 ax.set(xlabel='Time (s)',ylabel='Distance to final mean center (mm)',title='Settling of the ball center');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Viscous Z-spin damping: τz=15 s | original height 39.62 mm | reference magnet moment',fontsize=12)
 fig.savefig(OUT/'parked_trajectory_tau15.png',dpi=190)
 (OUT/'parked_trajectory_tau15_metrics.json').write_text(json.dumps([c[-1] for c in cases],indent=2))
 print(json.dumps([c[-1] for c in cases],indent=2))

if __name__=='__main__':main()
