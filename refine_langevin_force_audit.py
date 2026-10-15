import json,numpy as np
import langevin_force_audit as a
if __name__=='__main__':
 records=[];arrays={}
 for full in [False,True]:
  q,r=a.solve_case(70,full,cycles=128);records.append(q);arrays[full]=r
 q,r=a.solve_case(70,True,rtol=5e-10,max_step=.03,cycles=128);records.append(q)
 default=records[1];lin=records[0];record=dict(records=records,force_ratio_full_to_linear=default['force_rms_N']/lin['force_rms_N'],normal_ratio_full_to_linear=default['normal_attraction_mean_N']/lin['normal_attraction_mean_N'],torque_ratio_full_to_linear=default['torque_rms_N_m']/lin['torque_rms_N_m'],force_refinement_relative_difference=q['force_rms_N']/default['force_rms_N']-1,torque_refinement_relative_difference=q['torque_rms_N_m']/default['torque_rms_N_m']-1)
 (a.OUT/'refinement.json').write_text(json.dumps(record,indent=2))
 fig,ax=a.plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for full,label in [(False,'Linear'),(True,'Langevin')]:
  r=arrays[full];ax[0,0].plot(r[:,0],r[:,7]*1000,label=label);ax[0,1].plot(r[:,0],-r[:,9]*1000,label=label);ax[1,0].plot(r[:,1]/1000,r[:,4]/1000,label=label);s,p=a.static_curve(full);ax[1,1].plot(p[:,0],p[:,1]*1000,label=label)
 ax[0,0].set(xlabel='Source field phase (rad)',ylabel='Radial force (mN)',title='128 source-field cycles, 70 mm fixed position');ax[0,1].set(xlabel='Source field phase (rad)',ylabel='Downward magnetic force (mN)');ax[1,0].set(xlabel='External Hx (kA/m)',ylabel='Ball Mx (kA/m)',title='Self-consistent magnetization, sphere N=1/3');ax[1,1].set(xlabel='Ball radius (mm)',ylabel='Radial force (mN)',title='Stopped source, reversible Q=0 reference')
 for x in ax.ravel():x.grid(alpha=.2);x.legend()
 fig.savefig(a.OUT/'linear_vs_langevin_force.png',dpi=160);a.plt.close(fig)
 print('REFINEMENT',json.dumps({k:v for k,v in record.items() if k!='records'}),flush=True)
