import sys,json,numpy as np
sys.path.insert(0,'src2d');import search_hysteresis15 as s
if __name__=='__main__':
 b=json.loads((s.OUT/'baseline_long_validation.json').read_text());tasks=[]
 for factor in [.98,1.02]:
  seed=np.array(b['final_state']);seed[3:5]*=factor;seed[9:11]*=factor
  tasks.append(dict(b['config'],seed=seed.tolist(),accel=0.,duration=0.,start=0.,seconds=600.,test_label='baseline_speed_'+str(factor)))
 for offset in [-.0005,.0005]:
  seed=np.array(b['final_state']);seed[1:3]*=1+offset/(np.linalg.norm(seed[1:3])/s.KS)
  tasks.append(dict(b['config'],seed=seed.tolist(),accel=0.,duration=0.,start=0.,seconds=600.,test_label='baseline_radius_'+str(offset)))
 c=json.loads((s.OUT/'1816308f00771308f922.json').read_text())['config'];tasks.append(dict(c,seconds=1500.,dt=.00025,test_label='near_target_long_check'))
 s.batch(tasks,'baseline_perturb_and_target_check')
