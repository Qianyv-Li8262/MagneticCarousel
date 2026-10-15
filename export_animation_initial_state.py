"""Export the current default animation release state and smoke-test rendering."""
import json
from pathlib import Path
import numpy as np
import animate_one_trajectory as animation

root=Path(__file__).resolve().parent
config=animation.read_build_config(kahan=False)
mod=animation.load_hysteresis()
values=animation.scaled_model_parameters(mod.magnet_number,somega=animation.ANIMATION_SOMEGA)
params=mod.SolverParams(*values)
state=animation.prepare_initial_state(mod,params,animation.DT_AT_EIGHT_MAGNETS,v0=animation.ANIMATION_V0)
labels=['disk_phase','x','y','vx','vy','qw','qx','qy','qz','omega_x','omega_y','omega_z','JA_qx_body','JA_qy_body','JA_qz_body']
record={'description':'Default animate_one_trajectory.py release; held for two disk turns, released with startup velocity v0=-0.075, no external push','disk_rpm':200,'ball_diameter_mm':12,'magnet_center_to_ball_center_mm':39.62,'tilt_deg':float(np.degrees(config['inclined_disk']['angle_rad'])),'downhill_direction':'+x (coordinate choice; laboratory azimuth not measured)','rolling_resistance_enabled':True,'v0':animation.ANIMATION_V0,'initial_ball_position_mm':[70,0],'initial_ball_velocity_mm_s':(state[3:5]*200*2*np.pi/60*17.5).tolist(),'initial_ball_spin_rad_s':(state[9:12]*200*2*np.pi/60).tolist(),'magnetize_turns':2,'magnetization_duration_s':.6,'initial_attitude_wxyz':[1,0,0,0],'state_order':labels,'release_state':state.tolist(),'state_named':dict(zip(labels,state.tolist())),'units':'State is dimensionless: length unit 17.5 mm, time unit 1/(200*2*pi/60) s. JA q is a body-frame magnetic internal state, not the attitude quaternion.','solver_parameters':dict(zip(['alpha','delta','gamma_g','gamma','gamma_m'],values)),'build_config':config}
out=root/'runs/laser-tilt-20261009/animation_initial_state_200rpm_rolling.json'
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
# Exercise the actual animation callbacks without leaving a window running.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ani=animation.animate(frames=range(5),expected_config=config)
for _ in range(5):ani._step()
fig.savefig(out.with_name('animation_start_200rpm_rolling.png'),dpi=100)
plt.close(fig)
print(json.dumps(record['state_named'],indent=2));print(out)
print('Five actual animation updates and rendering completed.')
