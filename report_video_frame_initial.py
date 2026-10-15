"""Results of measured-frame initial states with high/low Z spin."""
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/video-frame-initial-20261013'

def main():
 info=json.loads((OUT/'frame_initial.json').read_text());rows=[]
 for path in [OUT/'results.csv',OUT/'tau20/results.csv']:
  with path.open() as f:rows.extend(csv.DictReader(f))
 grouped=defaultdict(list)
 for r in rows:grouped[(float(r['tau_z_s']),float(r['initial_spin_z_rad_s']))].append(r)
 groups=[]
 for (tau,z),group in sorted(grouped.items()):
  stable=[r for r in group if r['status']=='stable_cw'];periods=[float(r['period_s']) for r in stable]
  groups.append(dict(tau_z_s=tau,initial_spin_z_rad_s=z,count=len(group),outcomes=dict(Counter(r['status'] for r in group)),min_stable_period_s=min(periods) if periods else None,max_stable_period_s=max(periods) if periods else None))
 breakdown=[]
 for tau in [15.,20.]:
  for scale in ['ball_diameter','orbit70']:
   for roll in [0,1]:
    group=[r for r in rows if float(r['tau_z_s'])==tau and r['label'].startswith(scale+'__') and f'__roll{roll}__' in r['label'] and float(r['initial_spin_z_rad_s'])==0]
    breakdown.append(dict(tau_z_s=tau,scale=scale,plane_spin='pure rolling' if roll else 'zero',outcomes=dict(Counter(r['status'] for r in group))))
 normalization_error=0
 jobs=list(csv.DictReader((OUT/'jobs.csv').open()))
 for r in rows:
  job=jobs[int(r['task_id'])];x,y,vx,vy=[float(job[k]) for k in ['x_m','y_m','vx_m_s','vy_m_s']];radius=np.hypot(x,y)
  vr=(x*vx+y*vy)/radius;vt=(x*vy-y*vx)/radius
  normalization_error=max(normalization_error,abs(vr-float(r['initial_vr_m_s'])),abs(vt-float(r['initial_vt_m_s'])),abs(radius*1000-float(r['initial_radius_mm']))*.001)
 result=dict(total=len(rows),groups=groups,zero_spin_breakdown=breakdown,input_roundtrip_max_error_SI=normalization_error,fast_stable_cw_count=sum(r['status']=='stable_cw' and float(r['period_s'])<3 for r in rows),note='No external blowing. Existing reference disk RPM retained; positions/velocities normalized consistently in C++.')
 (OUT/'summary.json').write_text(json.dumps(result,indent=2))
 lines=['# 短视频单帧实测速度/位置初态测试','',
 f"取短视频第 {info['frame_index']} 帧，t={info['time_s']:.6f} s，使用邻近九帧的三阶平滑导数估计速度。没有从单张静止图直接声称测出速度。模板相关系数 {info['ncc']:.4f}；已有独立圆检测作位置/球径辅助。",'',
 '图像轨道椭圆修正后，坐标采用 x 向右/y 向上。绝对尺度未独立标定，所以同时测试球径 12 mm 的粗尺度、以及先前暂取平均轨道半径 70 mm 的尺度。',
 '', '|尺度|帧时刻半径/mm|球心速度/m s⁻¹|平均轨道半径/mm|','|---|---:|---:|---:|']
 for scale in ['ball_diameter','orbit70']:
  c=next(c for c in info['cases'] if c['scale']==scale)
  mean=info['implied_mean_orbit_mm_from_12mm_ball'] if scale=='ball_diameter' else 70
  lines.append(f"|{scale}|{c['radius_mm']:.5f}|{c['speed_m_s']:.6f}|{mean:.5f}|")
 lines+=['', '## 参数与初态假设','',
 '保持原视频构建高度 39.62 mm、参考磁矩、原材料及接触参数。R70 RMS 场强约 102.027 G；倾角 0.123216°，沿 +y 下坡。盘转速保持已有构建的 199.83134 RPM；短视频此前测得约 201.815 RPM，此轮未改变模型盘速，物理位置、速度和自旋始终用同一个模型 Ω 换算。',
 '', '每个尺度分别测试：原图像坐标方向；以速度第一谐波最大方向作为下坡方向的估计，将其转到 +y。后一方向不是独立的坡向测量，所以两个方向都保留。',
 '', 'Z 自旋为 −100、0、+100 rad/s；平面自旋分别为零、或按照 ωx=−vy/a，ωy=vx/a 设置纯滚动。球心速度和位置始终取同一帧，不因为自旋组而改变。',
 '', '随机磁盘角来自同一组 8 个相位，在三个自旋组之间配对复用。预磁化组固定球心/姿态经历两圈盘转动，随后才加入给定自旋；未预磁化 Q=0 对照用其中 4 个相位。姿态取单位四元数。',
 '', '无吹气。两种衰减时间 τz=15、20 秒，各 288 条，每条运行到 1200 秒或出界。原生 C++ 线程池使用 24 个线程。实际逐任务初态在 jobs.csv，归一化前的速度分量在 results.csv；输入回算最大误差为 '+f'{normalization_error:.3g} SI。',
 '', '## 结果','', '|τz/s|初始 Z 自旋/rad s⁻¹|条数|结局|持续逆盘周期范围/s|','|---:|---:|---:|---|---|']
 for g in groups:
  period='—' if g['min_stable_period_s'] is None else f"{g['min_stable_period_s']:.6f}–{g['max_stable_period_s']:.6f}"
  lines.append(f"|{g['tau_z_s']:g}|{g['initial_spin_z_rad_s']:g}|{g['count']}|{g['outcomes']}|{period}|")
 lines+=['', '高正/负自旋两组均出界；近零组部分持续逆盘公转，仍落到约 4.64 秒的慢运动，没有持续周期 <3 秒的候选。给入视频所测初速后未维持视频的约 2.416 秒周期。',
 '', '## 近零自旋的尺度/滚动对照','', '|τz/s|尺度|平面自旋|结局|','|---:|---|---|---|']
 for b in breakdown:lines.append(f"|{b['tau_z_s']:g}|{b['scale']}|{b['plane_spin']}|{b['outcomes']}|")
 lines+=['','两种磁滞准备方式都已测试，不能据这组结果证明磁滞历史永远不重要；这里只说明这两种准备没有找到所需快运动。',
 '', '这项测试削弱了“仅因之前初始平动速度太低而找不到快支”的解释，但不证明不存在其他吸引子。实际尺度、坡向、不可测的滚动/自旋与磁化状态，以及模型力和力矩仍可能与实验不同。尤其 70 mm 只是旧假设，球径粗标定给出平均约 83.78 mm，不应将其未经独立校准就作为精确测量。']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for tau,color in [(15,'tab:blue'),(20,'tab:orange')]:
  q=[r for r in rows if float(r['tau_z_s'])==tau and r['status']=='stable_cw']
  ax[0].scatter([float(r['initial_radius_mm']) for r in q],[float(r['period_s']) for r in q],s=16,alpha=.5,label=f'τz={tau} s',color=color)
 ax[0].axhline(2.416257,color='.4',ls='--',label='Short video ~2.416 s');ax[0].set(xlabel='Measured-frame initial radius (mm)',ylabel='Final orbital period (s)',title='Surviving motions return to slow period');ax[0].legend(fontsize=8);ax[0].grid(alpha=.25)
 x=np.arange(3);zvalues=[-100,0,100]
 counts=[sum(float(r['initial_spin_z_rad_s'])==z and r['status']=='stable_cw' for r in rows) for z in zvalues]
 escaped=[sum(float(r['initial_spin_z_rad_s'])==z and r['status']=='out_of_domain' for r in rows) for z in zvalues]
 ax[1].bar(x,counts,label='Sustained clockwise');ax[1].bar(x,escaped,bottom=counts,label='Out of domain',color='.7');ax[1].set(xticks=x,xticklabels=zvalues,xlabel='Initial Z spin (rad/s)',ylabel='Trajectories',title='Both damping times, 576 cases');ax[1].legend(fontsize=8)
 fig.savefig(OUT/'video_initial_results.png',dpi=180);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
