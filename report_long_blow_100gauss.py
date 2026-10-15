"""Show whether a longer air pulse finds a faster unforced 100 G orbit."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import long_blow_100gauss as task
g=task.g;OUT=task.OUT

def main():
 records=json.loads((OUT/'scan_summary.json').read_text())
 lines=['# 100 G：延长吹气后是否有更快的稳定公转','',
 '保持初始速度和三个自旋分量为零、参考磁矩、100 G 高度 39.89015 mm、原倾角沿 +y 下坡。τz=12、15、23 秒。吹气在第 30 秒开始，沿逆盘切向通过球心施力，不直接施加 Z 轴力矩。',
 '', '新增 27 组：加速度 0.1、0.2、0.5 m/s²，各吹 1、3、10 秒；每组在停止吹气后继续测试，最多 600 秒。稳定判据与九组场强比较一致。场强是 R=70 mm 处 RMS，磁场及梯度均来自改变高度后的完整 LUT。',
 '', '|吹气加速度/m s⁻²|时长/s|τ=12 秒|τ=15 秒|τ=23 秒|','|---:|---:|---|---|---|']
 for acc in [.1,.2,.5]:
  for duration in [1.,3.,10.]:
   cells=[]
   for tau in [12.,15.,23.]:
    r=next(r for r in records if r['tau_z_s']==tau and r['protocol']==f'blow_a{acc:g}_d{duration:g}')
    if r['stable_clockwise_orbit']:cell=f"稳定 {r['period_s']:.5f} s"
    elif r['error']:cell=f"出界（吹气开始后 {r['completed_s']-30:.2f} s）"
    else:cell='回到局部停留'
    cells.append(cell)
   lines.append(f'|{acc:g}|{duration:g}|'+ '|'.join(cells)+'|')
 lines+=['','这些结果没有发现更快的稳定公转分支：成功维持的运动仍接近 5 秒，和此前短吹代表分支相差不到 0.1%；该分支此前已用半步长测试到 1200 秒。较大的持续推力使球离开磁环并越过 R=175 mm 的 LUT 外边界，不能将出界前的暂态绕行速度当作稳定周期。',
 '', '这只排除了本次通过球心的逆盘切向吹气范围，没有穷尽初态、力的方向和施加力矩的方式。当前 100 G 模型未能复现实验约 2.486 秒的公转；不能把这一模拟结果当作实验速度的上限。需继续区分磁力/磁滞建模问题和未找到的运动分支。',
 '', '另外两种场强、全部倾角与短吹对照见 ../low-spin-field-grid-20261013/REPORT.md。']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
 colors=['#347fa8','#309368','#d46b35']
 axes[0].add_patch(Circle((0,0),70,fill=False,color='.5',ls='--'))
 summaries=[]
 for duration,color in zip([1.,3.,10.],colors):
  r=next(r for r in records if r['tau_z_s']==15 and r['protocol']==f'blow_a0.1_d{duration:g}')
  for p in OUT.glob('*.json'):
   try:q=json.loads(p.read_text())
   except Exception:continue
   if isinstance(q,dict) and all(q.get(k)==r[k] for k in ['tau_z_s','protocol','seconds','dt']):
    rows=np.load(p.with_suffix('.npz'))['rows'];break
  xy=rows[:,2:4]/g.a.K*1000;radius=np.linalg.norm(xy,axis=1)
  speed=-(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.linalg.norm(rows[:,2:4],axis=1)*g.a.OMEGA/g.a.K
  label=f'Blow {duration:g} s'
  axes[0].plot(xy[:,0],xy[:,1],color=color,lw=.8,label=label)
  if r['error']:axes[0].scatter(*xy[-1],color=color,marker='x',s=45)
  part=(rows[:,0]>=28)&(rows[:,0]<=90)
  axes[1].plot(rows[part,0]-30,speed[part],color=color,lw=.8,label=label)
  axes[2].plot(rows[part,0]-30,radius[part],color=color,lw=.8,label=label)
  summaries.append(dict(duration_s=duration,stable=r['stable_clockwise_orbit'],period_s=r['period_s'] if r['stable_clockwise_orbit'] else None,completed_s=r['completed_s']))
 axes[0].set(xlabel='x (mm)',ylabel='y (mm)',title='Ball-center paths');axes[0].set_aspect('equal');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
 axes[1].axhline(2*np.pi*.07/2.48587994094,color='.4',ls='--',label='2.486 s at R=70 mm')
 axes[1].set(xlabel='Time since blowing starts (s)',ylabel='Clockwise tangential speed (m/s)',title='Speed during and after blowing');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
 axes[2].axhline(175,color='.4',ls='--',label='LUT outer boundary')
 axes[2].set(xlabel='Time since blowing starts (s)',ylabel='Radius (mm)',title='Longer push can leave the ring');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
 fig.suptitle('100 G | zero initial spin | τz=15 s | air acceleration 0.1 m/s²',fontsize=12)
 fig.savefig(OUT/'long_blow_trajectories.png',dpi=180)
 print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
