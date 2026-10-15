"""Compare unforced branches after finite pushes, retaining failure evidence."""
import hashlib
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import search_blow_fast_branch as b

def read(name):return json.loads((b.OUT/(name+'.json')).read_text())
records=read('scan')+read('refined');verified=read('validated')
def outcome(r):
    if r.get('error'):
        return 'outside_field_domain' if 'kStepLUTOutOfDomain' in r['error'] else 'solver_error'
    return r['outcome']
counts=dict(Counter(outcome(r) for r in records))
good=[r for r in records if r.get('stationary')]
vg=[r for r in verified if r.get('stationary')]
fast=[r for r in vg if r['period_s']<.99*b.BASE['period_s']]
byphase={i:[r for r in records if r['seed_index']==i] for i in range(4)}
summary=dict(baseline=b.BASE,run_count=len(records),outcomes=counts,
             verified=verified,verified_fast_candidates=fast,
             settled_period_range_s=[min(r['period_s'] for r in good),max(r['period_s'] for r in good)],
             verified_period_range_s=[min(r['period_s'] for r in vg),max(r['period_s'] for r in vg)],
             verified_radius_range_mm=[min(r['radius_mean_mm'] for r in vg),max(r['radius_mean_mm'] for r in vg)],
             verified_spin_rms_range_rad_s=[min(r['normal_spin_rms_rad_s'] for r in vg),max(r['normal_spin_rms_rad_s'] for r in vg)],
             phase_outcomes={i:dict(Counter(outcome(r) for r in rs)) for i,rs in byphase.items()},
             workflow_sha256={n:hashlib.sha256((b.f.ROOT/n).read_bytes()).hexdigest() for n in ['search_blow_fast_branch.py','report_blow_fast_branch.py']})
(b.OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 当前 2.76972 s 候选：吹气后快分支搜索','',
       '**本轮没有找到撤力后能维持的快分支。**' if not fast else '**发现需进一步核验的快周期候选，见 summary.json。**','',
       f"共 {len(records)} 次扫描：{len(good)} 次保持有界、撤力后回到原周期附近，{counts.get('outside_field_domain',0)} 次触及磁场表外边界；其余分类见 summary.json。",
       f"有界组撤力后周期范围 {summary['settled_period_range_s'][0]:.9f}–{summary['settled_period_range_s'][1]:.9f} s。",'',
       '## 固定模型与实际施力方式','',
       '固定上一轮已验证候选的全部磁滞、滚阻及自旋阻尼参数；高度 39.62 mm、每块磁铁磁矩约 4.222663 A·m²、水平盘、下盘恒定 199.831340 RPM。未重新调参。完整值与二进制/磁场表哈希见 manifest.json。',
       '从该候选 1200 秒轨迹的最后一圈中取四个时间点，间隔约四分之一公转周期，保留每个时间点完整的位置、速度、姿态、三轴自旋、磁滞记忆与磁盘相位。不是重新释放，也不重置 q 或自旋。',
       '吹气理想化为通过球心、沿当前轨道逆盘切向的有限外力 F=ma，方向随球心位置变化；不直接加转矩。只允许摩擦与磁力矩自然改变自旋。吹气结束后外力精确归零，继续计算 600 秒。外力只影响吹气阶段，不进入最后自由运动的统计。',
       '外力强度用等效平动加速度 a 表示；a×时长只是无其他力、自由质点的参考速度增量，实际速度变化还受摩擦、滚动约束和磁力影响。', '',
       '## 覆盖范围','',
       '- 基础网格：a=0.001、0.003、0.01、0.03、0.1、0.3 m/s²；时长 0.1、0.3、1、3、10 秒；四个稳定轨道时间点，另加四条零吹气对照。',
       '- 局部细化：a=0.003、0.01、0.03、0.1 m/s²，时长 0.5、1.5、5 秒。',
       '- 接近目标速度所需增量的脉冲：(a,时长)=(0.001,20)、(0.003,7)、(0.01,2)、(0.03,0.7)、(0.1,0.2)、(0.3,0.07)，单位 m/s²、秒。',
       '- 温和长吹：a=0.0003、0.001、0.003 m/s²，持续 30、60、120 秒。上述各项均测四个时间点。','',
       '## 代表结果','',
       '下面列第一个时间点的结果；全部四个时间点的逐条记录见 scan.json、refined.json。','',
       '|加速度/m s⁻²|持续/s|撤力后周期/s|结果|', '|---:|---:|---:|---|']
selected=[]
for a,d in [(0,0),(.001,10),(.003,7),(.01,2),(.01,3),(.0003,120),(.001,120)]:
    r=next(r for r in records if r['seed_index']==0 and r['acceleration_m_s2']==a and r['push_duration_s']==d)
    selected.append(r)
    period=f"{r['period_s']:.8f}" if r.get('stationary') else '—'
    label='回到原周期附近' if r.get('stationary') else ('向外触及磁场计算边界' if outcome(r)=='outside_field_domain' else outcome(r))
    lines.append(f'|{a:g}|{d:g}|{period}|{label}|')
lines += ['', '## 长期与步长复核','',
          f"额外复核 {len(verified)} 条，包含零力对照、最快的有界结果、最强仍有界的脉冲、最长仍有界的长吹和临近失去约束的样本。撤力后改算 1200 秒，归一化步长由 0.0005 减半为 0.00025。",'',
          f"复核中有界组周期范围 {summary['verified_period_range_s'][0]:.9f}–{summary['verified_period_range_s'][1]:.9f} s，平均半径范围 {summary['verified_radius_range_mm'][0]:.6f}–{summary['verified_radius_range_mm'][1]:.6f} mm，Z 自旋 RMS 范围 {summary['verified_spin_rms_range_rad_s'][0]:.6f}–{summary['verified_spin_rms_range_rad_s'][1]:.6f} rad/s。",'',
          '稳态判据：撤力后末尾 180 秒分三窗口，平均角速度变化 <0.5%、平均半径变化 <0.5 mm、逆盘运动占比 >99%；“快候选”还要求周期比基线至少缩短 1%。这只是有限时长统计，不能证明严格吸引子。',
          '“触及计算边界”指球心轨道向外扩张至约 175 mm 的磁场表边界，求解器停止；没有模拟竖直飞离盘面。失败前短暂的高速或吹气期间的周期不作为快稳态证据。',
          '本轮只覆盖这种通过球心的逆盘切向吹气与所列范围，未穷尽其他力方向、偏心吹气引入的直接力矩、其他初态或所有分支。结果支持“这批吹气未跨入快分支”，不能据此证明快分支不存在。', '',
          '## 复现','',
          '用 daily Python 依次运行 search_blow_fast_branch.py scan、refine、verify，再运行 report_blow_fast_branch.py。每条结果的同名 NPZ 保存全部轨迹：列为 [time_s,15 个状态分量,sliding,施加的加速度_m_s2]，状态顺序同上一轮。每条 JSON 保存初始完整状态、实际推力、时长、失败阶段、末尾窗口及构建哈希。','']
(b.OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

fig,ax=plt.subplots(2,1,figsize=(10,6.5),layout='constrained',sharex=True)
for a,d in [(0,0),(.003,7),(.0003,120)]:
    r=next(r for r in records if r['seed_index']==0 and r['acceleration_m_s2']==a and r['push_duration_s']==d)
    if not r.get('stationary'):continue
    rows=np.load(b.OUT/(r['key']+'.npz'))['rows'];free=rows[rows[:,0]>=r['push_actual_s']]
    t=free[:,0]-r['push_actual_s'];phi=np.unwrap(np.arctan2(free[:,3],free[:,2]));tc=[];period=[]
    for lo in np.arange(0,t[-1]-10,10):
        ok=(t>=lo)&(t<lo+10);rate=np.polyfit(t[ok]-t[ok].mean(),phi[ok],1)[0]
        tc.append(lo+5);period.append(-2*np.pi/rate if rate<-.01 else np.nan)
    label='No push' if a==0 else f'a={a:g} m/s2, duration={d:g} s'
    ax[0].plot(tc,period,label=label)
    ax[1].plot(t[::5],np.linalg.norm(free[::5,2:4],axis=1)/b.f.KSPACE*1000,label=label)
ax[0].axhspan(2.416257,2.485880,color='orange',alpha=.3,label='Measured mean periods')
ax[0].set(ylabel='Orbital period (s)',title='After force withdrawal: fixed model, different finite pushes')
ax[1].set(xlabel='Time after force withdrawal (s)',ylabel='Orbit radius (mm)')
for a in ax:a.legend();a.grid(alpha=.2)
images=b.f.ROOT/'scratch/blow-fast-branch-20261015';images.mkdir(exist_ok=True)
fig.savefig(images/'post_withdrawal.png',dpi=170)
print(json.dumps({k:summary[k] for k in ['run_count','outcomes','settled_period_range_s','verified_period_range_s','verified_radius_range_mm','verified_spin_rms_range_rad_s']},indent=2))
