"""Summarize fixed-material, 1.5x-moment rolling/blow searches."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/rolling15-search-20261015'
names=['screen','kick','boundary','fine','sharp','basin','crosscheck','crosscheck_sharp']
records={}
counts={}
for name in names:
    a=json.loads((OUT/(name+'.json')).read_text());counts[name]=len(a)
    records.update({r['key']:r for r in a})
rows=list(records.values())
checks=json.loads((OUT/'crosscheck.json').read_text())
sharp=json.loads((OUT/'crosscheck_sharp.json').read_text())
chosen=[checks[2],checks[1],checks[4],checks[5],sharp[1]]
labels=['Constant 0.001305: long transient','Constant 0.001330: slow drift',
        'Gaussian decay: outward transient','Gaussian decay: slow drift',
        'Sharp decay (power 8): outward transient']
fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
for rec,label in zip(chosen,labels):
    q=np.load(OUT/(rec['key']+'.npz'))['rows'];om=rec['config']['rpm']*2*np.pi/60
    n=max(1,len(q)//12000);z=q[::n]
    line,=axes[0].plot(z[:,0],np.linalg.norm(z[:,4:6],axis=1)*om/(4/.07),lw=1,label=label)
    axes[1].plot(z[:,0],np.linalg.norm(z[:,2:4],axis=1)/(4/.07)*1000,lw=1,color=line.get_color())
    phi=-np.unwrap(np.arctan2(q[:,3],q[:,2]));phi-=phi[0]
    indices=np.where((phi[:-1]<2*np.pi*np.floor(phi[1:]/(2*np.pi))) & (phi[1:]>phi[:-1]))[0]
    crossings=[]
    for i in indices:
        level=2*np.pi*np.floor(phi[i+1]/(2*np.pi))
        if level<=0:continue
        crossings.append(float(q[i,0]+(level-phi[i])/(phi[i+1]-phi[i])*(q[i+1,0]-q[i,0])))
    crossings=np.array(crossings)
    if len(crossings)>1:axes[2].plot((crossings[1:]+crossings[:-1])/2,np.diff(crossings),'.-',ms=2,lw=.7,color=line.get_color())
axes[0].set(ylabel='Centre speed (m/s)',xlim=(0,320));axes[0].legend(fontsize=8,ncol=2)
axes[1].set(ylabel='Orbit radius (mm)',xlim=(0,320));axes[1].axhline(105,color='k',ls=':',label='LUT boundary')
axes[2].set(ylabel='Full-lap time (s)',xlabel='Time after release (s)',xlim=(0,320),ylim=(1,5))
axes[2].axhline(2.416257,color='k',ls='--',lw=1,label='Video: 2.416 s');axes[2].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('1.5x magnet moment; fixed measured material; finite blowing only\nLong transients near 2.4 s do not settle into a fast orbit')
fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)

speed=np.linspace(0,.25,500);omega=200*2*np.pi/60
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
for rec,label in zip(chosen,labels):
    c=rec['config'];w=speed/(.006*omega)
    mu=c['high']+(c['low']-c['high'])*np.exp(-(w/c['transition'])**c.get('exponent',2))
    ax.plot(speed,mu,label=label)
ax.axhline(.0049,color='k',ls='--',lw=1,label='Measured constant reference')
ax.axvspan(.169,.195,color='gray',alpha=.15,label='Video mean-speed estimate')
ax.set(xlabel='Equivalent pure-rolling centre speed (m/s)',ylabel='Rolling coefficient',ylim=(0,.0052))
ax.grid(alpha=.25);ax.legend(fontsize=7);fig.savefig(OUT/'rolling_laws.png',dpi=180);plt.close(fig)

summary=dict(counts=counts,unique_runs=len(rows),fast_stationary_count=sum(bool(r.get('stationary')) for r in rows),
             fixed=dict(beta=.0854985,magnet_scale_vs_original=.0854985/.056999326368103,
                        cylinder_moment_Am2=4.222663*.0854985/.056999326368103,rpm=200,contact_mu=.2,
                        material=[605.8974337495617,.0021323083849413885,.22415154978863677,.005375927782155882],
                        tilt=False,added_normal_spin_damping=False),
             representative_keys=[r['key'] for r in chosen],formal_equivalence=json.loads((OUT/'formal_equivalence.json').read_text()))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
text=f'''# 1.5 倍磁矩：固定材料参数的滚动阻力与吹气搜索

## 结果

共完成 {len(rows)} 个不同输入/步长/计算时长的数值运行，**未找到撤去吹气后接近视频 2.416 s 周期且长期稳定的绕圈解**。
这只说明本次扫描没有找到，不是不存在此类解的数学证明。慢漂移与接近 2.4 s 的长暂态均不计作成功。

## 固定条件

- 使用当前正式源码和正式有限圆柱磁场表，8 个磁体，球直径 12 mm，磁环半径 70 mm，球心高度差 39.62 mm。
- Beta=0.0854985，为原始值的 {summary['fixed']['magnet_scale_vs_original']:.8f} 倍；单磁体磁矩约 {summary['fixed']['cylinder_moment_Am2']:.6f} A m²。
- 盘速 200 rpm，水平盘；接触摩擦系数 0.20。
- 材料参数严格保留正式值：Mu_r=605.8974337495617，Alpha_JA=0.0021323083849413885，C_JA=0.22415154978863677，K=0.005375927782155882。
- 没有加入法向自转阻尼；没有调整磁矩、磁体高度或材料参数。
- 使用当前动画默认释放条件 v0=-0.05（无量纲）、初始盘角 0.58；释放前保持球两圈以初始化磁滞记忆。
- 仅在隔离副本中开放滚动参数；正式源码、正式构建和测量恢复基线均未改动。

## 两类假设和吹气

恒定模型：C(w)=C_const。速度衰减模型：

    C(w)=C_high+(0.0049-C_high)*exp[-(w/wc)^p]

w 为球面内自转角速度除以盘角速度，p=2、4 或 8；纯滚时 v=R*Omega_disk*w。系数仍乘实际法向载荷 N，原模型的阻力矩为 C*N*R；“恒定”是指系数不依赖速度，磁性法向载荷仍可能变化。线性阻尼项始终为零。

扫描包括恒定系数 0–0.0049，并在 0.00128–0.00140 以 0.000005 间隔细扫；衰减模型高速极限 0.0002、0.0005、0.001、0.0013、0.0016，自转过渡尺度 0.3、0.6、1.0、1.4、1.5（不同组合在各批次中列出）。

吹气沿反向切线、力通过球心，没有直接施加转矩；加速度幅度 0.03–0.3 m/s²，对应约 0.213–2.131 mN；有限作用 1–15 s，开始时间测试了释放后 0、10、30 s。另有无吹气对照。阶段边界按积分步精确切换，撤去后外力设为零。

更陡的 p=8 模型可使低速系数更接近 0.0049；但它仍是假设，不能由本次拟合宣称已被实验确认，尤其原测量已经覆盖到 0.218 m/s。

## 代表结果：减半步长复核

| 模型 | 参数 | 吹气起点/加速度/持续时间，SI | 结果 |
|---|---|---|---|
| 恒定 | C=0.001305 | 10 s / 0.15 m/s² / 2.5 s | 到约 248.26 s 越出磁场表；尾段平均周期约 2.448 s，但三个时间窗角速度相差约 24%，半径持续变化 |
| 恒定 | C=0.001325 | 10 s / 0.1 m/s² / 3 s | 到约 215.24 s 越出磁场表；尾段平均周期约 2.362 s，并未稳定 |
| 恒定 | C=0.001330 | 10 s / 0.1 m/s² / 3 s | 运行到 1200 s，回到约 89.42 s/圈的慢漂移，半径约 69.16 mm |
| Gaussian 衰减 | low=0.0049, high=0.001, wc=0.6, p=2 | 10 s / 0.3 m/s² / 1 s | 约 53.12 s 越出磁场表 |
| Gaussian 衰减 | low=0.0049, high=0.001, wc=1, p=2 | 10 s / 0.1 m/s² / 3 s | 运行到 1200 s，回到约 307.47 s/圈的慢漂移 |
| 陡峭衰减 | low=0.0049, high=0.001, wc=1, p=8 | 10 s / 0.3 m/s² / 1 s | 约 44.76 s 越出磁场表；尾段平均周期约 2.343 s，仍非稳定解 |

“越出磁场表”指半径接近 105 mm，或积分试探位置越界，之后停止计算。它不能被直接当作已模拟到物理逃逸，更不能把停止前短片段平均周期当作稳定周期。

## 数值检查和判据

主扫描 dt=0.0005，8 个代表案例以 dt=0.00025 复算；这些是无量纲时间步，对应约 23.87 和 11.94 微秒。长时对照延长到 1200 s。

正式构建和隔离构建在实测恒定阻力下，保持初始磁化后继续演化 100000 步，完整初态和终态的最大绝对差均为 0，接触模式一致。见 formal_equivalence.json。

快分支判据：撤去吹气后持续运行；尾段三个时间窗角速度相对极差小于 0.5%，平均半径极差小于 0.5 mm，反向运动占比大于 99%，平均反向角速度大于 0.1 rad/s。该判据只筛选持续稳定绕圈候选；即使通过，也还需更长时间和扰动验证才能称为稳定周期吸引子。本次没有候选通过。

## 可复现资料

- summary.json：固定参数、运行数量和结果摘要。
- screen.json、kick.json、boundary.json、fine.json、sharp.json、basin.json：完整扫描输入与输出。
- crosscheck.json、crosscheck_sharp.json：半步长复核。
- 各 key.json / key.npz：全部参数、二进制/磁场表哈希、错误诊断和完整时间轨迹。
- comparison.png：速度、半径和逐圈周期；rolling_laws.png：代表假设曲线。
- 源码：src2d/search_rolling15.py；隔离构建：src2d/prepare_rolling15_search.py 和 prepare_rolling15_sharp.py。

当前证据支持的表述是：单纯降低阻力可产生接近视频速度的长暂态，但在本次固定材料模型及扫描范围内，没有找到撤去吹气后保持的快周期解。
'''
(OUT/'REPORT.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2))
