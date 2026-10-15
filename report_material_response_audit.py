from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/material-response-20261013'
s=json.loads((OUT/'scan_summary.json').read_text());j=json.loads((OUT/'joint_scan_summary.json').read_text());long=json.loads((OUT/'joint_long_confirm.json').read_text());base=s['loops'][0]['force_rms_N']
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
mur=[q for q in s['loops'] if q['label']=='baseline' or q['label'].startswith('mur=')];mur=sorted(mur,key=lambda q:q['material'][0]);axes[0].semilogx([q['material'][0] for q in mur],[q['force_rms_N']/base for q in mur],'o-');axes[0].axhline(1,color='0.5',ls='--');axes[0].set(xlabel='mu_r',ylabel='Force RMS / baseline (fixed pose)',title='Increasing permeability saturates the sphere response')
for q in long:
 path=next(p for p in OUT.glob('*.json') if p.name not in ['joint_scan_summary.json','joint_long_confirm.json','scan_summary.json','static_well.json','stationary_simulation_summary.json'] and (lambda a:a.get('label')==q['label'] and a.get('seconds')==600)(json.loads(p.read_text())))
 rows=np.load(path.with_suffix('.npz'))['rows'];period=[];times=[]
 for lo in np.arange(0,rows[-1,0]-30,10):
  a=rows[(rows[:,0]>=lo)&(rows[:,0]<lo+30)];ang=np.unwrap(np.arctan2(a[:,3],a[:,2]));rate=np.polyfit(a[:,0]-a[:,0].mean(),ang,1)[0]
  if rate<0:times.append(lo+15);period.append(-2*np.pi/rate)
 axes[1].plot(times,period,label=q['label'].replace('600s joint ',''),alpha=.8)
axes[1].axhspan(2.416257,2.48588,color='green',alpha=.18,label='Video mean periods');axes[1].set(xlabel='Simulation time (s)',ylabel='Ball period (s), 30 s regression',title='Long-run confirmation of the five nearest candidates');axes[1].legend(fontsize=7);axes[1].set_ylim(2.3,4.5)
for a in axes:a.grid(alpha=.2)
fig.savefig(OUT/'material_response_summary.png',dpi=160);plt.close(fig)
validated=[q for q in long if q.get('stationary')];rec=dict(single_parameter_count=len(s['runs']),joint_parameter_count=len(j),long_confirmation_count=len(long),validated=validated,goal_period_s=[2.41625700075,2.48587994094],coarse_static_frequency_used_as_filter=False,production_moment_scale=1.,production_Beta=.056999326368103,production_material=s['baseline'])
(OUT/'final_summary.json').write_text(json.dumps(rec,indent=2))
text='''# 固定磁铁磁矩的材料参数探索，2026-10-13

结论：34 组单参数测试和 60 组联合测试中，没有找到已通过长期复核、且周期符合视频约 2.42–2.49 s 的候选。最近的五组进行 600 s、时间步长减半复核，四组在约 201–290 s 向外越过宽 LUT 的 175 mm 半径边界；另一组保持有界且窗口统计稳定，但周期为 3.18257 s，仍比长视频慢 28.03%。这是有限扫描的结果，不是不存在解的数学证明。

## 已恢复的正式配置

二维正式模型 Beta 从 0.17327795215903312 恢复为 0.056999326368103，对应每块磁铁 4.222663 A·m²。CPU FP64、FP32 Kahan 和四个 GPU 扫描模块均已重建，构建清单同步更新，正式启动配置检查通过。正式材料参数仍保留原 Q235 基线；测试材料值仅存在隔离诊断模块，不把未匹配候选写入默认值。

2026-10-13 用户明确重新授权材料探索，覆盖之前固定四个参数的探索限制。原研究基线保留。

## 扫描条件

磁矩始终为参考值 1 倍。球直径 12 mm、磁铁直径 28 mm、厚 15 mm、环半径 70 mm、球心到磁铁中心面 39.62 mm。盘速固定为长视频 199.83134 RPM；空间 k=4/0.07，M_ref=1.1e6 A/m、rho=7850 kg/m³。接触摩擦 0.02；滚阻低/高系数 1e-5/1e-4、线性阻尼 0.004；倾角固定为 0.123216 度。没有把每秒 5–10 次的口述径向频率作为过滤或拟合目标。

初态统一为半径 70 mm、v0*=-0.075（切向实际速度约 -0.1099 m/s），初始纯滚动自旋；球固定两圈盘子时间建立磁化，在盘相位 0.58 rad 释放。所有材料测试时间尺度相同，没有改变 somega 后直接复用不同单位初态的问题。扫描范围未覆盖所有可能的自旋、磁化与释放状态。

单参数范围：mu_r=2 至 1e6；Alpha_JA 为基线的 0.01–100 倍；C_JA=0 至 1；K 为基线的 0.01–100 倍。联合扫描含 36 组规则网格及 24 组拉丁超立方采样，涉及 mu_r、Alpha_JA、C_JA 和 K。每组先运行 180 s；筛选较接近的五组后，运行 600 s、dt*=0.00025。

## 600 秒复核

| mu_r | Alpha_JA | C_JA | K | 结果 |
|---:|---:|---:|---:|---|
| 1000 | 0.00213231 | 0.25 | 0.00161278 | 201.25 s 越界 |
| 650 | 0.00213231 | 0.20 | 0.00537593 | 203.67 s 越界 |
| 1000 | 0.00213231 | 0.20 | 0.00537593 | 218.99 s 越界 |
| 450 | 0.00213231 | 0.25 | 0.00537593 | 289.66 s 越界 |
| 1000 | 0.00213231 | 0.30 | 0.00161278 | 有界，周期 3.18257 s |

最后一组末尾三个 30 s 窗口周期为 3.18288、3.18273、3.18247 s，均半径约 76.02 mm；速度第一谐波半幅约 4.17%，也未匹配视频 6–8%。其半径略高于此前 65–75 mm 的先验范围。因此它是长时间有界的数值候选，不是完整实验拟合。

“稳定”在这里指末尾窗口周期变化 <1%、平均半径变化 <0.5 mm、反向运动占比 >99%，并在 600 s 内无越界；不是对无限时间稳定性的证明。失败前回归出的平均周期不作为成功候选。

## 哪个参数影响梯度力？

当前代码的磁力是 F*=gamma J_h^T(D h−E q_world)，其中 D=(mu_r−1)/(mu_r+2)，E=(1−C_JA)/[(1−Xi_JA)(mu_r+2)]。物理磁化 M=3 M_ref(D h−E q_world)。mu_r 影响诱导磁化与磁滞动力学，而不是独立力增益。

原 mu_r=605.897 时 D=0.995065，即便 mu_r→无穷，单独 D 项也只增加约 0.50%。固定位置、零自旋、 rotating-field 磁化两圈后比较：mu_r 增至 1e6，磁力 RMS 增加约 2.15%；完整单参数扫描中最大的磁力 RMS 增加约 3.88%。这些是该位置和磁化协议下的总力指标，不能当作所有运动状态、所有力分量的严格上界。磁滞参数可以显著改变转矩、相位和驱动功率，从而改变公转周期，即使总吸引力变化很小。

从诊断输出的 H、磁化 M 和梯度 J_H，独立以 SI 式 F=mu0 V J_H^T M 重建力，最大相对差约 6.2e-16；质量/体积/时间归一化没有发现漏乘因子。圆柱磁场及梯度查表与独立精确场比较、磁力方向原生检查均通过。这检验了数值与归一化的一致性，不等于证明球的本构模型准确。

## 静止圆盘的补充计算

固定圆盘角度为零，同时修正随体磁场导数中的盘速项，保留完整 JA、滚滑接触与阻尼，参考磁矩下 1 mm 径向偏移的初期频率约 1.448 Hz；3 mm 偏移约 1.435 Hz。mu_r=1e6 时约 1.379 Hz。原 3.04 倍磁矩对照约 4.405 Hz。该比较仅描述模型响应，不使用后来撤回的口述频率排除参数。

另计算 q=0 可逆基准的径向势阱：平衡半径约 69.816 mm、刚度 0.63346 N/m、到零场的势能深度约 1.438e-4 J；这些是指定磁化基准下的势能估计，磁滞运动不存在通用的单值保守势能。它们与完整 JA 振动计算不是同一近似。

## 后续方向

本轮结果不支持把 mu_r 当作放大梯度力几十倍的旋钮，也没有支持直接写入某组新磁滞参数。应继续把吸引力/径向刚度与磁滞驱动分开约束。可逆周期平均磁力的理想圆周力平衡在约 81.8 mm 处允许约 2.35 s 的周期，说明不能仅凭吸引力上界断言参考磁矩下目标周期在数学上绝无可能；但它没有验证驱动、阻力、稳定性或实际半径，也不是已有拟合解。

输出：scan_summary.json、joint_scan_summary.json、joint_long_confirm.json、final_summary.json、material_response_summary.png、stationary_simulation_summary.json 及保存轨迹。隔离模块源与构建散列保存在 src2d/scratch/material-response-20261013-source/audit_build.json；静止盘诊断在对应 static-source 目录。
'''
(OUT/'REPORT.md').write_text(text,encoding='utf-8')
