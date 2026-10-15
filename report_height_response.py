from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/height-response-20261013'
stats=json.loads((OUT/'field_stats.json').read_text());scan=json.loads((OUT/'scan_summary.json').read_text());long=json.loads((OUT/'long_confirm.json').read_text());ref=json.loads((OUT/'height_refine_summary.json').read_text());reflong=json.loads((OUT/'height_refine_long.json').read_text());matches=json.loads((OUT/'matched_periods.json').read_text());two=json.loads((OUT/'200gauss_result.json').read_text());uniform=json.loads((OUT/'200gauss_uniform_amplitude.json').read_text());surface=json.loads((OUT/'center_vs_surface_fields.json').read_text())
stats={float(k):v for k,v in stats.items()};ref['field_stats']={float(k):v for k,v in ref['field_stats'].items()}
fig,ax=plt.subplots(2,1,figsize=(11,8),layout='constrained');h=np.array(sorted(stats));rms=[stats[x][1]['field_rms_G'] for x in h];peak=[stats[x][1]['field_max_G'] for x in h]
ax[0].plot(h,rms,'o-',label='Ball-center field RMS at R=70 mm');ax[0].plot(h,peak,'o--',label='Ball-center peak');ax[0].axvspan(26.68,26.91,color='green',alpha=.12,label='Mean-period match');ax[0].axhline(200,color='0.5',ls=':');ax[0].set(xlabel='Ball center to magnet center plane (mm)',ylabel='External field magnitude (Gauss)',title='Fixed reference magnet moment; changing actual separation');ax[0].legend(fontsize=8)
records={q['height_mm']:q for q in scan}
for q in long+ref['runs']+reflong+[x['run'] for x in matches]+[two['run']]:records[q['height_mm']]=q
points=[]
for height,q in records.items():
 if q['error'] is None and q['stationary']:
  if height in stats:b=stats[height][1]['field_rms_G']
  elif height in ref['field_stats']:b=ref['field_stats'][height][1]['field_rms_G']
  elif abs(height-two['height_for_200G_rms_at_70mm'])<1e-8:b=200.
  else:b=next(x['field_stats'][1]['field_rms_G'] for x in matches if x['height_mm']==height)
  points.append([b,q['period_s']])
points=np.array(sorted(points));ax[1].plot(points[:,0],points[:,1],'o-',label='Height-adjusted branch');ax[1].scatter([200],[uniform['run']['period_s']],marker='s',s=65,label='200 G by source amplitude, original height');ax[1].axhspan(2.416257,2.48588,color='green',alpha=.14,label='Video mean-period band');ax[1].set(xlabel='Ball-center field RMS at R=70 mm (Gauss)',ylabel='Ball period (s)',title='Same field magnitude does not determine the gradient or period');ax[1].legend(fontsize=8)
for a in ax:a.grid(alpha=.2)
fig.savefig(OUT/'height_field_period.png',dpi=160);plt.close(fig)
report='''# 轨道场强、200 Gauss 与改变高度的周期预测

2026-10-13。计算使用固定参考磁铁磁矩 4.222663 A·m²、8 个交替圆柱磁铁（直径 28 mm、厚 15 mm、环半径 70 mm），球直径 12 mm。正式配置未改变。以下 Gauss 均表示未放入球时的外部磁感应强度模 |B_ext|，不是球内部 B，也不一定是单独 Bz 分量。

## 1. 当前是不是约 100 Gauss？

按球心到磁铁中心面 39.62 mm 计算，是。半径 70 mm 处，随磁铁盘相位变化，外场模为约 87.10–115.67 G，均值 101.53 G，均方根 102.03 G。半径 65–75 mm 内，各半径的均方根约 97.65–102.03 G。

这是模型估计，不是新的高斯计实测标定。若“轨道上”指探头放在球接触的上盘表面，该表面比球心低 6 mm，中心面距离为 33.62 mm，外部场均方根约 161.30 G、峰值约 188.65 G。球心、球底/盘面和磁铁表面读数不能混用。

## 2. 改成 200 Gauss 能否补上周期差距？

本报告统一用半径 70 mm、一个磁场周期内的均方根值来定义“200 G”，并分别测试两种实现：

| 场强实现 | 球心中心面距离 | 磁矩 | 600 s 预测周期 |
|---|---:|---:|---:|
| 原高度下将整个源场幅度放大到 200 G RMS | 39.62 mm | 参考值的 1.9603 倍，仅诊断 | 3.06090 s |
| 保持参考磁矩，减小间距到 200 G RMS | 30.90885 mm | 参考值不变 | 2.85007 s |

两次都保持有界并通过末尾窗口稳定性检查，但都慢于长视频 2.48588 s，分别约慢 23.13%、14.65%。这里没有将诊断中的放大磁矩写回正式模型。

在相同场形状的近线性诱导磁化情况下，F~M grad B，M~B，故 B 翻倍、梯度也同步翻倍时，磁力约变成四倍。但同样的场模不规定梯度；减小高度还改变了径向力曲线、转矩、法向载荷、摩擦和相对运动。因此不能直接根据 B² 或向心力平衡推断实际公转周期。

减小高度至 200 G RMS 时，70 mm 处峰值约 238.22 G。若 200 G 指峰值，则对应的 RMS 更低；不能与本表的 200 G RMS 等同。

## 3. 只允许改变高度，目标周期需要多少 Gauss？

沿用当前线性化 JA 运动模型、原材料参数、接触及滚动阻力、倾角，得到下列通过 600 s 复核的候选：

| 球心中心面距离 | 球心 RMS 场（70 mm） | 球心峰值场 | 平均公转周期 | 速度第一谐波半幅 |
|---|---:|---:|---:|---:|
| 26.91 mm | 约 277.81 G | 约 341.73 G | 2.48761 s | 10.43% |
| 26.68 mm | 约 283.25 G | 约 349.12 G | 2.41800 s | 10.04% |

因此这套参数下，匹配约 2.42–2.49 s 的平均周期，大致要求球心处 **280 G RMS，峰值约 340–350 G**，球心到磁铁中心面约 **26.7–26.9 mm**。相对当前 39.62 mm，需减小间距约 **12.7–12.9 mm**。

按此前尺寸关系 H=D−3 mm，D 为上盘上面到下盘下面的距离，这相当于将 D 从 42.62 mm 调到约 29.7–29.9 mm。磁铁顶面到球心约 19.2–19.4 mm。

若在这两组候选的盘面/球底高度测外场，70 mm 处 RMS 约 470–480 G，峰值约 611–626 G。这个盘面读数与上面的球心 280 G 不矛盾，是 6 mm 高度差造成的。

## 模型与验证范围

- 所有改变高度的动力学测试固定磁矩、Mu_r=605.8974、Alpha_JA=0.00213231、C_JA=0.22415155、K=0.00537593，保持球/磁铁实际尺寸与环半径不变。
- 盘速统一使用长视频 199.83134 RPM，倾角 0.123216 度；Mu=0.02，滚阻低/高系数 1e-5/1e-4，lambda=0.004。统一初态半径 70 mm、v0*=-0.075、初始纯滚动，两圈固定磁化后释放。
- 第二组只说明在相同约 200 RPM 下匹配短片数值周期，不是已按短片实测 201.815 RPM 完成联合拟合。
- 这些是当前线性化 JA 运动求解器的条件预测，尚未将完整 Langevin 非线性本构加入机械积分。此前完整本构力比较针对规定的固定姿态场历史，不能替代这次轨迹的完整非线性验证。
- 末尾三个 60 s 窗口周期差 <1%、均半径差 <0.5 mm、反向运动比例 >99%，且模拟 600 s 无越界，才记为稳定候选；不宣称无限时间稳定。
- 数值磁场的平均周期匹配不等于完整实验匹配。上述候选速度波动约 10%，仍高于原图/仿射视频目标 6–8%，慢周期调制也未作为此次拟合目标。不能把改变高度后的假设解当成实测 39.62 mm 装置的解释。

## 磁场与梯度数值核验

重新计算每个高度的有限圆柱 Hankel 积分，保留相同五个谐波、600 径向节点。FP64 Gauss 32 点分段积分加倍面板密度，在六个半径处的系数差约为浮点舍入量级；39.62 mm 表与先前高精度表的相对范数差约 6.8e-12。

细扫高度通过 A/Z 及其径向样条系数的高度 Hermite 插值生成，并与独立双端面磁荷积分核验。26.7–26.9 mm 附近外场相对范数差约 1e-4，最大场误差约 0.05 G；梯度相对范数差约 1.2–1.5e-4，最大梯度误差约 0.0007–0.0008 T/m。200 G 点同样完成独立场与梯度核验。Gauss 报告中列出的精细场值采用独立实空间积分，避免把插值精度当作实测有效数字。

输出：field_stats.json、center_vs_surface_fields.json、scan_summary.json、long_confirm.json、height_refine_summary.json、height_refine_long.json、matched_periods.json、200gauss_result.json、200gauss_uniform_amplitude.json、gradient_validation.json、height_field_period.png，以及各次轨迹 NPZ。脚本位于 src2d 的 generate_height_tables.py、height_response_audit.py、refine_height_response.py、match_height_period.py、test_200gauss_height.py。
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
(OUT/'final_summary.json').write_text(json.dumps(dict(current_center_rms_G=102.02728120593089,current_track_surface_rms_G=161.29727220026408,two_hundred_G_uniform_period_s=uniform['run']['period_s'],two_hundred_G_height_period_s=two['run']['period_s'],matched_period_cases=matches,mechanical_model='Existing linearized JA, not full nonlinear Langevin mechanical simulation',production_geometry_unchanged=True),indent=2))
