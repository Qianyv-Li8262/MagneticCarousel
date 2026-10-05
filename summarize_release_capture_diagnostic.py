"""Write the measured capture comparisons and a human-readable report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs'/'legacy-release-capture'
RESULTS = ROOT/'results'/'legacy-release-capture'
PLOTS = ROOT/'scratch'/'legacy-release-capture'
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

all_runs = [r for path in RUNS.glob('*/summary.json')
            for r in json.loads(path.read_text(encoding='utf-8'))['runs']]


def select(label, phi, rpm):
    return next(r for r in all_runs if r['label']==label and
                abs(r['phi0_rad']-phi)<1e-8 and r['rpm']==rpm and 'trace' in r)


fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout='constrained')
new = np.load(ROOT/'runs'/'release-stop-diagnostic'/'phi0.0000_dt0.001_T20.npz')
old = np.load(select('legacy_base', 0, 100)['trace'])
axes[0].plot(new['state_table'][:,0], new['angle_rad'], label='现有磁性参数 CJA=0.0198')
keep = old['state_table'][:,0] <= 20
axes[0].plot(old['state_table'][keep,0], old['displacement_rad'][keep], label='原磁性参数 CJA=0.38')
axes[0].set(title='只恢复整组磁性参数：持续反向绕行变为局部往返（100 rpm，初始方位 0）',
            xlabel='时间 / s', ylabel='相对释放位置 / rad')

for label, name in [('legacy_base', '原参数，μ=0.02'),
                    ('legacy_mu01', '原磁性参数，μ=0.01'),
                    ('legacy_halfK', '原参数，K 减半')]:
    trace = np.load(select(label, np.pi/2, 50)['trace'])
    axes[1].plot(trace['state_table'][:,0], trace['smoothed_displacement_rad'], label=name)
axes[1].axhline(1.35-np.pi, color='black', linestyle='--', linewidth=1,
               label='几何最低方位')
axes[1].set(title='捕获与过冲：50 rpm，初始方位 π/2（曲线平均掉磁铁经过的快速起伏）',
            xlabel='时间 / s', ylabel='相对释放位置 / rad')

for rpm in [50,100,150]:
    row = select('legacy_long',0,rpm)
    trace = np.load(row['trace'])
    axes[2].plot(trace['state_table'][:,0],trace['smoothed_displacement_rad'],label=f'{rpm} rpm')
axes[2].axhline(1.35-np.pi/2,color='black',linestyle='--',linewidth=1,label='几何最低方位')
axes[2].set(title='原参数：三种转速在同一片区域附近捕获（初始方位 0）',
            xlabel='时间 / s', ylabel='相对释放位置 / rad')
for ax in axes:
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
fig.suptitle('磁性参数回退与捕获诊断：5 mm 钢球、固定场高度 30 mm、倾角 0.02 rad')
fig.savefig(PLOTS/'capture_comparison.png',dpi=160)
plt.close(fig)

failures = [r for r in all_runs if 'trace' not in r]
failure_types = {}
for row in failures:
    status = row['error'].split('failed with status ')[1].split(';')[0]
    failure_types[status] = failure_types.get(status,0)+1
base = select('legacy_long',0,100)
fine = select('legacy_long_halfdt',0,100)
tuned = select('legacy_mu01',np.pi/2,50)
tuned_fine = select('legacy_mu01_halfdt',np.pi/2,50)
checks = dict(
    baseline_300s_equilibrium_difference_rad=abs(base['final_mean_displacement_rad']-fine['final_mean_displacement_rad']),
    baseline_300s_raw_span_difference_rad=abs(base['tail_raw_angle_span_rad']-fine['tail_raw_angle_span_rad']),
    tuned_60s_equilibrium_difference_rad=abs(tuned['final_mean_displacement_rad']-tuned_fine['final_mean_displacement_rad']),
    tuned_60s_min_displacement_difference_rad=abs(tuned['min_displacement_rad']-tuned_fine['min_displacement_rad']))
(RESULTS/'summary.json').write_text(json.dumps(dict(runs=all_runs,
    failure_status_counts=failure_types, half_dt_checks=checks),indent=2),encoding='utf-8')

table=[]
for rpm in [50,100,150]:
    r=select('legacy_long',0,rpm)
    table.append(f"| {rpm} | {r['final_mean_displacement_rad']:.6f} | {r['offset_from_downhill_rad']:.6f} | {r['tail_raw_angle_span_rad']:.6f} | {r['tail_smooth_angle_span_rad']:.6g} |")
report = '''# 旧磁性参数下的释放与捕获诊断

日期：2026-10-04。原生产源码和构建没有被覆盖；旧参数独立源码及可运行 FP64 构建位于 `src2d/scratch/legacy-release-source`。

## 参数与假设

从 `src/kernels/headers/hysteresis_config.hpp` 和 `research/q235/fit_results.json` 找回整组旧参数：
Mu_r=2292.6667，Alpha_JA=0.0032，C_JA=0.38，K=0.0010909091。
Beta、磁铁尺寸、8 个磁铁、中心面到球心高度约 30 mm 不变。球半径 5 mm，SR=5/6；初始平动、自转和 q 均为零。
采用现有倾斜修正，倾斜轴方位 1.35 rad、倾角 0.02 rad。几何最低方位为 1.35−π/2。
现有模型仍在固定高度取磁场，未新增位置相关间隙，也未新增滚动阻力、停止阈值或人为速度截断。

## 确认的行为

恢复整组旧磁性参数可把部分持续绕行轨迹变为局部过冲、往返及捕获。
这说明此前仅从缺少滚动阻力推断不捕获并不充分；磁性参数本身能改变驱动与耗散竞争。
捕获并非所有初态与转速的唯一结局：例如初始方位 π/2、100 rpm 的旧参数轨迹先绕行多圈，到 300 s 仍有约 1.39 rad 的末窗往返幅度。

同一初态（方位 0）下的三种转速，积分 300 s，末窗为最后约 5 s：

| rpm | 平均相对位移/rad | 相对几何最低点偏移/rad | 原始角度峰峰值/rad | 平滑角度峰峰值/rad |
|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

三种转速的平均捕获位置只差约 0.041 rad（2.9 mm 弧长，取半径 70 mm），但剩余周期运动不同。
原始信号仍存在磁铁经过造成的快速运动，不可把平滑曲线的平坦当成精确静止。
100 rpm 示例的原始角度峰峰值约 0.0122 rad（0.85 mm 弧长）；50 rpm 示例约 0.0509 rad（3.56 mm）。
小球平动速度仍不为零，存在径向快速运动；该结果是捕获附近的周期运动，而不是全部状态变量静止。

## 少量调参结果

比较接触摩擦 Mu=0.01、0.02、0.05、0.1，倾角 0.01、0.02、0.03、0.04 rad，
以及旧磁性参数的 K 减半和 C_JA 改为 0.6；并非全笛卡尔扫描。
各组尝试 50/100/150 rpm 与 0/π/2/π/3π/2 四个释放方位。

提高滑动摩擦至 0.05/0.1 不保证衰减更快，部分轨迹进入近乎全程纯滚动后往返更加持久。
降低 Mu 至 0.01 或将 K 减半，部分轨迹的慢往返衰减更快；但未复现所有实验条件下“一次反向走 1–2 rad 后立即停住且不返回”。
例如 50 rpm、初始 π/2、Mu=0.01 最终平均位移约 −2.1303 rad，但最初过冲至约 −3.0614 rad，随后仍返回。
因此这个调参例子说明可改变捕获和衰减，不能称为已经拟合实验释放过程。

## 数值验证与未完成项

原参数 100 rpm、初始方位 0 的 300 s 轨迹和 Mu=0.01、50 rpm、初始 π/2 的 60 s 轨迹，均做了 dt=0.001 与 0.0005 对照。
平均捕获位置差分别为 '''+f"{checks['baseline_300s_equilibrium_difference_rad']:.3g} rad、{checks['tuned_60s_equilibrium_difference_rad']:.3g} rad。"+'''
慢振荡平均使用两个瞬时吸引周期（0.15/SOMEGA 秒），末窗剔除滤波端点；原始轨迹同时保存。
部分探索轨迹失败：status 6 是离开 LUT 的有效半径范围，status 5 是释放时初始接触分支自洽性检查失败。
这些轨迹没有算成停止或捕获；具体失败记录保留在 summary.json。

## 对实验补充的解释

不同转速停在同一片区域，支持固定空间捕获区域。
压低红色支架 2–3 mm 后，红色附近从慢变快、对侧从快变慢，支持固定空间势能调制随倾斜改变。
持续绕行时低势能区域可有更高速度；捕获后同一区域又可以成为停留区域，两者不矛盾。
这个现象不能单独区分重力、间隙引起的磁吸引变化和板面形变，也不能作为已经测量停止点随上盘方向变化的证据。
简单纯滚动能量近似给出 v²(φ)≈2[E−U_eff(φ)]/[m(1+η)]，仅用于理解一圈内驱动/耗散变化较小的调制。
含磁滞和持续转动的完整系统不能由一个静态磁势能函数完全描述。

下一步应把实验的实际倾角/间隙、释放位置与磁化准备过程纳入约束，同时检验固定势阱与耗散能否解释首次到达就停住。
仅靠这次调参不能认定旧参数是材料真实磁性参数。
'''
(RESULTS/'report.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(successes=len(all_runs)-len(failures),failures=len(failures),
                     failure_status_counts=failure_types,checks=checks,report=str(RESULTS/'report.md'),
                     figure=str(PLOTS/'capture_comparison.png')),indent=2))
