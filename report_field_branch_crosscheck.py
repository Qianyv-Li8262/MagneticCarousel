"""Summarize matched field/protocol comparisons, without conflating branches."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/field-branch-crosscheck-20261013'

def main():
 data=json.loads((OUT/'comparison.json').read_text())
 peak=max(r['relative_period_difference'] for r in data if r['relative_period_difference'] is not None)
 lines=['# 120 G 周期与旧结果：固定条件对拍','',
 '120 G 没有唯一的自旋衰减时间：之前扫描 τz=0–20 秒；2.532222 秒对应 τz=20 秒；τz=15 秒为 2.546013 秒。倾角为 0.0021505243734457795 rad≈0.123216°，沿 +y 下坡。这里的周期为绕盘公转，不是 Z 自旋周期。',
 '', '## 本次对拍条件','',
 '100、120、140、160、200、300 G；共同 τz=15 秒，120 G 额外核对 τz=20 秒。每组均从 R=70 mm、位置角 0、速度及三个自旋分量为零、单位姿态、源相位 0.58 开始。先固定球心/姿态经历两圈盘转动形成磁滞记忆。倾角、盘转速、材料参数、滚动/滑动阻力、参考磁矩全部一致。',
 '', '两种流程：直接释放；第 30 秒开始沿逆盘切向吹气，等效加速度 0.1 m/s²，持续 3 秒后归零。每组运行 1200 秒。场强指 R70 的 RMS，通过高度变化实现，不是仅把力乘系数。',
 '', '新实现：原生 C++ 线程池、精确阻尼分裂。旧实现：之前独立 Python 调用 C++ 扩展、RHS 内加入线性 Z 阻尼。两者使用完全同一个 LUT 文件和相同物理参数。共 14 对；没有运行额外高自旋初态。',
 '', f'公转周期最大相对差 {peak:.8g}，即 {peak*100:.6f}%。两边的局部停留结局也一致。少量差异包含周期估计窗口/采样及分裂积分差异。',
 '', '## 固定 τz=15 秒的结果','',
 '|场强/G|直接释放周期/s|吹气后周期/s|吹气后平均半径/mm|吹气后 Z 自旋/rad s⁻¹|','|---:|---:|---:|---:|---:|']
 for field in [100,120,140,160,200,300]:
  release=next(r for r in data if r['field_G']==field and r['tau_z_s']==15 and r['blow_accel_m_s2']==0)
  blow=next(r for r in data if r['field_G']==field and r['tau_z_s']==15 and r['blow_accel_m_s2']==.1)
  period='局部停留' if release['new_period_s'] is None else f"{release['new_period_s']:.6f}"
  lines.append(f"|{field}|{period}|{blow['new_period_s']:.6f}|{blow['new_radius_mm']:.5f}|{blow['new_spin_z']:.5f}|")
 lines+=['', '## 为什么场强增加后所见周期反而变长','',
 '本次结果支持分支/吸引域选择解释，而不是计算后端差异。最直接的证据是同一个 160 G、同一个阻尼时间：无吹气得到约 2.706 秒、Z 自旋 +86.77 rad/s；短吹后得到约 1.719 秒、Z 自旋 −90.00 rad/s，半径分别约 71.55 和 76.18 mm。吹气结束后它们仍维持不同运动。',
 '', '统一吹气流程也不保证不同场强落到同一连续分支：120 G 得到约 2.546 秒、Z 自旋 −9.72 rad/s；200 G 得到约 2.777 秒、Z 自旋 +85.49 rad/s，且平均半径不同。把这两个值直接当成同一分支的“周期随 B”变化会混淆运动状态。',
 '', '截图里的 200 G/300 G 直接释放周期在这次对拍中复现。100 G 的较弱长吹周期也复现。120 G 若不吹气，τz=15、20 秒均局部停留；2.532 秒是停止吹气后保持的公转状态。',
 '', '这是有限释放流程的分支证据，并未证明所有吸引子或各分支如何连续延拓。若要定量回答某一分支自身随场强怎样变化，应保留完整状态和磁滞记忆做缓慢高度延拓，不能每档都重新释放后连接曲线。',
 '', '## 逐组新旧周期比较','', '|场强/G|τz/s|吹气加速度/m s⁻²|新周期/s|旧周期/s|相对差/%|','|---:|---:|---:|---:|---:|---:|']
 for r in data:
  cells=['停留','停留','—'] if r['new_period_s'] is None else [f"{r['new_period_s']:.8f}",f"{r['old_period_s']:.8f}",f"{r['relative_period_difference']*100:.7f}"]
  lines.append(f"|{r['field_G']}|{r['tau_z_s']:g}|{r['blow_accel_m_s2']:g}|"+'|'.join(cells)+'|')
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
 for blow,label,color,marker in [(0.,'Release, no air pulse','tab:orange','s'),(.1,'3 s air pulse, then unforced','tab:blue','o')]:
  q=[r for r in data if r['tau_z_s']==15 and r['blow_accel_m_s2']==blow and r['new_period_s'] is not None]
  ax[0].scatter([r['field_G'] for r in q],[r['new_period_s'] for r in q],label=label,color=color,marker=marker,s=55)
  ax[0].scatter([r['field_G'] for r in q],[r['old_period_s'] for r in q],color='k',marker='+',s=25)
  ax[1].scatter([r['field_G'] for r in q],[r['new_spin_z'] for r in q],label=label,color=color,marker=marker,s=55)
 ax[0].axhline(2.49,color='.5',ls='--',label='Video ~2.49 s')
 ax[0].set(ylabel='Orbital period (s)',title='Matched scan: τz=15 s; crosses = old backend')
 ax[1].axhline(0,color='.5',lw=.8);ax[1].set(ylabel='Mean normal-axis spin (rad/s)',title='Different protocols can select different states')
 for panel in ax:panel.set_xlabel('Source field RMS at R=70 mm (G)');panel.grid(alpha=.25);panel.legend(fontsize=8)
 fig.savefig(OUT/'matched_field_scan.png',dpi=180)
 print(f'14 matched cases; max period relative error {peak*100:.7f}%')

if __name__=='__main__':main()
