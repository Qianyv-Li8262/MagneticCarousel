"""Summarize the paired small-spin trajectories without counting transients."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'runs/video-frame-small-spin-20261013/paired-pilot'


def main():
    config = json.loads((OUT / 'config.json').read_text())
    rows = []
    for tau in config['taus_s']:
        with (OUT / f'tau{tau}/results.csv').open() as stream:
            group = list(csv.DictReader(stream))
        if len(group) != config['jobs_per_tau']:
            raise RuntimeError(f'tau {tau}: incomplete {len(group)}')
        rows.extend(group)
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row['tau_z_s']), float(row['initial_spin_z_rad_s']))].append(row)
    summary = []
    for (tau, spin), group in sorted(groups.items()):
        periodic = [r for r in group if r['status'] in ('stable_cw', 'stable_ccw', 'modulated_cw', 'modulated_ccw')]
        periods = [float(r['period_s']) for r in periodic]
        summary.append(dict(tau_s=tau, initial_z_rad_s=spin,
                            outcomes=dict(Counter(r['status'] for r in group)),
                            period_range_s=[min(periods), max(periods)] if periods else None))
    detail = defaultdict(list)
    for r in rows:
        label = r['label'].split('__')
        detail[(float(r['tau_z_s']), label[0], label[2], float(r['initial_spin_z_rad_s']))].append(r)
    detail = [dict(tau_s=k[0], scale=k[1], plane_spin=k[2], initial_z_rad_s=k[3],
                   outcomes=dict(Counter(r['status'] for r in g))) for k, g in sorted(detail.items())]
    escaped = [r for r in rows if r['status'] == 'out_of_domain']
    final_radii = [math.hypot(float(r['final_state_1']), float(r['final_state_2'])) / (4/.07) * 1000 for r in escaped]
    fast = [r for r in rows if r['status'] in ('stable_cw', 'modulated_cw') and float(r['period_s']) < 3]
    # The zero-spin paired controls must reproduce the existing scan exactly.
    control_error = 0.0
    source = OUT.parent.parent / 'video-frame-initial-20261013'
    for tau in config['taus_s']:
        oldpath = source / ('results.csv' if tau == 15 else f'tau{tau}/results.csv')
        with oldpath.open() as stream:
            old = {r['label']: r for r in csv.DictReader(stream)}
        for r in rows:
            if float(r['tau_z_s']) != tau or float(r['initial_spin_z_rad_s']) != 0:
                continue
            previous = old[r['label']]
            assert r['status'] == previous['status'], (r, previous)
            if r['status'] == 'stable_cw':
                control_error = max(control_error, abs(float(r['period_s']) - float(previous['period_s'])))
    summary_data = dict(total=len(rows), groups=summary, breakdown=detail,
                        fast_reverse_candidates=fast,
                        escape_final_radius_range_mm=[min(final_radii), max(final_radii)] if final_radii else None,
                        escape_time_range_s=[min(float(r['completed_s']) for r in escaped), max(float(r['completed_s']) for r in escaped)] if escaped else None,
                        overall_outcomes=dict(Counter(r['status'] for r in rows)),
                        zero_spin_control_max_period_error_s=control_error)
    (OUT / 'summary.json').write_text(json.dumps(summary_data, indent=2), encoding='utf-8')
    lines = ['# 视频初态的小 Z 自旋配对扫描', '',
             '沿用短视频第 180 帧的位置和邻帧估计速度，原构建高度 39.62 mm、R70 处 RMS 磁场约 102.027 G、参考磁矩、倾角 0.123216°，沿 +y 下坡。保持原盘速 199.83134 RPM。', '',
             '两种长度尺度（球径粗标定、旧的平均半径 70 mm 假设）、两种坐标方向、两种平面自旋（零、纯滚动）和两个随机磁盘相位。各自旋档使用完全相同的配对初态；磁滞准备固定为球心和姿态保持两圈盘转动。无吹气。', '',
             'Z 自旋为 0、±1、±3、±6、±10、±20、±40 rad/s；τz 为 15、20 s。共 416 条，各运行 1200 s 或至计算域边界。统计最后连续三个 120 s 窗口，同时检查角速度、半径和自旋是否收敛，并用整圈时间交叉检查周期。', '',
             '积分步长是无量纲 Δt=0.0005，对应物理时间约 23.89 μs；不应误写为 0.5 ms。', '',
             '|τz/s|初始 Z 自旋/rad s⁻¹|结局计数|持续公转周期范围/s|', '|---:|---:|---|---|']
    for g in summary:
        p = g['period_range_s']
        ptext = f'{p[0]:.6f}–{p[1]:.6f}' if p else '—'
        lines.append(f"|{g['tau_s']:g}|{g['initial_z_rad_s']:g}|{g['outcomes']}|{ptext}|")
    lines += ['', f"总计：{summary_data['overall_outcomes']}。持续逆盘周期 <3 s 的候选：{len(fast)}。", '',
              '出界表示场表不能继续求值，不能解释成离开盘面；模型没有计算竖直飞行。逃逸任务的最后有效半径范围：'+str(summary_data['escape_final_radius_range_mm'])+' mm。', '',
              '这是有限初态扫描，不能证明不存在其他分支。球径标定和坡向仍有估计误差。与 ±100 rad/s 的旧结果比较时，只能将旧结果理解为高自旋测试，不能推广到较小自旋。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for tau, color in [(15, 'tab:blue'), (20, 'tab:orange')]:
        q = [r for r in rows if float(r['tau_z_s']) == tau and r['status'].startswith(('stable_', 'modulated_'))]
        axes[0].scatter([float(r['initial_spin_z_rad_s']) for r in q], [float(r['period_s']) for r in q], s=20, alpha=.4, color=color, label=f'tau={tau} s')
        g = [g for g in summary if g['tau_s'] == tau]
        axes[1].plot([r['initial_z_rad_s'] for r in g], [r['outcomes'].get('out_of_domain', 0) for r in g], 'o-', color=color, label=f'tau={tau} s')
    axes[0].axhline(2.416257, color='.4', linestyle='--', label='Short video ~2.416 s')
    axes[0].set(xlabel='Initial Z spin (rad/s)', ylabel='Long-term orbital period (s)', title='Paired video-frame initial states')
    axes[1].set(xlabel='Initial Z spin (rad/s)', ylabel='Escaped cases / 16', title='Computational-domain exits', ylim=(-.5, 16.5))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(OUT / 'small_spin_results.png', dpi=180)
    print(json.dumps({k: v for k, v in summary_data.items() if k not in ('breakdown', 'fast_reverse_candidates')}, indent=2))


if __name__ == '__main__':
    main()
