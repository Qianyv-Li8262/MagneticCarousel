"""Package a small-tilt partial candidate with its immutable matching binary."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np


ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'prepared-release-fit'
DEST=ROOT/'results'/'small-tilt-release-fit'


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',type=Path,required=True)
    p.add_argument('--runtime-trace',type=Path);args=p.parse_args()
    records=json.loads(args.verify.read_text(encoding='utf-8'))
    r=next(r for r in records if r['turns']==2 and r['dt']==.0005)
    DEST.mkdir(parents=True,exist_ok=True)
    source=ROOT/'scratch'/'legacy-release-source'/'build'/'release'/'extensions'/'fp64'
    shutil.copytree(source,DEST/'extension',dirs_exist_ok=True)
    manifest=json.loads((DEST/'extension'/'build_config.json').read_text(encoding='utf-8'))
    names=['Mu_r','Alpha_JA','C_JA','K','kTiltAngleRad','kTiltAxisAzimuthRad']
    case=dict(schema_version=2,status='small_tilt_partial_candidate_not_full_experimental_fit',
         extension_directory='results/small-tilt-release-fit/extension',binary_sha256=manifest['binary_sha256'],
         magnetic_and_tilt_parameters=dict(zip(names,r['settings'])),tilt_degrees=float(r['settings'][4]*180/np.pi),
         initial_state_order=['disk_angle','x','y','vx','vy','quaternion_w','quaternion_x','quaternion_y','quaternion_z',
                              'omega_x','omega_y','omega_z','q_x','q_y','q_z'],
         initial_state=r['initial_state'],initial_sliding=r['initial_sliding'],
         initial_sliding_direction=r['initial_sliding_direction'],solver_parameters=dict(sr=5/6,somega=1),
         disk_rpm=100,
         disk_angular_speed_rad_s=100*2*np.pi/60,dt=.0005,sample_stride=20,
         pre_magnetization_disk_revolutions=2,
         pre_magnetization='fixed center/attitude; original JA equation; zero initial center speed and spin',
         mechanical_constants_unchanged=dict(Beta=.056999326368103,Eta=.4,Mu=.02),
         geometry_unchanged=manifest['magnet_geometry'],field_parameters_unchanged=manifest['field_parameters'],
         full_experimental_fit=False,metric_record=r)
    (DEST/'release_case.json').write_text(json.dumps(case,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(r['trace'],DEST/'trajectory.npz')
    shutil.copy2(RUN/'candidate_config.hpp',DEST/'candidate_config.hpp')
    shutil.copy2(args.verify,DEST/'verification.json')
    counts={}
    for name in ['small_tilt_c38','small_tilt_free','small_tilt_radius','small_capture_refine','small_capture_radius_grid',
                 'small_capture_optimize','small_capture_c38_grid','small_capture_weaker_k']:
        path=RUN/(name+'.json')
        if path.exists():
            rows=json.loads(path.read_text(encoding='utf-8'))
            counts[name]=dict(total=len(rows),numerical_ok=sum(r['status']=='ok' for r in rows),
                              full_criterion_passes=sum(r.get('passes',False) for r in rows))
    (DEST/'search_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    escape_path=RUN/'small_final_escape.json'
    escape_note=''
    if escape_path.exists():
        escape_rows=json.loads(escape_path.read_text(encoding='utf-8'))
        shutil.copy2(escape_path,DEST/'after_capture_impulses.json')
        escape_note='''\n捕获60秒后，中心叠加5、10、20、40、80、160 mm/s的反向速度，继续推进50秒，均重新捕获。320 mm/s的强推动则使球在50秒中净反向行进约25.05圈，但最后10秒纯滚动分支占比仅约0.88%，主要仍是接触滑动，因此不能称为实验中持续反向纯滚动的复现。强推动后的轨道半径也扩大。详见 `after_capture_impulses.json`。\n'''
    half=np.load(r['trace']);whole=np.load(next(q['trace'] for q in records if q['turns']==2 and q['dt']==.001))
    xyhalf=half['position_mm'];xywhole=whole['position_mm']
    interp=np.column_stack([np.interp(whole['time_s'],half['time_s'],xyhalf[:,i]) for i in range(2)])
    diff=float(np.max(np.linalg.norm(xywhole-interp,axis=1)))
    runtime_diff=None
    if args.runtime_trace:
        raw=np.load(args.runtime_trace)
        interpolation=np.column_stack([np.interp(raw['time_s'],half['time_s'],xyhalf[:,i]) for i in range(2)])
        runtime_diff=float(np.max(np.linalg.norm(raw['position_mm']-interpolation,axis=1)))
    tail=half['time_s']>half['time_s'][-1]-10
    span=float(np.linalg.norm(np.ptp(xyhalf[tail],axis=0)))
    table='\n'.join(f"| {q['turns']} | {q['dt']} | {q['final_displacement_rad']:.7f} | {q['tail_position_box_mm']:.5f} | {q['max_slow_rebound_rad']:.5f} |" for q in records)
    constants='\n'.join(f'{n} = {v:.17g}' for n,v in zip(names,r['settings']))
    states=',\n  '.join(f'{v:.17g}' for v in r['initial_state'])
    count=sum(v['total'] for v in counts.values())
    report=f'''# 小倾角轨迹：部分复现，尚无完整成功初态

本轮限制倾角 ≤ 1°，搜索记录 {count} 条；磁场几何、磁铁数量/尺寸/高度、球尺寸、摩擦与机械参数不变。释放时中心速度与自旋为零，并先固定球预磁化 1–3 圈。吹气只允许发生在捕获后。搜索没有给出满足全部初始“反向走 1–2 rad、无明显回摆、局部近停”的初态；这里给出可以直接复跑的部分候选，不把它叫作完整成功解。

当前候选倾角 **{case['tilt_degrees']:.6f}°**，140 mm 直径两端高差约 **{140*np.sin(r['settings'][4]):.4f} mm**。这个高差是倾角换算，并不代表模型已加入随位置变化的磁场间隙；倾斜力仍使用已有实现。

## 参数与初态

100 rpm；中心初始半径 {r.get('release_radius_mm',70.):.6f} mm；预磁化 2 圈；`dt=0.0005`（无量纲下盘相位步长，约47.75微秒）。JSON保存了编译磁参数、机械参数和完整初态。

```text
{constants}
```

完整状态顺序 `[下盘相位,x,y,vx,vy,四元数w,x,y,z,自旋x,y,z,qx,qy,qz]`：

```python
state = np.array([
  {states}
], dtype=np.float64)
sliding = {r['initial_sliding']}
sliding_dir = np.array({r['initial_sliding_direction']!r})
```

初始接触分支也必须携带。sliding/sticking是滑动/纯滚动，不是球心移动/静止。q已含磁化历史，复跑时不要归零，也不要再重复磁化。

## 实际轨迹

初始正向超前 {r['initial_forward_rad']:.8f} rad；初始角加速度 {r['initial_angular_acceleration_rad_s2']:.4f} rad/s²。最后停留区域均值 **{r['final_displacement_rad']:.5f} rad**；最后10秒原始二维位置包围盒对角线 **{span:.4f} mm**。

但最远到 **{r['minimum_displacement_rad']:.5f} rad**，最大宏观回摆 **{r['max_slow_rebound_rad']:.5f} rad**。因此仍未满足“到停止点后无明显返回”。局部近停也含磁场周期驱动的小幅快速抖动；最后3秒平均瞬时中心速率 {r['tail_mean_speed_mm_s']:.3f} mm/s，不能把平均角位移不变当成瞬时速度为零。

宏观回摆用0.15秒移动平均轨迹相对历史最小值计算；原始角度和二维位置同时保存。原始轨迹图明确保留回摆，不只画末段。

半径范围 {r['radial_range_mm'][0]:.5f}–{r['radial_range_mm'][1]:.5f} mm。
{escape_note}

## 直接复跑

在仓库根目录使用现有 daily 环境：

```powershell
& C:/Users/38359/miniconda3/envs/daily/python.exe src2d/replay_prepared_release.py --case src2d/results/small-tilt-release-fit/release_case.json --animate
```

捕获后吹气示例（强度为探索值）：

```powershell
& C:/Users/38359/miniconda3/envs/daily/python.exe src2d/replay_prepared_release.py --case src2d/results/small-tilt-release-fit/release_case.json --seconds 110 --kick-at 60 --kick-mm-s 20 --animate
```

吹气通过给中心叠加反向切向速度实现，保留当时的姿态、自旋、q，不重新初始化、不强制纯滚动。输出完整NPZ、CSV、图片和摘要。复制候选六个常量到自己构建后需要重新编译；只有改Python初态不能改变编译进二进制的磁参数。这里附带独立的匹配二进制，原生产构建未覆盖。

## 核对

| 预磁化圈数 | 无量纲步长 | 最后角位移均值/rad | 最后3秒包围盒/mm | 最大回摆/rad |
| --- | --- | --- | --- | --- |
{table}

步长减半，按相同物理时间插值比较，最大球心位置差 {diff:.6f} mm。常量构建与运行时扫描构建位置差：{runtime_diff} mm。局部位移的有限步长和接触分支敏感性以这些实际差值为准。

`C_JA=0.38` 固定搜索做过1200次；该组短时间评分最好的0.117°候选延长到60秒后仍明显往返，最后3秒包围盒约53.85mm。原始完整磁参数组也在上一轮搜索过，没有完整成功结果。本候选不能因此被当作真实材料磁参数标定。

结果仅说明这轮有限搜索没有找到全部条件同时满足的解，不能证明现有模型绝无该轨迹。4°候选保留作旧参考，已退出小倾角候选集。
'''
    (DEST/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(destination=str(DEST),tilt_degrees=case['tilt_degrees'],tail10_box_mm=span,
                         half_step_difference_mm=diff,constant_runtime_difference_mm=runtime_diff),indent=2))


if __name__=='__main__':main()
