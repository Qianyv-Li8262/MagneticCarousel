"""Export a clearly labelled partial candidate and verification evidence."""
from pathlib import Path
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'prepared-release-fit'
DEST=ROOT/'results'/'prepared-release-fit'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    records=json.loads((RUN/'fixed_verify.json').read_text(encoding='utf-8'))
    record=next(r for r in records if r['turns']==2 and r['dt']==.0005)
    ext=Path('scratch/legacy-release-source/build/release/extensions/fp64')
    manifest=json.loads((ROOT/ext/'build_config.json').read_text(encoding='utf-8'))
    # Import only scaling; it does not load another extension.

    names=['Mu_r','Alpha_JA','C_JA','K','kTiltAngleRad','kTiltAxisAzimuthRad']
    case=dict(schema_version=2,status='partial_candidate_not_full_experimental_fit',
        experiment_sequence='zero-speed release; 5–10 s of near arrest; then a reverse air impulse; sustained reverse rolling',
        matches=['initial reverse motion','eventual local capture near -1.03 rad'],
        mismatches=['large rebound before capture','late capture onset','large fitted tilt',
                    'after-capture impulses tested did not establish sustained reverse pure rolling'],
        extension_directory=ext.as_posix(),binary_sha256=manifest['binary_sha256'],
        magnetic_and_tilt_parameters=dict(zip(names,record['settings'])),
        tilt_degrees=record['settings'][4]*180/np.pi,
        initial_state_order=['disk_angle','x','y','vx','vy','quaternion_w','quaternion_x',
                             'quaternion_y','quaternion_z','omega_x','omega_y','omega_z','q_x','q_y','q_z'],
        initial_state=record['initial_state'],initial_sliding=record['initial_sliding'],
        initial_sliding_direction=record['initial_sliding_direction'],
        solver_parameters=dict(sr=5/6,somega=1),

        disk_rpm=100,disk_angular_speed_rad_s=100*2*np.pi/60,
        dt=.0005,sample_stride=20,pre_magnetization_disk_revolutions=2,
        pre_magnetization='fixed center and attitude; evolve JA q only; zero center speed/spin until release',
        geometry_unchanged=manifest['magnet_geometry'],field_parameters_unchanged=manifest['field_parameters'],
        mechanical_constants_unchanged=dict(Beta=.056999326368103,Eta=.4,Mu=.02),
        metric_record=record)
    (DEST/'release_case.json').write_text(json.dumps(case,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(record['trace'],DEST/'trajectory.npz')
    shutil.copy2(RUN/'candidate_config.hpp',DEST/'candidate_config.hpp')
    shutil.copy2(RUN/'fixed_verify.json',DEST/'verification.json')
    stage_counts={}
    for stage in ['coarse','refine','optimize','wide_optimize','impulse_zero_free','impulse_zero_old','old_large_tilt','c38_partial']:
        path=RUN/(stage+'.json')
        rows=json.loads(path.read_text(encoding='utf-8'))
        stage_counts[stage]=dict(total=len(rows),ok=sum(r['status']=='ok' for r in rows),
                               passes=sum(r.get('passes',False) for r in rows))
    (DEST/'search_counts.json').write_text(json.dumps(stage_counts,indent=2),encoding='utf-8')
    total=sum(r['total'] for r in stage_counts.values())
    half=np.load(record['trace'])
    whole=np.load(next(r['trace'] for r in records if r['turns']==2 and r['dt']==.001))
    interp=np.column_stack([np.interp(whole['time_s'],half['time_s'],half['position_mm'][:,i]) for i in range(2)])
    difference=float(np.max(np.linalg.norm(whole['position_mm']-interp,axis=1)))
    runtime=np.load(RUN/'capture_2.npz');n=len(runtime['time_s'])
    frozen_difference=float(np.max(np.linalg.norm(half['position_mm'][:n]-runtime['position_mm'],axis=1)))
    xy=half['position_mm'];t=half['time_s'];ang=half['angle_rad']
    span=float(np.linalg.norm(np.ptp(xy[(t>=30)&(t<=40)],axis=0)))
    tilt_height=140*np.sin(record['settings'][4])
    table='\n'.join(f"| {r['turns']} | {r['dt']} | {r['final_displacement_rad']:.8f} | {r['tail_position_box_mm']:.6f} | {r['max_slow_rebound_rad']:.6f} |" for r in records)
    parameters='\n'.join(f'{name} = {value:.17g}' for name,value in zip(names,record['settings']))
    initial=',\n  '.join(f'{v:.17g}' for v in record['initial_state'])
    report=f'''# 零速释放与近停：本轮没有完整成功初态

本轮 {total} 条零速释放搜索记录（包括失败的积分）未找到同时符合“几乎无正向行进、反向走 1–2 rad 后无明显回摆并近停”的解。附带的是部分候选，不能把它称为实验复现。有限搜索失败也不能证明模型绝无此解。

实验条件按最终澄清记录：球以接近零速释放，运动后被捕获并极慢运动 5–10 s，再向反向吹气；吹气后的反向滚动稳态维持 60–70 圈。不是释放时吹气，也不是捕获维持 60–70 圈。倾斜一直开启。

## 可复跑的部分候选

原生产源码和构建未改动。候选使用单独 FP64 常量构建，几何、磁场 LUT、摩擦系数和机械参数不变。新增的预磁化助手只在球被固定时推进原有 JA 方程；释放后仍使用原动力学，没有加入滚动阻力、涡电流阻尼或人为速度归零。

100 rpm；初始位置 (70, 0) mm；初始中心速度和自旋均为零；固定位置/姿态预磁化 2 圈。磁化历史 q 和下盘相位一起保存，不能用 q=0 替代。

```text
{parameters}
```

倾角 {case['tilt_degrees']:.6f}°；在 140 mm 直径两端对应约 {tilt_height:.3f} mm 高差。这个值远大于之前讨论的 0.5–1 mm 高差，不能当作实测倾斜拟合。当前模型的倾斜作用仍沿用已有实现；没有把倾斜换成新的空间变间隙磁场。

完整 15 维初态（无量纲）：

```python
state = np.array([
  {initial}
], dtype=np.float64)
```

顺序为 `[下盘相位, x, y, vx, vy, 四元数w,x,y,z, 自旋x,y,z, qx,qy,qz]`。步长 `0.0005` 是无量纲下盘相位步长，对应约 47.75 微秒。

接触初始分支：`sliding={record['initial_sliding']}`，方向 `{record['initial_sliding_direction']}`。这里的 sliding/sticking 判断接触点滑动/纯滚动；不等于球心运动/静止。初始中心速度为零时必须同时携带该分支，默认初始化可能选错分支。

## 轨迹实际表现与不符之处

- 初始角加速度为 {record['initial_angular_acceleration_rad_s2']:.4f} rad/s²，先反向，未测得正向超前。
- 最远到 {record['minimum_displacement_rad']:.5f} rad，最后停留区均值约 {record['final_displacement_rad']:.5f} rad。
- 最大宏观回摆约 {record['max_slow_rebound_rad']:.5f} rad，明显不符合“走到停止点后不返回”。该值按 0.15 s 移动平均轨迹与历史最小值计算，仅用于区分大回摆与高频小抖动；图中同时显示原始轨迹。
- 30–40 s 原始二维位置包围盒对角线 {span:.4f} mm；最后 3 s 为 {record['tail_position_box_mm']:.4f} mm。这是局部小幅抖动，不是数学上的零速度。
- 最后 3 s 平均瞬时中心速率仍为 {record['tail_mean_speed_mm_s']:.3f} mm/s，近停判断不能只依据平均角度。
- 半径范围 {record['radial_range_mm'][0]:.4f}–{record['radial_range_mm'][1]:.4f} mm，略超出此前 65–75 mm 的实验描述。
- 对该捕获候选保留姿态、自旋、q，只给中心叠加反向切向速度 2、5、10、20、40、80、160、320 mm/s，各推进 50 s，均未进入持续反向纯滚动；多数重新捕获。详细记录在 `src2d/runs/prepared-release-fit/capture_escape.json`。

## 复跑

工作目录在仓库根目录，使用已有 daily Python 环境：

```powershell
& C:/Users/38359/miniconda3/envs/daily/python.exe src2d/replay_prepared_release.py --animate
```

模拟捕获之后吹气（示例强度是探索值，不是实测值）：

```powershell
& C:/Users/38359/miniconda3/envs/daily/python.exe src2d/replay_prepared_release.py --seconds 90 --kick-at 35 --kick-mm-s 20 --animate
```

工具会输出 `trajectory.npz`（含全部状态）、`trajectory.csv`、`trajectory.png` 和摘要。后续吹气只改中心速度；q、自旋和姿态连续，不会重新磁化或强制切换到纯滚动。想把参数塞进自己的普通构建，可复制 `candidate_config.hpp` 中六个候选常量并重新编译；初态与初始接触分支见 JSON。

## 数值核对

| 预磁化圈数 | 无量纲步长 | 最后角位移均值 / rad | 最后3秒位置包围盒 / mm | 最大回摆 / rad |
| --- | --- | --- | --- | --- |
{table}

普通常量构建与运行时扫描构建在相同初态、相同采样时间下，35 s 内最大球心位置差 {frozen_difference:.6f} mm。步长减半后按相同物理时刻插值比较，60 s 内最大位置差 {difference:.6f} mm。这个核对支持轨迹数值稳定，不验证磁参数是真实材料参数，也不消除模型中缺失效应的可能。

原始磁参数 `Mu_r=2292.6667, Alpha_JA=0.0032, C_JA=0.38, K=0.0010909091` 已单独搜索倾斜量、方向和释放相位；没有找到完整成功解。其最佳 20 s 评分候选在延长到 60 s 后仍有明显运动，不能用短时间端点宣称捕获。记录保存在 `impulse_zero_old.json` 与 `old_best_two_stage.json`；文件名中的 impulse 是脚本沿用名称，这两组释放时速度仍为零。
'''
    (DEST/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(destination=str(DEST),searches=total,half_step_position_difference_mm=difference,
                         frozen_position_difference_mm=frozen_difference),indent=2))


if __name__=='__main__':main()
