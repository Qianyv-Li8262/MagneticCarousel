# 无量纲约定、参数换算与模型说明

本文对应当前 `src2d` 实现。调参前请先区分：物理参考量、运行时倍率、编译期材料/几何参数，以及纯数值设置。本文中的默认值用于说明当前配置，不表示它们都是独立实测值。

## 1. 参考尺度与坐标

物理参考量定义在 [`hysteresis_physics.hpp`](../kernels/headers/hysteresis_physics.hpp)。下标 `SI` 表示物理量，星号表示二维求解器中的无量纲量。

| 符号 | 源码 | 当前参考值 |
| --- | --- | --- |
| 钢球参考半径 $a_0$ | `Rref` | $0.006\ \mathrm m$ |
| 磁铁中心所在圆的参考半径 $L$ | `L` | $0.070\ \mathrm m$ |
| 参考磁铁数量 $N_0$ | `reference_magnet_count` | $8$ |
| 磁铁盘参考角速度 $\Omega_0$ | `omega_ref` | $100\ \mathrm{rpm}=100(2\pi/60)\ \mathrm{rad/s}$ |
| 磁化强度参考量 $M_0$ | `M_ref` | $1.1\times10^6\ \mathrm{A/m}$ |
| 钢球密度 $\rho_b$ | `rho_b` | $7850\ \mathrm{kg/m^3}$ |
| 空气密度 $\rho_a$ | `rho_air` | $1.225\ \mathrm{kg/m^3}$ |
| 空气阻力系数 $C_d$ | `Cd` | $0.47$ |
| 重力加速度 $g$ | `g` | $9.81\ \mathrm{m/s^2}$ |
| 真空磁导率 $\mu_0$ | `mu0` | 代码采用 $4\pi\times10^{-7}\ \mathrm{H/m}$ |
| 电导率 $\sigma$ | `sigma` | 当前条件拟合值约 $9.33127\times10^6\ \mathrm{S/m}$ |

当前实际磁铁数量为 $N=\texttt{magnet\_number}=8$，记 $p=N/2=4$。磁铁等间隔排列，磁矩沿盘面法向、相邻极性交替。

在当前 $N=N_0=8$ 配置下，取

$$
\ell=\frac{L}{p}=17.5\ \mathrm{mm},\qquad
k=\ell^{-1}=\frac{p}{L},\qquad
\lambda=2\pi\ell.
$$

$\lambda$ 对应沿磁铁圆周相邻同极性位置的弧长周期，相邻磁铁间隔为 $\lambda/2$。高度和几何长度均乘 $k$ 归一化，不能直接除以 $\lambda$。

使用固定于实验室的右手坐标系：$x,y$ 位于承载平面，$+z$ 从磁铁盘指向钢球所在一侧。$\phi=\operatorname{atan2}(y,x)$；沿 $+z$ 观察向原点看，正向为逆时针。磁铁盘角度为 $\theta_d$，其无量纲角速度为 $+1$。空间坐标系并不随磁铁盘一起转动。

球体姿态四元数 $Q=(w,x,y,z)$ 将随体系向量转到实验室系：$\mathbf u_w=R(Q)\mathbf u_b$。`state.q` 是磁滞记忆向量，不是这个四元数.

### 改变磁铁数量时的兼容约定

几何代码始终令磁铁圆半径为 $p=N/2$，`physical_units()` 用 $\ell_N=L/p$ 输出长度单位。但当前 `make_solver_params()` 保留历史约定：用固定 $k_0=N_0/(2L)$ 构造 `alpha` 和 `delta`，将 `gamma_g`、`gamma`、`gamma_m` 再乘 $(N/N_0)^2$。

因此，下文直接对应 SI 的统一换算以 **$N=N_0=8$** 为前提。$N\ne N_0$ 时，不能把几何换算、球半径参数和动力学系数当作已经统一重标度的物理装置；需先核对这套兼容约定，再比较不同磁铁数量的物理结果。修改 `magnet_number` 不只是修改场的谐波数量。

## 2. 运行时倍率与 SI 换算

运行时的两个倍率为

$$
a=a_0s_r,\qquad \Omega=\Omega_0s_\omega,
\qquad m_b=\frac{4\pi}{3}\rho_ba^3,\qquad I=\eta m_ba^2.
$$

- `sr` / `--sr` 是**半径倍率**；钢球直径为 $12s_r\ \mathrm{mm}$。
- `somega` / `--somega` 是磁铁盘转速倍率；盘转速为 $100s_\omega\ \mathrm{rpm}$。
- 两个倍率均要求有限且大于零；负切向初速度和负自旋由各自初态参数设置。
- `Eta` 即 $\eta$，均匀实心球为 $2/5$。

二维求解器采用

$$
\mathbf x^*=\frac{\mathbf x_{SI}}\ell,
\quad t^*=\Omega t_{SI},
\quad \mathbf v^*=\frac{\mathbf v_{SI}}{\ell\Omega},
\quad \boldsymbol\omega^*=\frac{\boldsymbol\omega_{SI}}\Omega.
$$

| 输出或输入 | 转回物理量 |
| --- | --- |
| 位置、轨道半径、无量纲高度 $r^*,h^*$ | 乘 $\ell$，得到 m |
| 时间、步长、观测窗 $t^*,\Delta t^*$ | 除以 $\Omega$，得到 s |
| 速度、接触滑移速度 $v^*,s^*$ | 乘 $\ell\Omega$，得到 m/s |
| 自旋 $\omega_i^*$ | 乘 $\Omega$，得到 rad/s；再乘 $60/(2\pi)$ 得到 rpm |
| 加速度 $a^*$ | 乘 $\ell\Omega^2$，得到 m/s² |
| 角度、盘相位、倾角 | 数值为 rad，无额外长度或时间倍率 |
| 外场 $\mathbf h$ | $\mathbf H_{ext}=M_0\mathbf h$，单位 A/m |
| 外场梯度 $\nabla_*\mathbf h$ | 乘 $M_0/\ell$，单位 A/m² |
| 总磁化 $\mathbf M^*$ | 乘 $M_0$，单位 A/m；磁偶极矩还需乘球体积 $V$ |

真空中的外场满足 $\mathbf B_{ext}=\mu_0M_0\mathbf h$；这不是钢球内部的总磁感应强度。$M_0$ 是归一化参考量，不等于拟合中的物理饱和磁化强度 $M_s$。

例如 `sr=5/3, somega=2` 对应直径 20 mm、磁铁盘 200 rpm：

$$
\Omega=20.9439510\ \mathrm{rad/s},\quad
\ell\Omega=0.3665191\ \mathrm{m/s},\quad
\alpha=0.57142857.
$$

此时 $t^*=1$ 为约 $0.04775$ s，$\Delta t^*=0.0005$ 为约 $23.87\ \mu$s；物理自旋 $40\ \mathrm{rad/s}$ 对应 $\omega_z^*\approx1.90986$。`somega` 已在 $\Omega$ 中，转换后不能再乘一次。

## 3. 一维兼容量、初态和扫描输出

部分入口保留旧一维模型的归一化。记

$$
t_1=p\,t^*,\qquad v_1=\frac{v_{tan}^*}{p},\qquad
\omega_1=\frac{\omega_r^*}{p}.
$$

其中 $v_{tan}^*=\mathbf v^*\cdot\mathbf e_\phi$，$\omega_r^*=\boldsymbol\omega^*\cdot\mathbf e_r$。这里的 $\omega_1$ 是**径向轴转动**的归一化量，与法向自旋 $\omega_z^*$ 不同。物理切向速度为 $v_{tan,SI}=L\Omega v_1$。

### 动画与初态扫描的初值

`animate_one_trajectory.py` 和初态扫描的 `v0`、`s0` / $S_0$ 使用一维兼容单位，而 `-oz` 使用二维单位。动画初始化为

$$
\mathbf r_0^*=p\mathbf e_r(\phi_0),\qquad
\mathbf v_0^*=p v_0\mathbf e_\phi(\phi_0),
$$

$$
\boldsymbol\omega_{\parallel,0}^*
=\frac{p(S_0-v_0)}\alpha\mathbf e_r(\phi_0),
\qquad \omega_{z,0}^*=\texttt{oz}.
$$

接触点相对静止承载面的滑移速度定义为

$$
\mathbf s^*=\mathbf v^*+\alpha\,\mathbf e_z\times\boldsymbol\omega^*
=(v_x^*-\alpha\omega_y^*,\ v_y^*+\alpha\omega_x^*).
$$

因此初始切向滑移为 $pS_0$；$S_0=0$ 表示初始纯滚条件，不表示球心速度为零。纯滚时

$$
\boldsymbol\omega_\parallel^*=\frac{\mathbf e_z\times\mathbf v^*}\alpha.
$$

`phi0` 为球心初始方位，`disk-phase` 为释放时磁铁盘角度，二者均为 rad。初始化姿态为单位四元数，磁滞和涡流记忆默认置零。`magnetize-turns` 按磁铁盘整圈数计算预磁化时长 $2\pi n$，在保持球不动的条件下演化记忆，再设置释放速度；零记忆只是模型初态约定，不保证对应真实钢球已消磁。

### 时间、动画与外力

`--dt`、`--pre`、扫描的 `--total-time` / `--observation-window` / `--sample-time`、吹气起止时间，以及 `--t,oz` 中的时间均为 $t^*$，不是秒。定时重置的 `oz` 仍为 $\omega_z^*$。

`--frames`、`--steps-per-frame` 和 `--fps` 控制显示采样和刷新；`fps` 不能替代物理时间换算。动画中的 `1D t` 为 $pt^*$，`release time` 才是释放后的秒数；`x_eq=mean(r)·Δφ` 是显示采样平均半径乘展开角位移，不是积分得到的真实路程。

`--blow-force` 用 **mN** 输入。程序按

$$
a_{push}^*=\frac{F_{mN}\,10^{-3}}{m_b\ell\Omega^2}
$$

换成球心反切向加速度，不施加直接力矩。这里不需要再手动无量纲化力。

### 扫描指标

扫描文件使用 `state_names`、`metric_names` 记录字段名，读取时应按这些名称索引。

| 字段 | 约定 |
| --- | --- |
| `mean_v`、`final_v`、`min_v`、`max_v`、`std_v` | 切向 $v_{tan}^*/p$ 及其统计量，不是二维速度模长 |
| `mean_omega`、`final_omega` | 径向 $\omega_r^*/p$，不是 $\omega_z^*$ |
| `mean_phase_rate`、`final_phase_rate`、`std_phase_rate` | $1-\dot\phi-\omega_r^*/p$ 的统计量，$\dot\phi=(xv_y-yv_x)/r^2$ 按 $t^*$ 求导 |
| `mean_abs_slip` | $\langle|\mathbf s^*|/p\rangle$ |
| `mean_radius` | $\langle r^*\rangle$，仍为二维长度单位 |
| `mean_abs_q`、`final_abs_q` | 磁滞记忆向量的模，不是总磁化或四元数范数 |
| `delta_mean_v`、`delta_mean_phase_rate` | 观测窗后半段均值减前半段均值，单位与对应字段相同 |
| `sliding_fraction` | 采样点处于滑动模式的比例 |
| `sample_count`、`global_minimum_step` | 计数/步号，不是时间；步号乘实际 `dt` 才得到 $t^*$ |
| `velocity_recovered` | 速率达到全局最小值后是否再次增大的标记 |

相位率是沿用一维定义的诊断量，不是球的法向自旋或直接测得的磁滞相位。图中的锁定、速度接近 1 和半径容差均为后处理分类阈值，不是材料参数；看图时也应区分裁剪后的色标与原始数值。

## 4. 几何参数与磁场幅值

几何配置见 [`hysteresis_config_field.hpp`](../kernels/headers/hysteresis_config_field.hpp)，有限圆柱尺寸通过 CMake 设置。

| 参数 | 含义与当前默认值 |
| --- | --- |
| `IYPT16_FIELD_HEIGHT` | 球心到磁铁中心平面的无量纲高度 $h^*$ |
| `magnet_number` | 交替极性的磁铁数 $N$，必须为正偶数 |
| `IYPT16_MAGNET_RADIUS` | 单颗磁铁半径 $b^*$ |
| `IYPT16_MAGNET_THICKNESS` | 磁铁完整厚度 $d^*$ |
| `IYPT16_LUT_LOWER_BOUND_RATIO` / `UPPER` | LUT 径向范围相对磁铁圆半径 $p$ 的倍率，默认 $[0.5p,1.5p]$ |
| `kFieldFourierModeCount` | 保留的角向谐波数，默认 5 |
| `kSplineNodes` | 每个谐波的径向节点数，默认 204 |

当前默认 LUT 覆盖半径 $[35,105]$ mm。范围外状态应作为求解失败处理，不应把边界区间选取当成可信外推。

更换球径时，`sr` 不会自动更新高度或 LUT。应先从装置几何计算

$$
h_{SI}=\text{两承载面高度差}+a-\text{磁铁中心高出下表面的距离},
\qquad h^*=h_{SI}/\ell.
$$

当前例子为 $32.5+10-6=36.5$ mm。高度不是盘面间距，也不是球面到磁铁端面的最小间隙；圆柱模型还要求 $h^*>d^*/2$。

每颗磁铁的磁矩幅值 $m_{mag}$ 通过 `Beta` 进入：

$$
\beta=\frac{m_{mag}k^3}{4\pi M_0},\qquad
m_{mag}=4\pi M_0\ell^3\beta.
$$

LUT 存的是未乘 $\beta$ 的几何场，运行时统一乘一次。改变圆柱尺寸不会自动改变 `Beta`：保持磁矩不变和保持磁铁材料磁化强度不变，是两种不同的物理调整。

`exact` 是点磁铁离散场求和；`lut` 使用 Fourier–spline 重建。当前有限圆柱磁铁必须使用 LUT，CMake 会将该配置下的 `exact` 改为 `lut`。这些设置不能被理解为同一磁场模型的任意精度切换。

## 5. 动力学、材料参数和量纲重名

### 求解器参数

在 $N=N_0$ 下，`SolverParams(sr, somega)` 自动构造

$$
\alpha=ka,\qquad
\delta=\frac{3\rho_a C_d}{8\rho_bak},\qquad
\gamma_g=\frac{kg}{\Omega^2},
$$

$$
\gamma=\frac{3\mu_0M_0^2k^2}{\rho_b\Omega^2},\qquad
\gamma_m=\frac\gamma{\eta\alpha^2},\qquad
s_{eddy}=s_\omega s_r^2.
$$

`alpha` 是无量纲球半径；`Alpha_JA` 是磁滞平均场反馈参数。`Mu` 是接触摩擦系数；`Mu_r` 是磁性参数。`gamma`、`gamma_m`、`gamma_g` 应由参考量和倍率生成，不宜在多处分别手动修改。

若改变 $N$，当前实现对上述后三个 $\gamma$ 系数额外乘 $(N/N_0)^2$；`alpha`、`delta` 中则仍使用固定的 $k_0$，见第 1 节。

### 力、力矩和功率

源码 `Force` 表示已除以质量的无量纲力，即加速度；`magnetic_angular_acceleration` 及 RHS 中的 `M` 表示角加速度，不是 SI 力矩：

$$
\mathbf F^*=\frac{\mathbf F_{SI}}{m_b\ell\Omega^2},\qquad
\mathbf A_\omega^*=\frac{\boldsymbol\tau_{SI}}{I\Omega^2}.
$$

法向载荷为 $N^*=\gamma_g-F_z^*$；$N^*<0$ 表示模型无法维持接触。滑动摩擦为 $-\texttt{Mu}\,N^*\widehat{\mathbf s}$；纯滚模式通过无滑动约束和摩擦可行性判据处理。

能量和功率的参考尺度分别为 $m_b\ell^2\Omega^2$、$m_b\ell^2\Omega^3$。`magnetic_power()` 输出的三项为

$$
P_{trans}^*=\mathbf F_{mag}^*\cdot\mathbf v^*,\qquad
P_{rot}^*=\eta\alpha^2\mathbf A_{\omega,mag}^*\cdot\boldsymbol\omega^*,\qquad
P_{mag}^*=P_{trans}^*+P_{rot}^*.
$$

它们只表示磁力与磁力矩的机械功率，不包含全部耗散或磁铁盘驱动装置的完整能量账本。

### 磁滞参数

设弱场无磁滞斜率 $c_a=M_s/(3a_{JA})$，其中 $a_{JA}$ 是 SI 磁场参数，和球半径 $a$ 不同。代码参数换算为

$$
\mu_r=1+\frac{c_a}{1-\alpha_{JA}c_a},\qquad
\Xi=\frac{\alpha_{JA}(\mu_r-1)}{1+\alpha_{JA}(\mu_r-1)},\qquad
c_a=\frac{\mu_r-1}{1+\alpha_{JA}(\mu_r-1)},
$$

$$
C=\texttt{C\_JA},\qquad \alpha_{JA}=\texttt{Alpha\_JA},\qquad
K=\frac{k_{JA,SI}}{M_0}.
$$

`Xi_JA` 是派生量，不是第五个独立拟合参数。当前 $K\approx0.00162768$ 对应 $k_{JA,SI}\approx1790.45\ \mathrm{A/m}$。这里的 $\mu_r$ 描述模型的无磁滞零场斜率，不等于实际磁滞工作点处处相同的 $B/H$ 或微分磁导率。修改它也会修改涡流系数。

常规调参应保持 $K>0$、$0\le C\le1$、正的材料斜率，并检查自洽支路分母是否接近零。当前实现含 `Xi_JA / Alpha_JA` 及其倒数，不能直接令 `Alpha_JA=0` 或 `Mu_r=1` 来测试极限；这些情形需专门实现对应的解析极限，不能依赖浮点除法自行处理。

### 滚阻与倾斜

滚阻只作用于水平角速度。记 $w=|\boldsymbol\omega_\parallel^*|$，代码采用

$$
\mu_{rr}(w)=\mu_{high}+(\mu_{low}-\mu_{high})e^{-(w/w_c)^2},
$$

$$
\Delta\dot{\boldsymbol\omega}_\parallel^*
=-\left[\frac{\mu_{rr}(w)N^*}{\eta\alpha\sqrt{w^2+\varepsilon^2}}+\Lambda\right]
\boldsymbol\omega_\parallel^*.
$$

对应参数为 `kRollingMuLow/High`、`kRollingTransitionOmega`、`kRollingOmegaEpsilon`、`kRollingLambda`。$w_c,\varepsilon$ 是以实际磁铁盘角速度归一化的**水平自转角速度**，不是绕行角速度。物理转动黏性系数若写作 $\tau=-c\omega$，则 $\Lambda=c/(I\Omega)$。

纯滚时，转折速度尺度为 $v_c=a\Omega w_c$。如果改变 `sr` 或 `somega` 后仍希望保持同一个物理 $v_c$，应重新换算 $w_c$；固定代码数值不是固定物理速度。当前没有由这一滚阻项直接产生的法向自旋阻尼。

倾斜开关为 `kEnableInclinedDisk`，角度 $\vartheta$ 和轴方位 $\chi$ 分别对应 `kTiltAngleRad`、`kTiltAxisAzimuthRad`。小角度修正为

$$
\Delta\mathbf F_\parallel^*=\vartheta(\gamma_g-F_z^*)(\sin\chi,-\cos\chi).
$$

例如 $\chi=\pi/2$ 时下坡沿 $+x$。这里的方位是倾斜轴方位，不是直接填写下坡方位。该近似不改变法向载荷和磁力矩。

### 状态布局与数值设置

Python 状态数组前 15 项依次为盘角 `[0]`、位置 `[1:3]`、速度 `[3:5]`、姿态四元数 `[5:9]`、实验室系角速度 `[9:12]`、随体系磁滞记忆 `[12:15]`。启用涡流后追加所有 $y_x$、所有 $y_y$、所有 $y_z$，不是按每个模态交错排列。

`kEddyCurrentModeCount` 表示四通道 AVX 组数；实际标量模态数 $J=4\,\texttt{kEddyCurrentModeCount}$。当前为 3 组、12 模态，完整状态长度 $15+3J=51$。部分拟合权重可以为零；关闭涡流时状态长度为 15。以模块实际导出的 `state_size`、`eddy_mode_count` 和 `real_dtype` 为准。

FP64 / Kahan FP32、步长、递归细分深度、四元数归一化方式、AVX 指数近似和 LUT 缓存模式属于数值设置。它们不能作为材料参数拟合。`IYPT16_LUT_CACHE_MODE` 的 0/1/2 分别为不缓存、缓存节点、缓存样条系数，不改变目标物理模型；改变精度或近似方式后仍需检查收敛。

## 6. 调参时修改哪里

| 调整内容 | 入口 | 后续动作 |
| --- | --- | --- |
| 球径、盘转速、初速、滑移、自旋、盘相位 | 动画 CLI / 扫描脚本；单位见第 2–3 节 | 重跑；换球时另核对高度 |
| $\mu_r,\alpha_{JA},C,K,\beta$、摩擦、滚阻、倾斜 | `hysteresis_config.hpp` | 重新构建二进制，材料头变更目前也触发涡流 codegen |
| 电导率、密度、参考球径和转速、$M_0$ 等 | `hysteresis_physics.hpp` | 重新构建；修改参考尺度时同步审查全部换算 |
| 高度、磁铁数量、LUT 范围和分辨率 | `hysteresis_config_field.hpp` | 重建生成器、LUT 和相关二进制 |
| 磁铁模型、半径、厚度、CUDA 架构 | CMake 配置选项 | 重新配置，再构建 |
| 涡流拟合频带、模式配置、误差门限 | `eddy_codegen_config.hpp`、模态组数 | 重生成系数并检查报告，再构建 |

`--dry-run` 可检查动画实际使用的参数和构建清单。源码参数变化后应重编，不能通过修改 manifest 绕过一致性检查。试算时请记录代码版本、球径、高度、盘速和所用精度；不要混用不同装置或旧高度下得到的“最佳参数”。

## 7. 模型概述

### 7.1 线性化矢量 Jiles–Atherton 磁滞

无磁滞曲线由 $M_s\mathcal L(H_e/a_{JA})$ 作弱场展开，$\mathcal L(x)\approx x/3$。因此线性化的是无磁滞曲线；历史记忆、方向投影和支路切换仍然存在，并非每步直接令磁化与外场成正比。

下面用当前代码的闭合式描述具体矢量实现，所有磁量按 $M_0$ 归一化，点号表示对 $t^*$ 求导。令

$$
D=\frac{\mu_r-1}{\mu_r+2},\qquad
E=\frac{1-C}{(1-\Xi)(\mu_r+2)},\qquad
c_a=\frac{\Xi}{\alpha_{JA}},\qquad
\mathbf h_0=\frac3{\mu_r+2}\mathbf h_b.
$$

随体系中的记忆变量 $\mathbf q$ 满足投影式演化

$$
\dot{\mathbf q}
=c_a\dot{\mathbf h}_e
-\frac{|\mathbf q|}{K}
 [\widehat{\mathbf q}\cdot\dot{\mathbf h}_e]_+\widehat{\mathbf q},
\qquad [x]_+=\max(x,0),
$$

有效场增量按代码中的关系自洽求解：

$$
(1-\Xi)\dot{\mathbf h}_e
=\dot{\mathbf h}_0+\big[E-\alpha_{JA}(1-C)\big]\dot{\mathbf q}.
$$

在 $\mathbf q=0$ 时取投影项为零，不计算 $\widehat{\mathbf q}$。这对应实现中先尝试非活动支路，再检验 $\mathbf q\cdot\dot{\mathbf h}_e$ 决定是否采用活动支路的过程。

用于载荷的无量纲磁滞磁化在实验室系重建为

$$
\mathbf M_{JA}^*=3\left(D\mathbf h-E R(Q)\mathbf q\right).
$$

$\mathbf q$ 的含义应连同这套闭合和重建式理解，不能直接作为总磁化输出。它是项目采用的线性化矢量 JA 实现，不表示所有矢量 JA 推广都具有相同形式；实际钢球的旋转磁化与材料参数仍需验证。

### 7.2 解析导电球响应与有限模态涡流反馈

涡流部分从均匀、线性导电导磁球的解析响应出发。以 $e^{i\omega t}$ 为频域约定，设 $a$ 为物理球半径，

$$
x=\sqrt{-i\omega\mu_0\mu_r\sigma a^2},\qquad
\mathcal A(\omega)=\frac32
\frac{(2\mu_r+1)j_1(x)-xj_0(x)}{(\mu_r-1)j_1(x)+xj_0(x)}.
$$

这里 $j_0,j_1$ 是球贝塞尔函数，$\mathbf m/V=\mathcal A\mathbf H_{ext}$。减去静态响应，避免与磁滞模型的静态磁化重复计入：

$$
\Delta\mathcal A(\omega)=\mathcal A(\omega)-3\frac{\mu_r-1}{\mu_r+2}
\approx-\sum_{j=1}^{J}a_j\frac{i\omega\tau_j}{1+i\omega\tau_j},
\qquad a_j\ge0.
$$

系数生成器保留最慢的两个物理扩散时间常数，为其余候选时间常数作对数布点，再用非负最小二乘逼近完整解析响应。不是简单截取物理解的前 12 项。当前拟合频带为参考半径下 0.1–300 Hz，并另检验低频斜率。

运行时使用随体系记忆模态

$$
\dot{\mathbf y}_j+\frac{\mathbf y_j}{\tau_j^*}
=-\dot{\mathbf h}_b,\qquad
\Delta\mathbf M_b^*=\sum_j a_j\mathbf y_j,
$$

$$
\tau_j^*=s_\omega s_r^2\tau_{j,ref}^*,\qquad
\tau_{j,ref}^*=\Omega_0\tau_{j,ref,SI}.
$$

若一个子步内冻结 $\dot{\mathbf h}_b$，则按解析指数形式推进：

$$
\mathbf y_j^{new}
=e^{-r_j}\mathbf y_j^{old}
-\Delta t^*\frac{1-e^{-r_j}}{r_j}\dot{\mathbf h}_b,
\qquad r_j=\Delta t^*/\tau_j^*.
$$

涡流状态单独推进，不由普通 RK 状态导数再次积分。随体场导数包含磁铁盘转动、球心平动和钢球自转：

$$
\dot{\mathbf h}_b
=R(Q)^T\left[\partial_{t^*}\mathbf h
+(\mathbf v^*\cdot\nabla_*)\mathbf h
-\boldsymbol\omega^*\times\mathbf h\right].
$$

总磁化为 $\mathbf M^*=\mathbf M_{JA}^*+R(Q)\Delta\mathbf M_b^*$，对运动的反馈为

$$
\mathbf F_{mag}^*=\frac\gamma3(\nabla_*\mathbf h)^T\mathbf M^*,\qquad
\mathbf A_{\omega,mag}^*=\frac{\gamma_m}3\mathbf M^*\times\mathbf h.
$$

该反馈是线性球响应对非线性磁滞模型的增量补充，并未联立求解球内完整磁滞与电磁扩散。当前直接复用固定 `Mu_r`，不能自动解释为真实工作点的增量磁导率。物理频率 $f$ 在球径倍率 $s_r$ 下应映射到参考频率 $fs_r^2$；频域拟合误差不是任意暂态或耦合轨迹的统一误差界。更多细节见 [EDDY_CODEGEN.md](../kernels/tools/EDDY_CODEGEN.md)。

### 7.3 Fourier–spline 磁场 LUT

对交替极性的轴对称磁铁阵列，令 $\psi=\phi-\theta_d$，未乘 `Beta` 的标量势展开为

$$
\Phi_0(\rho,z,\psi)
=\sum_{j=0}^{\infty} A_j(\rho,z)\cos(\nu_j\psi),\qquad
\nu_j=(2j+1)\frac N2.
$$

因此 $N=8$ 时保留的前五个谐波为 $4,12,20,28,36$，并非连续的 1–5 阶。求解器采用 $\mathbf h=-\beta\nabla_*\Phi_0$。记 $Z_j=\partial_z A_j$，则

$$
h_\rho=-\beta\sum_j A_j'\cos(\nu_j\psi),\quad
h_\phi=\beta\sum_j\frac{\nu_j A_j}{\rho}\sin(\nu_j\psi),\quad
h_z=-\beta\sum_j Z_j\cos(\nu_j\psi).
$$

撇号表示对无量纲径向坐标 $\rho$ 求导。场梯度由同一组势和样条导数重建；柱坐标向量/张量再用实验室方位 $\phi$ 转到直角坐标，不能用相对相位 $\psi$ 做这次基变换。场外区域利用无源、无旋条件恢复对称无迹梯度的其余分量。

有限圆柱生成器把径向系数写成谱积分。用 $u$ 表示无量纲谱变量，$b,d$ 表示单颗圆柱的无量纲半径和完整厚度，磁铁圆半径为 $p$：

$$
A_j(\rho,h)=2N\int_0^\infty
J_{\nu_j}(u\rho)J_{\nu_j}(up)
\frac{2J_1(ub)}{ub}
\frac{e^{-u(h-d/2)}(1-e^{-ud})}{d}\,du.
$$

这里 $J_\nu$ 是普通贝塞尔函数，与涡流解析式中的球贝塞尔函数 $j_\nu$ 不同。径向导数通过贝塞尔导数得到，$Z_j$ 通过对高度求导得到。生成器以高精度数值求积、尾部截断和加密面板对照检查系数；点磁铁生成器则采用相应的超几何函数形式。

在固定高度的径向网格上，对每个谐波分别构造 $A_j(\rho)$ 和 $Z_j(\rho)$ 的端点一阶导数约束三次样条。每个节点存储

$$
(A_j,\ M_{A,j},\ Z_j,\ M_{Z,j}),\qquad
M_{A,j}=\partial_\rho^2 A_j,\quad M_{Z,j}=\partial_\rho^2 Z_j.
$$

这里的 $M_A,M_Z$ 是样条二阶导数，不是磁化强度。对区间宽 $\Delta\rho$ 和局部坐标 $u=(\rho-\rho_i)/\Delta\rho$，任一分量 $f$ 的插值为

$$
S(\rho)=(1-u)f_i+uf_{i+1}
+\frac{\Delta\rho^2}{6}\left[\big((1-u)^3-(1-u)\big)M_i+(u^3-u)M_{i+1}\right].
$$

运行时由同一个多项式求值、求一阶及二阶导数，避免对离散场再作有限差分。两种输出精度共用高精度生成结果，按 FP32 / FP64 写入；二进制按“谐波 → 径向节点 → 四个量”排列。LUT 与球的磁化历史、转速和 `Beta` 无关，但依赖磁场几何、固定高度、截断谐波及径向网格。

误差来源需分别考虑：磁铁/钢球的物理近似、角向谐波截断、谱积分、径向样条插值和运行时浮点误差。生成器的高精度和面板验证不能自动保证所有位置上的总场误差，更不能替代轨迹收敛或实验检验。
