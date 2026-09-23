# 二维 kernel 物理公式独立审核

审核日期：2026-09-23。对象为当前工作区 `kernels/headers/2dmodel-hysteresis.hpp`、场系数生成器及参数定义，包含工作区未提交修改。仅审核物理量计算和解析公式，不评价积分器、查表越界、精度配置及程序接口。

**结论：在下述模型假设下，没有发现当前公式的符号、系数或坐标变换错误。** 磁滞部分的结论是对明确写出的本构模型进行独立代数推导后的自洽性确认；仓库未提供独立于实现的完整原始本构定义，因此不能将其等同于对所有矢量 Jiles–Atherton 模型的验证。

## 1. 记号与假设

- 空间坐标以 `k` 无量纲化，时间以底盘角速度 `Ω` 无量纲化；底盘角速度因此为 1。
- 球半径 `r_b`，`α=k r_b`；惯量 `I=η m_b r_b²`，当前 `η=0.4`。
- `h=H_ext/M_s` 是外加磁场；`m=M/M_s` 是球内磁化强度；球体均匀磁化退磁因子为 `1/3`。
- 场源是沿 z 方向、符号交替的点磁偶极子；球的受力按中心处场和总磁偶极矩计算。
- 姿态旋转 `R` 将随体坐标变为实验室坐标；`q` 存在随体系，`ω` 存在实验室系。
- 为区别两个 alpha，下文用 `a_J=Alpha_JA`、`C=C_JA`、`ξ=Xi_JA`。

## 2. 从标量磁势推出磁场与梯度

第 j 个磁体的位置为 `R_m(cos θ_j,sin θ_j,0)`，`θ_j=2πj/N+θ_d`，`s_j=(-1)^j`。令从磁体指向球心的位移为 `d_j=(x_j,y_j,z)`，`Q_j=x_j²+y_j²+z²`。

无量纲标量势为

\[
\Phi=\beta\sum_j s_j zQ_j^{-3/2},\qquad \mathbf h=-\nabla\Phi.
\]

因此每个磁体贡献

\[
\mathbf h_j=\beta s_j Q_j^{-5/2}
(3zx_j,\;3zy_j,\;2z^2-x_j^2-y_j^2).
\]

令 `T_j=3β s_j Q_j^(-7/2)`，独立梯度分量为

\[
\begin{aligned}
J_{xx}&=\sum_jT_j z(Q_j-5x_j^2),&
J_{xy}&=-\sum_j5T_jzx_jy_j,\\
J_{xz}&=\sum_jT_jx_j(x_j^2+y_j^2-4z^2),&
J_{yy}&=\sum_jT_jz(Q_j-5y_j^2),\\
J_{yz}&=\sum_jT_jy_j(x_j^2+y_j^2-4z^2).
\end{aligned}
\]

源外区域 `J=Jᵀ`、`tr J=0`，故 `Jzz=-Jxx-Jyy`。与 `eval_mag_field` 的精确场分支全部一致。

## 3. Fourier 场公式

交替排列意味着 `Φ(φ+2π/N)=-Φ(φ)`，镜面对称排除正弦项，因此仅保留 `m_l=(2l+1)N/2`：

\[
\Phi=\sum_l A_l(\rho,z)\cos[m_l(\phi-\theta_d)],\qquad Z_l=\partial_z A_l.
\]

对于单一模态，记 `ψ=m(φ-θ_d)`，由柱坐标梯度直接得到

\[
h_\rho=-A'\cos\psi,\quad h_\phi=mA\sin\psi/\rho,\quad h_z=-Z\cos\psi.
\]

对矢量求导必须同时求柱坐标基矢的导数，结果为

\[
\begin{aligned}
J_{\rho\rho}&=-A''\cos\psi,\\
J_{\rho\phi}&=m(A'/\rho-A/\rho^2)\sin\psi,\\
J_{\rho z}&=-Z'\cos\psi,\\
J_{\phi\phi}&=(m^2A/\rho^2-A'/\rho)\cos\psi,\\
J_{\phi z}&=mZ\sin\psi/\rho.
\end{aligned}
\]

实验室系变换应为 `h_lab=U(φ)h_cyl`、`J_lab=U(φ)J_cyl U(φ)ᵀ`，基矢旋转角是 `φ`，不是 `φ-θ_d`。当前代码与上述公式一致。

生成器的归一化也可从二项式展开独立得到。令

\[
\chi=\frac{z^2+\rho^2+R_m^2}{2\rho R_m},\quad
t=\chi-\sqrt{\chi^2-1}.
\]

由 `Q=(ρR_m/t)(1-2t cos θ+t²)` 展开两个因子 `(1-t exp(±iθ))^(-3/2)`，再对 N 个交替磁体求和，得到

\[
A_m=\frac{2Nz}{(\rho R_m)^{3/2}}
\frac{(3/2)_m}{m!}t^{m+3/2}
{}_2F_1(m+3/2,3/2;m+1;t^2).
\]

这与生成器的 `scale*z*radial.f` 一致；因子 `2N` 正确。链式求导中 `dt/dχ=-t/sqrt(χ²-1)`、`d²t/dχ²=(χ²-1)^(-3/2)`，也与生成器的混合导数一致。有限 Fourier 截断的误差不属于本次公式审核。

## 4. 球体退磁、磁力和磁矩

令无磁滞磁化近似为 `m_an=c h_e`，其中 `c=ξ/a_J`；定义

\[
\mathbf q=\mathbf m_{an}-\mathbf m_{irr},\qquad
\mathbf m=C\mathbf m_{an}+(1-C)\mathbf m_{irr}
=c\mathbf h_e-(1-C)\mathbf q.
\]

结合

\[
\mathbf h_{int}=\mathbf h-\mathbf m/3,\quad
\mathbf h_e=\mathbf h_{int}+a_J\mathbf m,\quad
\xi=\frac{a_J(\mu_r-1)}{1+a_J(\mu_r-1)},
\]

消元得

\[
\frac{\mathbf m}{3}=D\mathbf h-E\mathbf q,\quad
D=\frac{\mu_r-1}{\mu_r+2},\quad
E=\frac{1-C}{(1-\xi)(\mu_r+2)}.
\]

因此在实验室系，令 `q_w=R q`，由偶极子受力和转矩得到

\[
\boxed{\mathbf F=\gamma J^T(D\mathbf h-E\mathbf q_w)},\qquad
\boxed{\mathbf M=-\gamma_m E\,\mathbf q_w\times\mathbf h}.
\]

这里 `F` 是无量纲平动加速度贡献，`M` 是无量纲角加速度贡献；两者不是同单位的有量纲力和转矩。

一致的参数定义应满足

\[
\gamma=\frac{3\mu_0VM_s^2k^2}{m_b\Omega^2},\quad
\gamma_m=\frac{3\mu_0VM_s^2}{I\Omega^2},\quad
\boxed{\gamma=\eta\alpha^2\gamma_m}.
\]

代码正是按最后一个等式计算力的系数。`D h` 不贡献磁矩，因为 `h×h=0`。力中无需再乘 `1/2`：可逆项势能的 `1/2` 被 `∇|h|²=2Jᵀh` 抵消。

## 5. 随体磁场导数和姿态

刚性转动的源场满足

\[
\partial_t\mathbf h=\hat z\times\mathbf h-J(\hat z\times\mathbf r).
\]

沿球心运动并变换到随体系：

\[
\boxed{\dot{\mathbf h}_{body}=R^T
\{J(\mathbf v-\hat z\times\mathbf r)
+\hat z\times\mathbf h-\boldsymbol\omega\times\mathbf h\}}.
\]

所以相对速度是 `(vx+y,vy-x)`，世界系场导数的额外项是 `(-hy,hx,0)`。两个叉乘项描述底盘转动与球体转动，均不可省略。与 `eval_body_field_derivative` 一致。

`ω` 在实验室系，故姿态四元数应满足 `Qdot=(0,ω)Q/2`，当前乘法顺序正确。

## 6. 磁滞方程独立消元

本节明确以当前实现对应的投影式矢量闭合为前提：

\[
\dot{\mathbf m}_{irr}=\frac{\mathbf q}{K|\mathbf q|}
\max(\mathbf q\cdot\dot{\mathbf h}_e,0),\quad
\dot{\mathbf q}=c\dot{\mathbf h}_e-\dot{\mathbf m}_{irr}.
\]

令 `h0=3 h_body/(μr+2)`、`L=a_J(1-C)-E`。由上一节退磁关系独立得到

\[
\boxed{\dot{\mathbf h}_0=(1-\xi)\dot{\mathbf h}_e+L\dot{\mathbf q}}.
\]

零分支 `q·hedot≤0` 给出

\[
I_0=\frac{a_J(1-C\xi)}{\xi}-E,\quad
\dot{\mathbf q}=\frac{\dot{\mathbf h}_0}{I_0},\quad
\dot{\mathbf h}_e=\frac{a_J}{\xi I_0}\dot{\mathbf h}_0.
\]

活动分支令 `n=q/|q|`、`d=-|q|/K`，则

\[
\dot{\mathbf q}=(cI+d\mathbf n\mathbf n^T)\dot{\mathbf h}_e,\quad
\dot{\mathbf h}_0=(aI+b\mathbf n\mathbf n^T)\dot{\mathbf h}_e,
\]

其中

\[
a=1-C\xi-Ec,\qquad b=Ld.
\]

按平行与垂直于 n 的两个本征方向求逆：

\[
\boxed{\dot{\mathbf q}=\frac ca\dot{\mathbf h}_0+
\left(\frac{c+d}{a+b}-\frac ca\right)
(\mathbf n\cdot\dot{\mathbf h}_0)\mathbf n}.
\]

代码中第二项系数 `d/a-b(c+d)/(a(a+b))` 与方框中的系数完全等价。

当前常数满足 `a>0`、`L<0`，故 `b≥0`、`a+b>0`；零分支和活动分支的 `q·hedot` 符号一致，用零分支候选导数判断开关在这些参数下自洽。

**模型适用范围：** 这里验证的是线性 `m_an=c h_e` 和上述投影式不可逆演化律。若原始物理模型要求非线性饱和曲线，或不可逆演化率与 `|hedot|` 成正比，则需要另一套方程；当前仓库不足以确认这些是否属于原始建模意图。

## 7. 平动、转动、摩擦与纯滚动

令 `G=F_parallel-δ|v|v`，支持力 `N=γ_g-Fz`。接触点在球心下方，因此

\[
\dot{\mathbf v}=\mathbf G+\mathbf f,\quad
\dot{\boldsymbol\omega}=\mathbf M+
\frac{(f_y,-f_x,0)}{\eta\alpha},\quad
\mathbf s=(v_x-\alpha\omega_y,v_y+\alpha\omega_x).
\]

滑动时 `f=-μN s/|s|`。纯滚动要求 `sdot=0`，消去摩擦力得

\[
\boxed{\dot\omega_x=\frac\eta{1+\eta}M_x-\frac{G_y}{\alpha(1+\eta)}},\quad
\boxed{\dot\omega_y=\frac\eta{1+\eta}M_y+\frac{G_x}{\alpha(1+\eta)}},\quad
\dot\omega_z=M_z.
\]

`vdot=(α ωdot_y,-α ωdot_x)`。全部与 `eval_rhs` 一致。

滑移方程为

\[
\dot{\mathbf s}=\mathbf a-b\frac{\mathbf s}{|\mathbf s|},\quad
\mathbf a=\mathbf G+\alpha(-M_y,M_x),\quad
b=(1+1/\eta)\mu N.
\]

静摩擦可维持滚动的判据为 `|a|≤b`。若 a、b 固定且 `|a|<b`，则

\[
\frac d{dt}(b|\mathbf s|+\mathbf a\cdot\mathbf s)=|\mathbf a|^2-b^2,
\]

从而停止时间公式 `tc=(b|s0|+a·s0)/(b²-|a|²)` 正确；它不是任意时变外力下的精确停止时间。本次不评价如何将它用于时间推进。

## 8. 独立核对证据

辅助核对见 `scratch/audit_2d_formulas.cpp`：80 个位置/角度样本，从独立标量势差分磁场与梯度；对运动轨迹上的随体场作差分；将磁滞解代回未求逆的本构方程；将滚动角加速度代回力矩平衡。

| 核对量 | 最大误差/残差 |
|---|---:|
| 标量势差分得到的磁场 | `4.82e-10` |
| 独立差分得到的场梯度 | `4.26e-7` |
| 随体磁场导数 | `9.10e-10` |
| 原始磁滞方程残差 | `8.82e-16` |
| 滚动力矩平衡残差 | `1.77e-17` |

前三项和磁滞项按 `1+被核对量的范数` 缩放，滚动项为绝对残差；磁滞覆盖零分支 43 个、活动分支 37 个样本。这些核对只辅助验证推导，不用于声称模型符合实验，也不是积分器测试。未修改 kernel。
