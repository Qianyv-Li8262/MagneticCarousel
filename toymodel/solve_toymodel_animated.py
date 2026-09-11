from __future__ import annotations
import numpy as np

gamma = 2.5
mu_s = 0.02
alpha = 0.918
eta = 0.4
mu_k = 0.02
delta = 6.4e-5
epsilon = 50


def evalRhs(input: np.ndarray, sliding: bool, time: float,slip_sign: float = 0) -> np.ndarray:
    ans = np.zeros(4)
    ans[0] = input[1]
    ans[2] = input[3]
    if sliding:
        S = input[1] + alpha * input[3]
        # 一个滑动 segment 内固定摩擦方向，防止 midpoint 自己跨 S=0
        if slip_sign == 0:
            slip_sign = np.sign(S)
        N = gamma + epsilon * np.cos(time - input[0] - input[2])
        if N<0:
            raise ValueError("支持力小于0")
        F = -slip_sign * mu_k * N
        ans[1] = (
            epsilon * np.sin(time - input[0] - input[2])
            + F
            - delta * input[1] * np.abs(input[1])
        )
        ans[3] = (
            epsilon / (eta * alpha**2)
            * np.sin(time - input[0] - input[2])
            + F / (eta * alpha)
        )
    else:
        ans[1] = (
            epsilon * (1 - 1 / alpha)
            * np.sin(time - input[0] - input[2])
            - delta * input[1] * np.abs(input[1])
        ) / (1 + eta)
        ans[3] = -ans[1] / alpha

    return ans

def fwd1Step(input: np.ndarray,sliding: bool,time: float,dt: float,slip_sign: float = 0
             ) -> tuple[np.ndarray, bool, float]:

    # trial midpoint step
    midpoint_trial = (
        input
        + dt * 0.5 * evalRhs(input, sliding, time, slip_sign)
    )

    final_trial = (
        input
        + dt * evalRhs(
            midpoint_trial,
            sliding,
            time + dt * 0.5,
            slip_sign
        )
    )

    # ==========================================================
    # 当前是 sliding
    # ==========================================================
    if sliding:
        S_init = input[1] + alpha * input[3]
        S_final = final_trial[1] + alpha * final_trial[3]

        if slip_sign == 0:
            slip_sign = np.sign(S_init)

        # 没过零，整个 midpoint step 接受
        if S_init * S_final >= 0:
            return final_trial, True, slip_sign

        # ------------------------------------------------------
        # S 过零，线性插值得到事件时刻
        # ------------------------------------------------------
        t_star = dt * S_init / (S_init - S_final)

        # 用 midpoint 重新积分到 t_star
        midpoint_star = (
            input
            + t_star * 0.5
            * evalRhs(input, True, time, slip_sign)
        )

        event_tstar = (
            input
            + t_star
            * evalRhs(
                midpoint_star,
                True,
                time + t_star * 0.5,
                slip_sign
            )
        )

        # event 点理论上 S=0
        # 线性 event 定位会有小误差，直接投影回 rolling constraint
        u = event_tstar[1]
        omega = event_tstar[3]

        event_tstar[1] = (
            u - eta * alpha * omega
        ) / (1 + eta)

        event_tstar[3] = -event_tstar[1] / alpha

        # ------------------------------------------------------
        # 在 event 点重新算维持 stick 所需静摩擦
        # ------------------------------------------------------
        phase = (
            time + t_star
            - event_tstar[0]
            - event_tstar[2]
        )

        N = (
            gamma
            + epsilon * np.cos(phase)
        )
        if N<0:
             raise ValueError("支持力小于0")
        F_req = (
            eta * delta
            * event_tstar[1]
            * np.abs(event_tstar[1])
            - epsilon
            * (eta + 1 / alpha)
            * np.sin(phase)
        ) / (1 + eta)   

        remain = dt - t_star

        # ------------------------------------------------------
        # 静摩擦够：slip -> stick
        # ------------------------------------------------------
        if np.abs(F_req) <= mu_s * N:
            sliding = False
            slip_sign = 0

            midpoint_remain = (
                event_tstar
                + remain * 0.5
                * evalRhs(
                    event_tstar,
                    False,
                    time + t_star
                )
            )

            final = (
                event_tstar
                + remain
                * evalRhs(
                    midpoint_remain,
                    False,
                    time + t_star + remain * 0.5
                )
            )

            return final, False, 0

        # ------------------------------------------------------
        # 静摩擦不够：穿过 S=0 后继续滑
        # ------------------------------------------------------
        else:
            slip_sign = -np.sign(F_req)

            midpoint_remain = (
                event_tstar
                + remain * 0.5
                * evalRhs(
                    event_tstar,
                    True,
                    time + t_star,
                    slip_sign
                )
            )

            final = (
                event_tstar
                + remain
                * evalRhs(
                    midpoint_remain,
                    True,
                    time + t_star + remain * 0.5,
                    slip_sign
                )
            )

            return final, True, slip_sign

    # ==========================================================
    # 当前是 sticking
    # ==========================================================
    else:
        phase_init = time - input[0] - input[2]

        N_init = (
            gamma
            + epsilon * np.cos(phase_init)
        )
        if N_init<0:
            raise ValueError("支持力小于0")
        F_req_init = (
            eta * delta
            * input[1]
            * np.abs(input[1])
            - epsilon
            * (eta + 1 / alpha)
            * np.sin(phase_init)
        ) / (1 + eta)

        g_init = (
            mu_s * N_init
            - np.abs(F_req_init)
        )

        phase_final = (
            time + dt
            - final_trial[0]
            - final_trial[2]
        )

        N_final = (
            gamma
            + epsilon * np.cos(phase_final)
        )

        F_req_final = (
            eta * delta
            * final_trial[1]
            * np.abs(final_trial[1])
            - epsilon
            * (eta + 1 / alpha)
            * np.sin(phase_final)
        ) / (1 + eta)

        g_final = (
            mu_s * N_final
            - np.abs(F_req_final)
        )

        # 静摩擦仍然够
        if g_final >= 0:
            return final_trial, False, 0

        # ------------------------------------------------------
        # stick -> slip
        # ------------------------------------------------------
        t_star = dt * g_init / (g_init - g_final)

        midpoint_star = (
            input
            + t_star * 0.5
            * evalRhs(input, False, time)
        )

        event_tstar = (
            input
            + t_star
            * evalRhs(
                midpoint_star,
                False,
                time + t_star * 0.5
            )
        )

        phase = (
            time + t_star
            - event_tstar[0]
            - event_tstar[2]
        )

        F_req = (
            eta * delta
            * event_tstar[1]
            * np.abs(event_tstar[1])
            - epsilon
            * (eta + 1 / alpha)
            * np.sin(phase)
        ) / (1 + eta)

        # 实际摩擦不够时，S 朝 F_req 所需方向的反方向产生
        slip_sign = -np.sign(F_req)

        remain = dt - t_star

        midpoint_remain = (
            event_tstar
            + remain * 0.5
            * evalRhs(
                event_tstar,
                True,
                time + t_star,
                slip_sign
            )
        )

        final = (
            event_tstar
            + remain
            * evalRhs(
                midpoint_remain,
                True,
                time + t_star + remain * 0.5,
                slip_sign
            )
        )

        return final, True, slip_sign








dt = 1e-3
total_time = 300
save_every = 10


def evolve(initial: np.ndarray):
    state = np.array(initial, dtype=float)

    # state = [x, vx, theta, omega]

    time = 0.0

    # ---------------------------------------------------------
    # 自动判断初始 stick/slip 状态
    # ---------------------------------------------------------
    S = state[1] + alpha * state[3]

    if np.abs(S) > 1e-10:
        sliding = True
        slip_sign = np.sign(S)

    else:
        phase = time - state[0] - state[2]

        N = gamma + epsilon * np.cos(phase)

        F_req = (
            eta * delta
            * state[1] * np.abs(state[1])
            - epsilon
            * (eta + 1 / alpha)
            * np.sin(phase)
        ) / (1 + eta)

        if np.abs(F_req) <= mu_s * N:
            sliding = False
            slip_sign = 0

            # 消掉可能存在的极小初始数值误差
            state[3] = -state[1] / alpha

        else:
            sliding = True
            slip_sign = -np.sign(F_req)

    # ---------------------------------------------------------
    # 只保存 matplotlib 真正需要的数据
    # ---------------------------------------------------------
    x_history = [state[0]]
    theta_history = [state[2]]
    sliding_history = [sliding]

    steps = int(total_time / dt)

    for i in range(steps):

        state, sliding, slip_sign = fwd1Step(
            state,
            sliding,
            time,
            dt,
            slip_sign
        )

        time += dt

        if (i + 1) % save_every == 0:
            x_history.append(state[0])
            theta_history.append(state[2])
            sliding_history.append(sliding)

    return {
        "x": np.asarray(x_history),
        "theta": np.asarray(theta_history),
        "is_sliding": np.asarray(sliding_history, dtype=bool),
        "frame_dt": dt * save_every,
    }

# =============================================================
# matplotlib animation
# =============================================================
def animate(initial: np.ndarray):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, FancyArrowPatch

    history = evolve(initial)
    xs = history["x"]
    thetas = history["theta"]
    is_sliding = history["is_sliding"]
    frame_dt = history["frame_dt"]
    times = np.arange(len(xs)) * frame_dt

    wavelength = 2 * np.pi       # B ~ sin(t - x): k = 1
    wave_speed = 1.0             # omega / k = 1
    view_half_width = 0.85 * wavelength
    wave_y0 = 0.92
    wave_amp = 0.20
    moment_len = 0.38

    # 左边：球 + 行波；右边：x(t), theta(t)
    fig = plt.figure(figsize=(12.8, 5.0), dpi=120)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(1.75, 1.0),
        height_ratios=(1, 1),
        left=0.055, right=0.975, bottom=0.11, top=0.91,
        wspace=0.24, hspace=0.28
    )

    ax = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_theta = fig.add_subplot(gs[1, 1], sharex=ax_x)

    # ---------------------------------------------------------
    # 主动画
    # ---------------------------------------------------------
    ax.set_ylim(-0.45, 1.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.axhline(0, color="0.20", lw=1.8, zorder=1)

    x0 = xs[0]
    ax.set_xlim(x0 - view_half_width, x0 + view_half_width)

    xx = np.linspace(x0 - view_half_width, x0 + view_half_width, 900)
    wave, = ax.plot(xx, wave_y0 + wave_amp * np.sin(-xx), lw=2.0, zorder=2)
    ax.axhline(wave_y0, color="0.75", lw=0.8, ls="--", zorder=0)

    ball = Circle((x0, alpha), alpha, ec="0.12", lw=1.7, zorder=4)
    ball.set_facecolor("tab:orange" if is_sliding[0] else "tab:blue")
    ax.add_patch(ball)

    center, = ax.plot([x0], [alpha], "o", ms=2.8, color="0.10", zorder=6)
    magnetic_moment = FancyArrowPatch(
        (x0, alpha),
        (x0 + moment_len * np.cos(thetas[0]),
         alpha + moment_len * np.sin(thetas[0])),
        arrowstyle="-|>", mutation_scale=14, lw=2.0,
        color="0.08", zorder=7
    )
    ax.add_patch(magnetic_moment)

    ax.text(
        0.015, 0.96,
        r"$B \propto \sin(t-x)$     $\lambda=2\pi$     $c=1\;\rightarrow$",
        transform=ax.transAxes, fontsize=10.5, va="top"
    )
    status_text = ax.text(
        0.985, 0.96, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=10.0
    )

    # ---------------------------------------------------------
    # 实时 x(t), theta(t)
    # ---------------------------------------------------------
    def padded_limits(data):
        lo = float(np.min(data))
        hi = float(np.max(data))
        span = hi - lo
        if span < 1e-12:
            span = max(1.0, abs(lo) * 0.2)
        pad = 0.08 * span
        return lo - pad, hi + pad

    ax_x.set_title("trajectory", fontsize=11)
    ax_x.set_ylabel(r"$x(t)$")
    ax_theta.set_ylabel(r"$\theta(t)$")
    ax_theta.set_xlabel("t")

    t_end = times[-1] if len(times) > 1 else frame_dt
    ax_x.set_xlim(0, t_end)
    ax_x.set_ylim(*padded_limits(xs))
    ax_theta.set_ylim(*padded_limits(thetas))

    ax_x.grid(alpha=0.25)
    ax_theta.grid(alpha=0.25)

    x_trace, = ax_x.plot([], [], lw=1.8)
    theta_trace, = ax_theta.plot([], [], lw=1.8)
    x_now, = ax_x.plot([], [], "o", ms=4.5)
    theta_now, = ax_theta.plot([], [], "o", ms=4.5)
    x_time_marker = ax_x.axvline(0, lw=1.0, ls="--", color="0.35")
    theta_time_marker = ax_theta.axvline(0, lw=1.0, ls="--", color="0.35")

    # evolve() 默认每 0.01 保存一帧；显示时限制在约 50 fps
    frame_step = max(1, int(round(1.0 / (50.0 * frame_dt))))
    frames = range(0, len(xs), frame_step)
    interval = 1000.0 * frame_dt * frame_step

    def update(i):
        t = times[i]
        x = xs[i]
        theta = thetas[i]

        # 跟随小球移动主视窗
        left = x - view_half_width
        right = x + view_half_width
        ax.set_xlim(left, right)

        xx = np.linspace(left, right, 900)
        wave.set_data(xx, wave_y0 + wave_amp * np.sin(t - xx))

        ball.center = (x, alpha)
        ball.set_facecolor("tab:orange" if is_sliding[i] else "tab:blue")
        center.set_data([x], [alpha])
        magnetic_moment.set_positions(
            (x, alpha),
            (x + moment_len * np.cos(theta),
             alpha + moment_len * np.sin(theta))
        )

        mode = "SLIDING" if is_sliding[i] else "ROLLING / STICK"
        status_text.set_text(
            f"t = {t:6.2f}    x = {x:7.3f}\n"
            f"theta = {theta:7.3f}    {mode}"
        )

        # 轨迹只画到当前帧，因此是实时长出来的
        x_trace.set_data(times[:i + 1], xs[:i + 1])
        theta_trace.set_data(times[:i + 1], thetas[:i + 1])
        x_now.set_data([t], [x])
        theta_now.set_data([t], [theta])
        x_time_marker.set_xdata([t, t])
        theta_time_marker.set_xdata([t, t])

        return (
            wave, ball, center, magnetic_moment, status_text,
            x_trace, theta_trace, x_now, theta_now,
            x_time_marker, theta_time_marker
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=False,
        repeat=True
    )

    return ani


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 只改这一行初态即可： [x, vx, theta, omega]
    initial = np.array([0.0, 0.0, 0.0, 0.0])

    ani = animate(initial)
    plt.show()
