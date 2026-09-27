# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

from __future__ import annotations
import numpy as np

alpha = 0.1
gamma = 0
epsilon = 0
delta = 0
eta = 0.4
mu_s = 0.1
mu_k = 0.1


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
total_time = 20
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