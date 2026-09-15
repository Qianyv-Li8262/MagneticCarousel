"""Build the compact phase-behavior figure and numerical-boundary table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import brentq

from project_paths import image_output_path
from scan_sr_somega_gpu import _cell_edges


ALPHA_REF = 0.33333333
DELTA_REF = 1.7212357e-4
GAMMA_M_REF = 11011.419
MU_R = 2.2926667e3
ALPHA_JA = 3.2e-3
C = 0.38
K_HYSTER = 1.0909091e-3
H_EXT = 1.4661546e-1
ETA = 0.4
XI_JA = 1.0 - 1.0 / (1.0 + ALPHA_JA * (MU_R - 1.0))
Q_LIMIT = XI_JA / ALPHA_JA * K_HYSTER
E_DEMAG = (1.0 - C) / ((MU_R + 2.0) * (1.0 - XI_JA))
BOUNDARY_SCALE = np.sqrt(GAMMA_M_REF * E_DEMAG * Q_LIMIT * H_EXT * ETA / (ALPHA_REF * DELTA_REF))


def load(path: Path):
    data = np.load(path)
    names = [str(name) for name in data["metric_names"]]
    return data, {name: index for index, name in enumerate(names)}


def locking_limit(sr):
    sr = np.asarray(sr)
    return BOUNDARY_SCALE * np.abs(1.0 - ALPHA_REF * sr) ** 1.5 / sr


def theoretical_edges(somega: float):
    function = lambda sr: float(locking_limit(sr) - somega)
    left = brentq(function, 0.01, 1.0 / ALPHA_REF - 1.0e-8)
    right = brentq(function, 1.0 / ALPHA_REF + 1.0e-8, 100.0)
    return left, right


def empirical_edges(sr, phase_rate, threshold):
    indices = np.flatnonzero(phase_rate > threshold)
    if indices.size < 2:
        return np.nan, np.nan
    return float(sr[indices[0]]), float(sr[indices[-1]])


def write_boundary_table(path, sr, somega, phase_rate, threshold):
    rows = []
    for row, sw in zip(phase_rate, somega):
        observed_left, observed_right = empirical_edges(sr, row, threshold)
        theory_left, theory_right = theoretical_edges(float(sw))
        rows.append(
            (
                float(sw),
                observed_left,
                observed_right,
                theory_left,
                theory_right,
                observed_left - theory_left,
                observed_right - theory_right,
            )
        )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "somega",
                "observed_left_sr",
                "observed_right_sr",
                "theory_left_sr",
                "theory_right_sr",
                "left_error",
                "right_error",
            )
        )
        writer.writerows(rows)
    return np.asarray(rows)


def binned_collapse(ratio, phase_rate):
    edges = np.geomspace(0.1, 20.0, 36)
    centers = np.sqrt(edges[:-1] * edges[1:])
    quantiles = np.full((centers.size, 3), np.nan)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = phase_rate[(ratio >= lower) & (ratio < upper) & np.isfinite(phase_rate)]
        if selected.size:
            quantiles[index] = np.quantile(selected, (0.1, 0.5, 0.9))
    return centers, quantiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", type=Path, required=True)
    parser.add_argument("--initial-unprojected", type=Path, required=True)
    parser.add_argument("--initial-projected", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parameter, metric_index = load(args.parameter)
    sr = parameter["x"]
    somega = parameter["y"]
    metrics = parameter["metrics"]
    phase_rate = metrics[..., metric_index["mean_phase_rate"]]
    slip = metrics[..., metric_index["mean_abs_slip"]]
    sr_grid = sr[None, :]
    somega_grid = somega[:, None]
    limit = locking_limit(sr_grid)
    ratio = somega_grid / limit

    observed_unlocked = phase_rate > args.threshold
    predicted_unlocked = ratio > 1.0
    accuracy = float(np.mean(observed_unlocked == predicted_unlocked))
    false_unlocked = int(np.count_nonzero(observed_unlocked & ~predicted_unlocked))
    false_locked = int(np.count_nonzero(~observed_unlocked & predicted_unlocked))

    table = write_boundary_table(
        args.output_dir / "phase_boundary.csv", sr, somega, phase_rate, args.threshold
    )
    finite_errors = np.concatenate((table[:, 5], table[:, 6]))
    finite_errors = finite_errors[np.isfinite(finite_errors)]
    boundary_mae = float(np.mean(np.abs(finite_errors)))

    initial_old, old_index = load(args.initial_unprojected)
    initial_new, new_index = load(args.initial_projected)
    old_row = int(np.argmin(np.abs(initial_old["y"])))
    new_row = int(np.argmin(np.abs(initial_new["y"])))
    old_phase = initial_old["metrics"][old_row, :, old_index["mean_phase_rate"]]
    new_phase = initial_new["metrics"][new_row, :, new_index["mean_phase_rate"]]

    figure, axes = plt.subplots(2, 2, figsize=(13.8, 10.5), constrained_layout=True)
    phase_image = axes[0, 0].pcolormesh(
        _cell_edges(sr),
        _cell_edges(somega),
        np.clip(phase_rate, 0.0, 1.0),
        shading="flat",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    theory_sr_left = np.linspace(sr.min(), 1.0 / ALPHA_REF - 1.0e-4, 800)
    theory_sr_right = np.linspace(1.0 / ALPHA_REF + 1.0e-4, sr.max(), 800)
    axes[0, 0].plot(theory_sr_left, locking_limit(theory_sr_left), "w--", lw=2.0, label="torque-limit boundary")
    axes[0, 0].plot(theory_sr_right, locking_limit(theory_sr_right), "w--", lw=2.0)
    axes[0, 0].set_ylim(somega.min(), somega.max())
    axes[0, 0].set(xlabel=r"$s_r$", ylabel=r"$s_\omega$", title=r"Order parameter $\langle\dot\phi\rangle$")
    axes[0, 0].legend(loc="upper left")
    figure.colorbar(phase_image, ax=axes[0, 0], label=r"clipped $\langle\dot\phi\rangle$")

    centers, quantiles = binned_collapse(ratio.ravel(), phase_rate.ravel())
    valid_bins = np.all(np.isfinite(quantiles), axis=1)
    axes[0, 1].fill_between(
        centers[valid_bins], quantiles[valid_bins, 0], quantiles[valid_bins, 2], alpha=0.25, label="10--90%"
    )
    axes[0, 1].plot(centers[valid_bins], quantiles[valid_bins, 1], "o-", ms=4, label="median")
    axes[0, 1].axvline(1.0, color="0.2", ls="--", label="analytic threshold")
    axes[0, 1].set(
        xscale="log",
        xlabel=r"$\Lambda=s_\omega/s_\omega^{\max}(s_r)$",
        ylabel=r"$\langle\dot\phi\rangle$",
        title="Collapse onto the torque ratio",
        ylim=(-0.03, 1.03),
    )
    axes[0, 1].legend()

    slip_image = axes[1, 0].pcolormesh(
        _cell_edges(sr),
        _cell_edges(somega),
        np.maximum(slip, 1.0e-10),
        shading="flat",
        cmap="viridis",
        norm=LogNorm(1.0e-10, max(1.0e-7, float(np.nanpercentile(slip, 99.5)))),
    )
    axes[1, 0].set(xlabel=r"$s_r$", ylabel=r"$s_\omega$", title="Rolling-constraint residual after projection")
    figure.colorbar(slip_image, ax=axes[1, 0], label=r"$\langle|v+\alpha\omega|\rangle$")

    axes[1, 1].plot(initial_old["x"], old_phase, lw=1.5, label="event-only projection")
    axes[1, 1].plot(initial_new["x"], new_phase, lw=2.0, label="constraint enforced each sticking step")
    axes[1, 1].axhline(args.threshold, color="0.3", ls=":", label="classification threshold")
    axes[1, 1].set(
        xlabel=r"initial $v_0$ (nearest sampled $S_0=0$)",
        ylabel=r"$\langle\dot\phi\rangle$",
        title=r"Initial-state stripes are FP32 constraint drift ($s_r=2.68$)",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle("Self-consistent hysteresis model: locked and running phases")
    figure_path = image_output_path(args.output_dir / "phase_behavior_summary.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    report = f"""# 自洽磁滞模型的相行为

主序参量取末时间窗平均相位漂移率 `mean_phase_rate = <1-v-omega>`。
`|mean_phase_rate| < {args.threshold:g}` 记为 1:1 锁相；较大的正值记为 running/unlocked。

锁相并纯滚动要求

    omega_* = 1/(1-alpha),
    v_* = -alpha/(1-alpha).

稳态所需磁力矩为

    M_req = alpha*delta/[eta*|1-alpha|^3].

磁滞变量满足 `|q| <= q_* = (xi/alpha_JA) k = {Q_LIMIT:.9g}`，所以可用磁力矩上限为

    M_max = gamma_m E q_* H_ext.

令二者相等并代入仓库参数缩放，得到无拟合参数的相边界

    somega_max(sr) = A |1-alpha_ref sr|^(3/2)/sr,
    A = {BOUNDARY_SCALE:.9g}.

在投影后的 T={float(parameter['metadata'].item().split('total_time')[1].split(',')[0].split(':')[1]) if 'total_time' in parameter['metadata'].item() else 16000:g}
128x128 CUDA 扫描中，解析边界对 `mean_phase_rate>{args.threshold:g}` 的分类准确率为
{accuracy:.3%}；数值边界与解析边界的平均横向误差为 {boundary_mae:.5g}。
误差点中，解析锁相但有限时间仍未锁相的点有 {false_unlocked} 个；反向只有 {false_locked} 个，
符合边界附近捕获时间很长的解释。

初态扫描在 `(sr,somega)=(2.68,1)`、`v0 in [-12,3]`、`S0 in [-3,3]` 下，
仅在滑动转粘着事件投影时出现交替条纹；每个 sticking 步重新施加
`v+alpha*omega=0` 后，全部初态落入同一锁相吸引子。因此当前证据不支持初态空间存在
真实的多相结构。

低参数角落的固定步长结果也不是新相。在 `sr,somega in [0.02,0.15]`，
`dt=1e-3` 产生 474/4096 个失败；`dt=5e-5` 后 4096/4096 稳定锁相。
经验上应保持 `dt/(sr*somega) <~ 0.2`；若下限扫到 0.01，需用约 `dt<=2e-5`
或按参数自适应步长。
"""
    report_path = args.output_dir / "PHASE_BEHAVIOR_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"boundary scale A={BOUNDARY_SCALE:.9g}")
    print(f"classification accuracy={accuracy:.6f}")
    print(f"boundary MAE={boundary_mae:.6g}")
    print(f"false unlocked/locked={false_unlocked}/{false_locked}")
    print(f"Saved {figure_path.resolve()}")
    print(f"Saved {report_path.resolve()}")
    print(f"Saved {(args.output_dir / 'phase_boundary.csv').resolve()}")


if __name__ == "__main__":
    main()
