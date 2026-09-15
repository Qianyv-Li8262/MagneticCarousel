"""Analyze the full-range logarithm-to-cusp crossover after FP64 refinement."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from project_paths import RESULTS_DIR, SOMEGA_FP64_LOG_DIR, image_output_path  # noqa: E402

LOG_DIR = SOMEGA_FP64_LOG_DIR
OUTPUT_DIR = RESULTS_DIR / "critical-slowing"
SUMMARY = OUTPUT_DIR / "somega_fp64_limit_summary.csv"
FIGURE = image_output_path(OUTPUT_DIR / "somega_fp64_limit.png")


def rows(path: Path):
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    for line in raw.decode(encoding).splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            yield float(fields[0]), int(fields[1]), float(fields[2])
        except ValueError:
            pass


def load(max_fine: int | None = None):
    successful: dict[float, list[float]] = {}
    failed = []
    for path in LOG_DIR.glob("slow_*.log"):
        if max_fine is not None and not (path.stem == "slow_large" or path.stem.startswith("slow_fine")):
            continue
        if path.stem.startswith("slow_fine"):
            index = int(path.stem.removeprefix("slow_fine"))
            if max_fine is not None and index > max_fine:
                continue
        for x, classification, t in rows(path):
            if classification == 1 and t >= 0:
                successful.setdefault(x, []).append(t)
            elif classification == 0:
                failed.append(x)
    x = np.asarray(sorted(successful))
    t = np.asarray([np.mean(successful[value]) for value in x])
    return x, t, np.asarray(failed)


def fit_model(kind: str, r: np.ndarray, t: np.ndarray):
    if kind == "log":
        start, lower, upper = [18000, 800], [-1e6, 0], [1e6, 1e6]
        predict = lambda q: q[0] + q[1] * np.log(1 / r)
    elif kind == "crossover_half":
        start, lower, upper = [21210, 1730, np.log(0.9)], [0, 0, -30], [1e6, 1e6, 30]
        predict = lambda q: q[0] - q[1] * np.arcsinh(np.sqrt(r / np.exp(q[2])))
    elif kind == "crossover_free":
        start = [21210, 1710, np.log(0.9), 0.5]
        lower, upper = [0, 0, -30, 0.01], [1e6, 1e6, 30, 2]
        predict = lambda q: q[0] - q[1] * np.arcsinh((r / np.exp(q[2])) ** q[3])
    else:
        raise ValueError(kind)
    result = least_squares(lambda q: predict(q) - t, start, bounds=(lower, upper), x_scale="jac",
                           xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=100000)
    residual = predict(result.x) - t
    rss = float(residual @ residual)
    n, k = len(t), len(result.x)
    return result.x, predict, residual, n * np.log(rss / n) + 2 * k


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    x, t, failed = load()
    x_reliable, t_reliable, _ = load(max_fine=9)
    critical_lower = float(np.max(x))
    critical_upper = float(np.min(failed[failed > critical_lower]))
    ulp = float(np.spacing(critical_lower))
    # The midpoint is not representable in FP64; form distances without first
    # rounding the conceptual half-ULP midpoint back to an endpoint.
    d = (critical_lower - x) + 0.5 * ulp
    keep = d > 0
    x, t, d = x[keep], t[keep], d[keep]
    scale = 1e-10
    r = d / scale

    log_q, log_predict, log_residual, log_aic = fit_model("log", r, t)
    half_q, half_predict, half_residual, half_aic = fit_model("crossover_half", r, t)
    free_q, free_predict, free_residual, free_aic = fit_model("crossover_free", r, t)

    intermediate = (d >= 1e-8) & (d <= 1e-3)
    matrix = np.column_stack((np.ones(np.count_nonzero(intermediate)), np.log(1 / d[intermediate])))
    intermediate_q = np.linalg.lstsq(matrix, t[intermediate], rcond=None)[0]
    intermediate_residual = matrix @ intermediate_q - t[intermediate]

    d_reliable = (critical_lower - x_reliable) + 0.5 * ulp
    annulus = (d_reliable >= 1e-12) & (d_reliable <= 3e-11)
    da = d_reliable[annulus] / 3e-11
    ta = t_reliable[annulus]
    cusp = least_squares(
        lambda q: q[0] - q[1] * da ** q[2] - ta,
        [21200, 900, 0.5], bounds=([0, 0, 0.01], [1e6, 1e6, 2]), x_scale="jac",
        xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    cusp_residual = cusp.fun

    records = [
        {"quantity": "critical_lower", "value": critical_lower},
        {"quantity": "critical_upper", "value": critical_upper},
        {"quantity": "critical_width", "value": critical_upper - critical_lower},
        {"quantity": "fp64_ulp", "value": ulp},
        {"quantity": "full_log_rmse", "value": np.sqrt(np.mean(log_residual**2))},
        {"quantity": "full_log_r2", "value": 1 - log_residual @ log_residual / np.sum((t - t.mean()) ** 2)},
        {"quantity": "crossover_half_Tc", "value": half_q[0]},
        {"quantity": "crossover_half_A", "value": half_q[1]},
        {"quantity": "crossover_half_dx", "value": scale * np.exp(half_q[2])},
        {"quantity": "crossover_half_rmse", "value": np.sqrt(np.mean(half_residual**2))},
        {"quantity": "crossover_half_r2", "value": 1 - half_residual @ half_residual / np.sum((t - t.mean()) ** 2)},
        {"quantity": "crossover_free_beta", "value": free_q[3]},
        {"quantity": "crossover_free_Tc", "value": free_q[0]},
        {"quantity": "crossover_free_dx", "value": scale * np.exp(free_q[2])},
        {"quantity": "crossover_free_rmse", "value": np.sqrt(np.mean(free_residual**2))},
        {"quantity": "crossover_free_r2", "value": 1 - free_residual @ free_residual / np.sum((t - t.mean()) ** 2)},
        {"quantity": "delta_aic_log_minus_half", "value": log_aic - half_aic},
        {"quantity": "intermediate_log_B", "value": intermediate_q[1]},
        {"quantity": "intermediate_log_rmse", "value": np.sqrt(np.mean(intermediate_residual**2))},
        {"quantity": "annulus_cusp_beta", "value": cusp.x[2]},
        {"quantity": "annulus_cusp_rmse", "value": np.sqrt(np.mean(cusp_residual**2))},
    ]
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("quantity", "value"))
        writer.writeheader()
        writer.writerows(records)

    order = np.argsort(d)
    dline = np.geomspace(np.min(d), np.max(d), 1200)
    rline = dline / scale
    half_line = half_q[0] - half_q[1] * np.arcsinh(np.sqrt(rline / np.exp(half_q[2])))
    log_line = log_q[0] + log_q[1] * np.log(1 / rline)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].semilogx(d, t, ".", ms=3, alpha=0.55, label="all successful points")
    axes[0].semilogx(dline, half_line, "k-", lw=2,
                     label=r"global crossover, $\beta=1/2$")
    axes[0].semilogx(dline, log_line, "--", color="tab:red", lw=1.4, label="single global log")
    axes[0].set(xlabel=r"$d=s_c-s_\omega$", ylabel="transition time", title="Full dynamic range")
    axes[0].legend(fontsize=8)

    near = d <= 1e-9
    axes[1].semilogx(d[near], t[near], ".", ms=4, alpha=.7)
    axes[1].semilogx(dline[dline <= 1e-9], half_line[dline <= 1e-9], "k-", lw=2)
    axes[1].axvspan(np.min(d), 2e-14, color="tab:red", alpha=.10, label="FP64-sensitive region")
    axes[1].set(xlabel=r"$d=s_c-s_\omega$", ylabel="transition time", title="Cusp and numerical floor")
    axes[1].legend(fontsize=8)

    finest = sorted(set(rows(LOG_DIR / "slow_fp64_exhaustive.log")))
    fx = np.asarray([item[0] for item in finest])
    fc = np.asarray([item[1] for item in finest])
    ft = np.asarray([item[2] for item in finest])
    ulp_index = np.rint((fx - critical_lower) / ulp).astype(int)
    axes[2].plot(ulp_index[fc == 1], ft[fc == 1], "o", label="locked")
    axes[2].plot(ulp_index[fc == 0], np.full(np.count_nonzero(fc == 0), 21175), "x", label="unlocked")
    axes[2].set(xlabel="FP64 steps from last locked value", ylabel="transition time", title="Machine-limit interleaving")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.25, which="both")
    fig.savefig(FIGURE, dpi=200)
    plt.close(fig)

    for record in records:
        print(f"{record['quantity']}={record['value']:.17g}")
    print(f"summary={SUMMARY}")
    print(f"figure={FIGURE}")


if __name__ == "__main__":
    main()
