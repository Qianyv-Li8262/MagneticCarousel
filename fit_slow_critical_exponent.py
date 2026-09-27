# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Fit and diagnose the critical slowing in the seven slow_*.log files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from project_paths import RESULTS_DIR, SOMEGA_FP64_LOG_DIR, SR_FP64_LOG_DIR, image_output_path


HERE = Path(__file__).resolve().parent
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?\s+")


def read_log(path: Path):
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    rows = []
    for line in raw.decode(encoding).splitlines():
        if NUMBER.match(line):
            fields = line.split()
            if len(fields) == 3:
                rows.append((float(fields[0]), int(fields[1]), float(fields[2])))
    return np.asarray(rows, dtype=float)


def predictions(model: str, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    d = theta[-1] - x
    if model == "power":
        log_a, p, _ = theta
        return np.exp(log_a) * d ** (-p)
    if model == "shifted_power":
        t0, log_a, p, _ = theta
        return t0 + np.exp(log_a) * d ** (-p)
    if model == "log":
        a, log_b, _ = theta
        return a + np.exp(log_b) * np.log(1.0 / d)
    raise ValueError(model)


def fit(model: str, x: np.ndarray, t: np.ndarray, sc_bounds: tuple[float, float], start=None):
    sc_lo, sc_hi = sc_bounds
    sc0 = 0.5 * (sc_lo + sc_hi)
    dmed = np.median(sc0 - x)
    if model == "power":
        p0 = 0.1
        start = np.array([np.log(np.median(t) * dmed**p0), p0, sc0]) if start is None else start
        lower, upper = [-50, 1e-4, sc_lo], [50, 3, sc_hi]
    elif model == "shifted_power":
        p0, t00 = 0.2, 0.25 * np.min(t)
        start = np.array([t00, np.log((np.median(t) - t00) * dmed**p0), p0, sc0]) if start is None else start
        lower, upper = [0, -50, 1e-4, sc_lo], [0.999 * np.min(t), 50, 3, sc_hi]
    elif model == "log":
        b0 = (np.max(t) - np.min(t)) / max(np.log(np.max(sc0 - x) / np.min(sc0 - x)), 1)
        a0 = np.median(t) - b0 * np.log(1 / dmed)
        start = np.array([a0, np.log(max(b0, 1)), sc0]) if start is None else start
        lower, upper = [-1e6, -20, sc_lo], [1e6, 20, sc_hi]
    else:
        raise ValueError(model)

    def residual(theta):
        with np.errstate(all="ignore"):
            pred = predictions(model, theta, x)
        if np.any(pred <= 0) or not np.all(np.isfinite(pred)):
            return np.full_like(t, 1e10)
        return np.log(pred) - np.log(t)

    result = least_squares(residual, start, bounds=(lower, upper), max_nfev=100000,
                           xtol=1e-14, ftol=1e-14, gtol=1e-14)
    pred = predictions(model, result.x, x)
    resid = np.log(pred) - np.log(t)
    rss = float(resid @ resid)
    n, k = len(t), len(result.x)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    r2 = 1 - rss / float(np.sum((np.log(t) - np.mean(np.log(t))) ** 2))
    return {"theta": result.x, "pred": pred, "rss": rss, "aic": aic, "bic": bic, "r2_log": r2}


def unpack(model: str, result: dict) -> dict:
    q = result["theta"]
    if model == "power":
        return {"t0": 0.0, "A": np.exp(q[0]), "p": q[1], "sc": q[2]}
    if model == "shifted_power":
        return {"t0": q[0], "A": np.exp(q[1]), "p": q[2], "sc": q[3]}
    return {"t0": q[0], "A": np.exp(q[1]), "p": np.nan, "sc": q[2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="slow", help="log prefix, e.g. slow or slowsr")
    args = parser.parse_args()
    primary_log_dir = SR_FP64_LOG_DIR if args.prefix == "slowsr" else SOMEGA_FP64_LOG_DIR
    log_dirs = (primary_log_dir, HERE / "slow-somega-scan", HERE)
    logs = []
    for log_dir in log_dirs:
        logs.extend(sorted(log_dir.glob(f"{args.prefix}_fine*.log")))
        large = log_dir / f"{args.prefix}_large.log"
        if large.exists():
            logs.append(large)
    logs = list(dict.fromkeys(logs))
    logs = [path for path in logs if path.exists()]
    if len(logs) < 2:
        raise FileNotFoundError(f"not enough logs matching {args.prefix}_*.log in {HERE}")
    output_dir = RESULTS_DIR / "critical-slowing"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{args.prefix}_critical_fit.csv"
    out_fig = image_output_path(output_dir / f"{args.prefix}_critical_fit.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    blocks = [(path.name, read_log(path)) for path in logs]
    valid = np.concatenate([rows[(rows[:, 1] == 1) & (rows[:, 2] >= 0)] for _, rows in blocks])
    invalid = np.concatenate([rows[rows[:, 1] == 0] for _, rows in blocks])
    x_all, t_all = valid[:, 0], valid[:, 2]
    last_locked = float(np.max(x_all))
    first_unlocked = float(np.min(invalid[:, 0][invalid[:, 0] > last_locked]))
    eps = 1e-14
    sc_bounds = (last_locked + eps, first_unlocked - eps)
    sc_mid = np.mean(sc_bounds)

    cutoffs = [np.inf] + [10.0 ** (-power) for power in range(1, 14)]
    models = ["power", "shifted_power", "log"]
    rows_out, fitted = [], {}
    for cutoff in cutoffs:
        mask = sc_mid - x_all <= cutoff
        if np.count_nonzero(mask) < 12:
            continue
        for model in models:
            result = fit(model, x_all[mask], t_all[mask], sc_bounds)
            values = unpack(model, result)
            row = {"model": model, "max_distance": cutoff, "n": int(mask.sum()), **values,
                   "aic_log_residual": result["aic"], "bic_log_residual": result["bic"],
                   "r2_log": result["r2_log"]}
            rows_out.append(row)
            fitted[(model, cutoff)] = (mask, result, values)

    eligible = [cutoff for cutoff in cutoffs[1:] if ("power", cutoff) in fitted
                and fitted[("power", cutoff)][0].sum() >= 100]
    primary_cutoff = eligible[-1]
    mask, primary, primary_values = fitted[("power", primary_cutoff)]
    xp, tp = x_all[mask], t_all[mask]
    log_res = np.log(tp) - np.log(primary["pred"])
    rng = np.random.default_rng(20260912)
    boot_p, boot_sc = [], []
    for _ in range(2000):
        synthetic = primary["pred"] * np.exp(rng.choice(log_res, size=len(log_res), replace=True))
        candidate = fit("power", xp, synthetic, sc_bounds, start=primary["theta"])
        vals = unpack("power", candidate)
        boot_p.append(vals["p"])
        boot_sc.append(vals["sc"])
    p_ci = np.percentile(boot_p, [2.5, 50, 97.5])
    sc_ci = np.percentile(boot_sc, [2.5, 50, 97.5])

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    display_cutoff = next(c for c in cutoffs[1:] if ("log", c) in fitted)
    _, log_display, log_values = fitted[("log", display_cutoff)]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(blocks)))
    for (name, block), color in zip(blocks, colors):
        good = (block[:, 1] == 1) & (block[:, 2] >= 0)
        d = log_values["sc"] - block[good, 0]
        axes[0].semilogx(d, block[good, 2], ".", ms=3.5, color=color, alpha=.7, label=name)
    dline = np.geomspace(np.min(log_values["sc"] - x_all) * .8,
                         np.max(log_values["sc"] - x_all) * 1.1, 500)
    theta_log = log_display["theta"]
    axes[0].semilogx(dline, theta_log[0] + np.exp(theta_log[1]) * np.log(1 / dline),
                     "k-", lw=2, label=rf"log fit: $B={np.exp(theta_log[1]):.1f}$")
    axes[0].set(xlabel=r"$s_c-s_\omega$", ylabel="transition time", title="Logarithmic critical slowing")
    axes[0].grid(True, which="both", alpha=.25)
    axes[0].legend(fontsize=7, ncol=2)

    power_rows = [r for r in rows_out if r["model"] == "power" and np.isfinite(r["max_distance"])]
    shift_rows = [r for r in rows_out if r["model"] == "shifted_power" and np.isfinite(r["max_distance"])]
    axes[1].semilogx([r["max_distance"] for r in power_rows], [r["p"] for r in power_rows], "o-", label="pure power")
    axes[1].semilogx([r["max_distance"] for r in shift_rows], [r["p"] for r in shift_rows], "s-", label="shifted power")
    axes[1].invert_xaxis()
    axes[1].set(xlabel="maximum fitted distance", ylabel="effective exponent p", title="Fit-window dependence")
    axes[1].grid(True, which="both", alpha=.25)
    axes[1].legend()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    print(f"points={len(x_all)}; critical_bracket=[{last_locked:.15f}, {first_unlocked:.15f}]")
    print(f"primary cutoff={primary_cutoff:g}; n={len(xp)}")
    print(f"power p={primary_values['p']:.8f}; bootstrap95=[{p_ci[0]:.8f}, {p_ci[2]:.8f}]")
    print(f"power sc={primary_values['sc']:.15f}; bootstrap95=[{sc_ci[0]:.15f}, {sc_ci[2]:.15f}]")
    for cutoff in cutoffs:
        if ("power", cutoff) not in fitted:
            continue
        print(f"\ncutoff={cutoff:g}")
        for model in models:
            _, result, values = fitted[(model, cutoff)]
            print(f"  {model:13s} p={values['p']!s:>12s} sc={values['sc']:.15f} "
                  f"t0={values['t0']:.5g} A={values['A']:.5g} AIC={result['aic']:.3f} R2log={result['r2_log']:.8f}")
    print(f"saved={out_csv}")
    print(f"saved={out_fig}")


if __name__ == "__main__":
    main()
