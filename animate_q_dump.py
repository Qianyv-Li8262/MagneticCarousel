# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Interactively replay q and ReLU-branch behavior from a trajectory dump directory."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd


STATE_COLUMNS = {"step", "time", "qx", "qy", "phase_rate", "last_branch"}
CHANGE_COLUMNS = {"step", "step_time", "qx", "qy", "new_branch"}
BRANCH_COLORS = {0: "tab:blue", 1: "tab:red"}
BRANCH_NAMES = {0: "ReLU 0", 1: "ReLU x"}


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")


def read_dump(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_path = input_dir / "dump_states.csv"
    change_path = input_dir / "branch_changes.csv"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing dump state file: {state_path}")
    if not change_path.is_file():
        raise FileNotFoundError(f"missing branch-change file: {change_path}")

    states = pd.read_csv(state_path)
    changes = pd.read_csv(change_path)
    require_columns(states, STATE_COLUMNS, state_path)
    require_columns(changes, CHANGE_COLUMNS, change_path)
    if states.empty:
        raise ValueError(f"{state_path.name} contains no rows")

    states = states.sort_values(["time", "step"], kind="stable").reset_index(drop=True)
    if not np.all(np.isfinite(states[["time", "qx", "qy", "phase_rate"]].to_numpy(dtype=float))):
        raise ValueError(f"{state_path.name} contains non-finite time, q, or phase-rate data")
    if len(states) > 1 and np.any(np.diff(states["time"].to_numpy(dtype=float)) <= 0.0):
        raise ValueError(f"{state_path.name} times must be strictly increasing")

    changes = changes.sort_values(["step_time", "step"], kind="stable").reset_index(drop=True)
    if not changes.empty:
        if not np.all(np.isfinite(changes[["step_time", "qx", "qy"]].to_numpy(dtype=float))):
            raise ValueError(f"{change_path.name} contains non-finite time or q data")
        invalid_branch = ~changes["new_branch"].isin((0, 1))
        if invalid_branch.any():
            raise ValueError(f"{change_path.name} contains a branch other than 0 or 1")

    state_branch = states["last_branch"].where(states["last_branch"].isin((0, 1)), np.nan)
    state_branch = state_branch.ffill().bfill()
    if state_branch.isna().all():
        if changes.empty:
            raise ValueError("no valid ReLU branch is present in either CSV file")
        state_branch[:] = int(changes.iloc[0]["new_branch"])
    states = states.copy()
    states["branch"] = state_branch.astype(np.int8)

    state_times = states["time"].to_numpy(dtype=float)
    state_phase_rate = states["phase_rate"].to_numpy(dtype=float)
    state_frames = pd.DataFrame(
        {
            "time": state_times,
            "step": states["step"].to_numpy(dtype=np.int64),
            "qx": states["qx"].to_numpy(dtype=float),
            "qy": states["qy"].to_numpy(dtype=float),
            "phase_rate": state_phase_rate,
            "branch": states["branch"].to_numpy(dtype=np.int8),
            "is_change": False,
            "source_order": 1,
            "order": np.arange(len(states), dtype=np.int64),
        }
    )

    if changes.empty:
        frames = state_frames
    else:
        change_times = changes["step_time"].to_numpy(dtype=float)
        change_frames = pd.DataFrame(
            {
                "time": change_times,
                "step": changes["step"].to_numpy(dtype=np.int64),
                "qx": changes["qx"].to_numpy(dtype=float),
                "qy": changes["qy"].to_numpy(dtype=float),
                "phase_rate": np.interp(change_times, state_times, state_phase_rate),
                "branch": changes["new_branch"].to_numpy(dtype=np.int8),
                "is_change": True,
                "source_order": 0,
                "order": np.arange(len(changes), dtype=np.int64),
            }
        )
        frames = pd.concat((state_frames, change_frames), ignore_index=True)
        frames = frames.sort_values(["time", "source_order", "order"], kind="stable").reset_index(drop=True)

    return states, changes, frames


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    padding = 0.08 * span if span > 1.0e-15 else max(0.1, 0.08 * abs(maximum))
    return minimum - padding, maximum + padding


def build_player(
    states: pd.DataFrame,
    changes: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    interval_ms: int,
    initial_speed: float,
    trail_points: int,
    loop: bool,
):
    times = frames["time"].to_numpy(dtype=float)
    steps = frames["step"].to_numpy(dtype=np.int64)
    qx = frames["qx"].to_numpy(dtype=float)
    qy = frames["qy"].to_numpy(dtype=float)
    phase_rate = frames["phase_rate"].to_numpy(dtype=float)
    branches = frames["branch"].to_numpy(dtype=np.int8)
    is_change = frames["is_change"].to_numpy(dtype=bool)

    state_time = states["time"].to_numpy(dtype=float)
    state_phase_rate = states["phase_rate"].to_numpy(dtype=float)

    figure = plt.figure(figsize=(11.5, 8.5))
    grid = figure.add_gridspec(
        2, 1, height_ratios=(2.2, 1.0), left=0.09, right=0.97, top=0.92, bottom=0.23, hspace=0.28
    )
    q_axis = figure.add_subplot(grid[0])
    phase_axis = figure.add_subplot(grid[1])

    q_axis.plot(qx, qy, color="0.72", linewidth=0.9, alpha=0.65, label="complete q path")
    if not changes.empty:
        q_axis.scatter(
            changes["qx"],
            changes["qy"],
            c=changes["new_branch"],
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=14,
            alpha=0.35,
            linewidths=0,
            label="branch-change locations",
        )
    (trail,) = q_axis.plot([], [], color="0.15", linewidth=1.8, alpha=0.85)
    vector = q_axis.quiver(
        [0.0],
        [0.0],
        [qx[0]],
        [qy[0]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.008,
        color=BRANCH_COLORS[int(branches[0])],
        zorder=4,
    )
    point = q_axis.scatter(
        [qx[0]],
        [qy[0]],
        s=180,
        color=BRANCH_COLORS[int(branches[0])],
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )
    q_axis.axhline(0.0, color="0.82", linewidth=0.8)
    q_axis.axvline(0.0, color="0.82", linewidth=0.8)
    q_radius = max(float(np.max(np.hypot(qx, qy))) * 1.08, np.finfo(float).eps)
    q_axis.set_xlim(-q_radius, q_radius)
    q_axis.set_ylim(-q_radius, q_radius)
    q_axis.set_aspect("equal", adjustable="box")
    q_axis.set_xlabel(r"$q_x$")
    q_axis.set_ylabel(r"$q_y$")
    q_axis.grid(alpha=0.2)
    q_axis.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="", markersize=11, color=BRANCH_COLORS[1], label="ReLU x"),
            Line2D([], [], marker="o", linestyle="", markersize=11, color=BRANCH_COLORS[0], label="ReLU 0"),
            Line2D([], [], color="0.72", linewidth=1.2, label="complete q path"),
        ),
        loc="best",
    )

    phase_axis.plot(state_time, state_phase_rate, color="tab:purple", linewidth=1.2)
    time_line = phase_axis.axvline(times[0], color="tab:orange", linewidth=2.0)
    phase_point = phase_axis.scatter([times[0]], [phase_rate[0]], s=45, color="tab:orange", zorder=3)
    phase_axis.set_xlim(float(state_time[0]), float(state_time[-1]))
    phase_axis.set_ylim(*padded_limits(state_phase_rate))
    phase_axis.set_xlabel(r"grid time $t=\mathrm{step}\,\Delta t$")
    phase_axis.set_ylabel(r"$\dot{\phi}=1-v-\omega$")
    phase_axis.grid(alpha=0.25)

    title = figure.suptitle("")
    speed_axis = figure.add_axes((0.13, 0.09, 0.29, 0.035))
    button_axis = figure.add_axes((0.455, 0.082, 0.065, 0.05))
    time_axis = figure.add_axes((0.60, 0.09, 0.30, 0.035))
    speed_slider = Slider(speed_axis, "speed", 0.1, 20.0, valinit=initial_speed, valstep=0.1)
    time_slider = Slider(time_axis, "time", float(times[0]), float(times[-1]), valinit=float(times[0]))
    play_button = Button(button_axis, "Pause")

    playback = {
        "index": 0,
        "playing": True,
        "fraction": 0.0,
        "updating_slider": False,
    }

    def render(index: int, *, synchronize_slider: bool = True):
        index = int(np.clip(index, 0, len(times) - 1))
        playback["index"] = index
        branch = int(branches[index])
        color = BRANCH_COLORS[branch]
        start = max(0, index - trail_points + 1)
        trail.set_data(qx[start : index + 1], qy[start : index + 1])
        vector.set_UVC([qx[index]], [qy[index]])
        vector.set_color(color)
        point.set_offsets([[qx[index], qy[index]]])
        point.set_facecolor(color)
        time_line.set_xdata([times[index], times[index]])
        phase_point.set_offsets([[times[index], phase_rate[index]]])
        marker = " • branch change" if is_change[index] else ""
        title.set_text(
            f"step={steps[index]:,}    t={times[index]:.9g}    {BRANCH_NAMES[branch]}{marker}"
        )
        if synchronize_slider:
            playback["updating_slider"] = True
            time_slider.set_val(float(times[index]))
            playback["updating_slider"] = False
        figure.canvas.draw_idle()
        return trail, vector, point, time_line, phase_point, title

    def tick(_frame_number: int):
        if playback["playing"]:
            playback["fraction"] += float(speed_slider.val)
            advance = int(playback["fraction"])
            playback["fraction"] -= advance
            if advance:
                next_index = playback["index"] + advance
                if next_index >= len(times):
                    if loop:
                        next_index %= len(times)
                    else:
                        next_index = len(times) - 1
                        playback["playing"] = False
                        play_button.label.set_text("Play")
                render(next_index)
        return trail, vector, point, time_line, phase_point, title

    def toggle_play(_event) -> None:
        playback["playing"] = not playback["playing"]
        playback["fraction"] = 0.0
        play_button.label.set_text("Pause" if playback["playing"] else "Play")
        figure.canvas.draw_idle()

    def seek(value: float) -> None:
        if playback["updating_slider"]:
            return
        index = int(np.searchsorted(times, value, side="left"))
        if index > 0 and (index == len(times) or abs(times[index - 1] - value) <= abs(times[index] - value)):
            index -= 1
        render(index, synchronize_slider=False)

    def on_key(event) -> None:
        if event.key == " ":
            toggle_play(event)
        elif event.key == "left":
            playback["playing"] = False
            play_button.label.set_text("Play")
            render(playback["index"] - 1)
        elif event.key == "right":
            playback["playing"] = False
            play_button.label.set_text("Play")
            render(playback["index"] + 1)

    play_button.on_clicked(toggle_play)
    time_slider.on_changed(seek)
    figure.canvas.mpl_connect("key_press_event", on_key)
    render(0)
    animation = FuncAnimation(
        figure,
        tick,
        frames=itertools.count(),
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    return figure, animation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory produced by dump_one_trajectory.py")
    parser.add_argument("--interval-ms", type=int, default=40, help="GUI timer interval; default 40 ms")
    parser.add_argument("--speed", type=float, default=1.0, help="initial samples advanced per timer tick")
    parser.add_argument("--trail-points", type=int, default=500, help="number of recent q points highlighted")
    parser.add_argument("--loop", action="store_true", help="restart automatically after the final frame")
    args = parser.parse_args()

    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if not np.isfinite(args.speed) or not 0.1 <= args.speed <= 20.0:
        parser.error("--speed must be between 0.1 and 20")
    if args.trail_points <= 0:
        parser.error("--trail-points must be positive")
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")

    states, changes, frames = read_dump(input_dir)
    figure, animation = build_player(
        states,
        changes,
        frames,
        interval_ms=args.interval_ms,
        initial_speed=args.speed,
        trail_points=args.trail_points,
        loop=args.loop,
    )
    # Keep both objects alive until the interactive window closes.
    figure._q_dump_animation = animation
    plt.show()


if __name__ == "__main__":
    main()
