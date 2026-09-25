# 8-magnet exact-field contact-mode step study

Run on 2026-09-25 with the FP64 2D CPU integrator, `IYPT16_MAGNET_NUMBER=8`,
`IYPT16_FIELD_MODE=exact`, and `sr=0.428571`, `somega=0.1`. The integrator
time interval is `[0, 1]`. All cases start at `(x,y)=(4,0)` with identity
attitude and zero translational velocity. The other state components are zero
except those shown below. The script [check_sliding_switches_exact8.py](check_sliding_switches_exact8.py)
rebuilds neither the module nor its inputs; its docstring gives the dedicated
build commands and it checks module metadata before running.

| Case | Initial state change | Expected transition sequence |
| --- | --- | --- |
| Forward capture | Contact slip `(0,+1)` via `omega_x=1/alpha` | sliding → rolling |
| Backward capture | Contact slip `(0,-1)` via `omega_x=-1/alpha` | sliding → rolling |
| Radial capture | Contact slip `(+1,0)` via `omega_y=-1/alpha` | sliding → rolling |
| Release and recapture | `q_y=-0.1`, zero slip | rolling → sliding → rolling |

The nonzero `q_y` is a deliberately prepared hysteresis state for a state-machine
stress test; this experiment does not establish that it is reached by the usual
zero-`q` startup. Switch times are recorded at the first completed step in the
new mode, with uncertainty up to one `dt`.

| Case | `dt=0.004` switch times | `dt=0.0000625` switch times | Max position difference at `dt=0.004` vs finest |
| --- | --- | --- | ---: |
| Forward capture | 0.052 | 0.049375 | 0.0000567 |
| Backward capture | 0.044 | 0.0428125 | 0.0001015 |
| Radial capture | 0.048 | 0.047125 | 0.0000456 |
| Release and recapture | 0.048, 0.344 | 0.0465625, 0.325625 | 0.02886 |

All seven step sizes (`0.004` through `0.0000625`, halved each time) completed
with finite states and the expected transition sequence. Their maximum position
differences on a shared time grid decreased monotonically toward the finest
trajectory. The three capture cases have small trajectory differences even at
`dt=0.004`. Release and recapture is much more step-sensitive: the maximum
position difference versus the finest run is `0.00744` at `dt=0.001`, `0.00152`
at `dt=0.00025`, and `0.000511` at `dt=0.000125`. These are comparisons to a
finer numerical solution, not exact-solution error bounds. The release event and
following slip excursion need smaller `dt` for quantitative timing and position
estimates.

The script writes per-step CSVs, plots, and a machine-readable `summary.json`
under `runs/sliding-switches-exact8/`.
