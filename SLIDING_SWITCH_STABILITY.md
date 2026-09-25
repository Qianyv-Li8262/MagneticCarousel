# 8-magnet exact-field contact-mode step study

The trajectory figures and table immediately below are the baseline before the
release-stage direction correction. The correction and its focused local-order
check are recorded at the end of this note.

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

## Release-stage direction correction

The release substep now uses the normalized mean of the event and trial-end
driving-force directions for its first RK2 stage, then the driving-force
direction computed at the RK midpoint for its second stage. This changes only
the rolling-to-sliding substep.

A focused FP64 exact-field check located the rolling threshold at
`t=0.0465486590613`. For each `h`, it started a rolling state at `t_event-h/2`,
advanced exactly one release step of length `h`, and compared the resulting
contact slip against its continuous-time leading term
`(c/2)(h/2)^2 e_event`, with `c=d(A-b)/dt=39.4906717194` on the rolling path.

| `h` | Baseline slip residual | Corrected slip residual |
| ---: | ---: | ---: |
| 0.001 | 1.83997e-6 | 5.39484e-9 |
| 0.0005 | 4.58325e-7 | 6.74157e-10 |
| 0.00025 | 1.14372e-7 | 8.42572e-11 |
| 0.000125 | 2.85670e-8 | 1.05314e-11 |
| 0.0000625 | 7.13848e-9 | 1.31637e-12 |

Halving `h` reduces the baseline residual by about 4 and the corrected
residual by about 8. Thus the identified release-step slip error changed from
`O(h^2)` to `O(h^3)` in this nondegenerate case. The full trajectory still shows
approximately first-order step sensitivity after release, because ordinary
sliding steps close to zero slip are unchanged. The corrected release-and-
recapture trajectory has the expected two transitions at every tested step
size; its maximum position difference at `dt=0.004` versus `dt=0.0000625`
is `0.01016` (baseline `0.02886`). The corrected trajectory output is under
`runs/sliding-switches-exact8-stage-fix/`.

## Near-zero-slip sliding-step guard

After the release-stage correction, an ordinary sliding step now requests
bisection when `dt * (A + b) > 0.5 * |s|` for nonzero starting slip. The existing
start-of-step capture prediction runs before this check. At the accepted step,
the Euler RK2 midpoint satisfies `|s_mid| >= 0.75 * |s|`; in particular, its
friction direction cannot be determined by an almost cancelled slip vector.
Zero starting slip bypasses this relative-size test and remains subject to the
existing mode handling.

The same FP64 exact-eight-magnet scan completed all seven step sizes with the
expected transitions in all four cases. Its output is under
`runs/sliding-switches-exact8-adaptive-guard/`. For the release-and-recapture
case, the maximum position difference against the finest computed trajectory
changed as follows:

| `dt` | Release-stage correction only | With sliding-step guard |
| ---: | ---: | ---: |
| 0.004 | 1.01557e-2 | 2.02227e-4 |
| 0.002 | 5.34396e-3 | 5.46083e-5 |
| 0.001 | 2.47208e-3 | 1.44312e-5 |
| 0.0005 | 1.22247e-3 | 3.58207e-6 |
| 0.00025 | 5.22529e-4 | 8.58019e-7 |
| 0.000125 | 1.75543e-4 | 1.77749e-7 |

Adjacent-step **final-position** differences with the guard, from `dt=0.004`
downward, are `1.48059e-4`, `4.01844e-5`, `1.08507e-5`, `2.72416e-6`,
`6.80313e-7`, and `1.77749e-7`. Their successive ratios are `3.68`, `3.70`,
`3.98`, `4.00`, and `3.83`, consistent with second-order step sensitivity for
this trajectory. These are self-convergence measurements, not errors against
an analytic solution or proof of global second order for every transition.
The guard can require very small substeps immediately after release and remains
subject to the configured refinement-depth limit.
