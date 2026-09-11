"""Run with the FP64 extension directory on PYTHONPATH.

The framewise reference is test-only; production evolution stays in C++.
"""
import unittest

import hysteresis
import numpy as np

from analyze_critical_slowing_fp64 import detect_outcome


def reference(v0, dt, frames, params):
    ball = np.array([0, 0, v0, 0, -v0 / params.alpha, 0, 0, 0], dtype=np.float64)
    times, speeds, rates, slips = [], [], [], []
    minimum_index = -1
    locked = 0
    smooth_entries = []
    steps = hysteresis.critical_slow_confirm_evolve_steps
    for i in range(frames):
        old_x = ball[1]
        ball, _ = hysteresis.evolve_with_mode(ball, dt, steps, params)
        speed = (ball[1] - old_x) / (steps * dt)
        target = -params.alpha / (1 - params.alpha)
        locked = locked + 1 if target - 1e-3 < speed < target + 1e-3 else 0
        # Independent offline weighted-window reference at every frame.
        all_speeds = np.asarray(speeds + [float(speed)])
        end = (i + 1) * steps * dt
        if end >= 1000:
            ends = np.arange(1, i + 2) * steps * dt
            weights = np.maximum(0, ends - np.maximum(ends - steps * dt, end - 1000))
            smooth = float(weights @ all_speeds / 1000)
            if abs(smooth - target) < 1e-3:
                smooth_entries.append(end - 500)
            else:
                smooth_entries.clear()
        if locked >= hysteresis.critical_slow_lock_confirm_window and smooth_entries:
            return 1, smooth_entries[0]
        times.append(float(ball[0]))
        speeds.append(float(speed))
        rates.append(float(1 - ball[2] - ball[4]))
        slips.append(float(ball[2] + params.alpha * ball[4]))
        if minimum_index < 0 or speed < speeds[minimum_index]:
            minimum_index = i
        detection = detect_outcome(times, speeds, rates, slips, minimum_index,
                                   -params.alpha / (1 - params.alpha))
        if detection is not None and detection.outcome == 'unlocked':
            return 2, detection.event_time
    return 0, -1.0


class CoreTests(unittest.TestCase):
    def test_dtype(self):
        self.assertEqual(hysteresis.real_dtype, np.dtype('float64'))

    def test_confirmation_and_direct_v0(self):
        # No magnetic force or drag: exactly stationary/constant-speed rolling.
        params = hysteresis.SolverParams(0.25, 0, 1, 0, 0)
        window = hysteresis.critical_slow_lock_confirm_window
        self.assertEqual(hysteresis.evolve_critical_slow(-1 / 3, 0.001, window - 1, params), (0, -1.0))
        result = hysteresis.evolve_critical_slow(-1 / 3, 0.001, window, params)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 500.0)
        self.assertEqual(hysteresis.evolve_critical_slow(0, 0.001, window, params), (0, -1.0))

    def test_full_physical_smoothing_window(self):
        params = hysteresis.SolverParams(0.25, 0, 1, 0, 0)
        # Raw confirmation takes only 250 time units here; do not use a partial mean.
        self.assertEqual(hysteresis.evolve_critical_slow(-1/3, .0001, 199, params), (0, -1.0))
        self.assertEqual(hysteresis.evolve_critical_slow(-1/3, .0001, 200, params), (1, 500.0))
        # 35-unit frames require a fractional oldest frame in the 1000-unit mean.
        actual = hysteresis.evolve_critical_slow(-1/3, .0007, 50, params)
        self.assertEqual(actual, reference(-1/3, .0007, 50, params))
        self.assertEqual(actual, (1, 515.0))

    def test_rising_reference(self):
        # Weak drag, ample normal force: rolling throughout, so the old evolve
        # entry's mode reinitialization does not alter the reference dynamics.
        params = hysteresis.SolverParams(0.25, 0.01, 100, 0, 0)
        frames = hysteresis.critical_slow_confirm_window + 10
        expected = reference(-0.1, 0.001, frames, params)
        frame_time = hysteresis.critical_slow_confirm_evolve_steps * 0.001
        self.assertEqual(expected[0], 2)
        self.assertAlmostEqual(expected[1], frame_time, places=7)
        self.assertEqual(hysteresis.evolve_critical_slow(-0.1, 0.001, frames, params), expected)

    def test_analysis_script_physical_events(self):
        params = hysteresis.SolverParams(0.33333333 * 0.8, 1.7212357e-4 / 0.8,
                                        0.16560356 / 0.6**2, 489.39639 / 0.6**2,
                                        11011.419 / 0.8**2 / 0.6**2)
        for v0, classification in [(0.189186667, 1), (0.189213333, 2)]:
            with self.subTest(v0=v0):
                expected = reference(v0, 0.001, 1600, params)
                actual = hysteresis.evolve_critical_slow(v0, 0.001, 1600, params)
                self.assertEqual(expected[0], classification)
                self.assertEqual(actual[0], expected[0])
                self.assertAlmostEqual(actual[1], expected[1], places=5)

    def test_errors_and_int64_budget(self):
        params = hysteresis.SolverParams(0.25, 0, -1, 0, 0)
        self.assertEqual(hysteresis.evolve_critical_slow(0.1, 0.001, 2**31 + 1, params), (-1, -1.0))
        for v0, dt, frames in [(float('nan'), 0.001, 1), (0.1, 0, 1), (0.1, 0.001, -1)]:
            with self.assertRaises(ValueError):
                hysteresis.evolve_critical_slow(v0, dt, frames, params)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.params = hysteresis.SolverParams(0.25, 0.01, 100, 0, 0)

    def test_thread_counts_and_order(self):
        velocities = np.array([0, -0.1, -1/3, 0.2, -0.2, 0, -0.1], dtype=np.float64)
        velocities.flags.writeable = False
        expected = [hysteresis.evolve_critical_slow(v, 0.001, 110, self.params) for v in velocities]
        for threads in (1, 2, 4, 20):
            answer = np.full(velocities.size, 99, dtype=np.int32)
            times = np.full(velocities.size, np.nan)
            result = hysteresis.calc_slow_cpu(answer, times, velocities, 0.001, 110, self.params, threads)
            self.assertIsNone(result)
            np.testing.assert_array_equal(answer, [r[0] for r in expected])
            np.testing.assert_array_equal(times, [r[1] for r in expected])

    def test_physical_parameters(self):
        params = hysteresis.SolverParams(0.33333333 * 0.8, 1.7212357e-4 / 0.8,
                                        0.16560356 / 0.6**2, 489.39639 / 0.6**2,
                                        11011.419 / 0.8**2 / 0.6**2)
        velocities = np.array([0.18923, -0.36, 0.18922, 1.0])
        expected = [hysteresis.evolve_critical_slow(v, 0.001, 150, params) for v in velocities]
        answer, times = np.empty(4, dtype=np.int32), np.empty(4)
        hysteresis.calc_slow_cpu(answer, times, velocities, 0.001, 150, params, 3)
        np.testing.assert_array_equal(answer, [r[0] for r in expected])
        np.testing.assert_array_equal(times, [r[1] for r in expected])

    def test_empty_and_failure_results(self):
        hysteresis.calc_slow_cpu(np.empty(0, dtype=np.int32), np.empty(0), np.empty(0),
                                 0.001, 0, self.params, 4)
        answer, times = np.empty(3, dtype=np.int32), np.empty(3)
        params = hysteresis.SolverParams(0.25, 0, -1, 0, 0)
        hysteresis.calc_slow_cpu(answer, times, np.zeros(3), 0.001, 2**31 + 1, params, 2)
        np.testing.assert_array_equal(answer, [-1]*3)
        np.testing.assert_array_equal(times, [-1.0]*3)

    def test_invalid_buffers_and_arguments(self):
        answer, times, v0 = np.full(3, 99, dtype=np.int32), np.full(3, -99.0), np.zeros(3)
        def call(a=answer, t=times, v=v0, threads=2, dt=0.001):
            hysteresis.calc_slow_cpu(a, t, v, dt, 1, self.params, threads)
        for args in ({'a': answer.astype(np.int64)}, {'v': v0.astype(np.float32)},
                     {'t': np.zeros(6)[::2]}, {'v': [0., 0., 0.]},
                     {'a': answer[:2]}, {'v': v0.reshape(1, 3)}, {'threads': 0},
                     {'threads': -1}, {'dt': float('nan')}, {'v': np.array([0., np.nan, 0.])},
                     {'t': v0}, {'a': times.view(np.int32)[:3]}):
            with self.assertRaises((TypeError, ValueError)):
                call(**args)
        times.flags.writeable = False
        with self.assertRaises(ValueError):
            call()
        np.testing.assert_array_equal(answer, [99]*3)
        np.testing.assert_array_equal(times, [-99.0]*3)


if __name__ == '__main__':
    unittest.main()
