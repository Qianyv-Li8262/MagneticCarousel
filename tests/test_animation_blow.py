"""Check mN conversion and 2D-time force switches inside solver batches."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animate_one_trajectory import BlowSchedule, validate_blow


class RecordingSolver:
    magnet_number = 8

    def __init__(self):
        self.calls = []

    def evolve_trajectory(self, state, dt, steps, params, sliding, direction, offset):
        self.calls.append((dt * steps, params.reverse_push_acceleration, sliding, direction))
        return state + dt * steps * params.reverse_push_acceleration, True, direction


class BlowTests(unittest.TestCase):
    def test_switches_inside_one_batch_and_preserves_contact(self):
        module = RecordingSolver()
        params = SimpleNamespace(reverse_push_acceleration=0.)
        schedule = BlowSchedule(module, params, .1, sr=1, somega=2,
                                start=.013, duration=.019)
        direction = object()
        state, mode, returned_direction = schedule.evolve(0., 10, False, direction)
        expected = .7e-3 / (7850 * 4 * np.pi / 3 * .006**3) / .0175 / schedule.omega**2
        self.assertAlmostEqual(schedule.acceleration, expected)
        self.assertAlmostEqual(state, expected * .019)
        self.assertAlmostEqual(sum(call[0] for call in module.calls), 1.)
        self.assertTrue(mode)
        self.assertIs(returned_direction, direction)
        self.assertEqual(params.reverse_push_acceleration, 0.)
        self.assertEqual(module.calls[0][1], 0.)
        self.assertEqual(module.calls[-1][1], 0.)

    def test_preevolution_crosses_both_switches(self):
        module = RecordingSolver()
        params = SimpleNamespace(reverse_push_acceleration=0.)
        schedule = BlowSchedule(module, params, .1, sr=1, somega=1,
                                start=0, duration=.01, force_mn=1.4)
        state, mode, direction = schedule.evolve(0., 5)
        before = state
        state, _, _ = schedule.evolve(state, 5, mode, direction)
        self.assertEqual(state, before)
        self.assertAlmostEqual(before, schedule.acceleration * .01)

    def test_disabled_schedule_keeps_original_steps(self):
        module = RecordingSolver()
        schedule = BlowSchedule(module, SimpleNamespace(reverse_push_acceleration=0.),
                                .0005, sr=1, somega=2)
        for _ in range(30):
            schedule.evolve(0., 700)
        self.assertEqual(len(module.calls), 30)
        self.assertTrue(all(dt == 700 * .0005 and force == 0 for dt, force, _, _ in module.calls))

    def test_invalid_arguments(self):
        for args in [(0, None, .7), (None, 1, .7), (-1, 1, .7),
                     (0, 0, .7), (0, float('nan'), .7), (0, 1, -1),
                     (0, 1, float('inf'))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_blow(*args)


if __name__ == '__main__':
    unittest.main()
