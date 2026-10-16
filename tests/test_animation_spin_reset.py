"""Verify instantaneous spin resets at exact solver and display boundaries."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animate_one_trajectory import BlowSchedule, parse_spin_reset


class Solver:
    def __init__(self):
        self.calls = []
        self.time = 0.

    def physical_units(self, **kwargs):
        return {'disk_omega_rad_s': 1.}

    def push_acceleration(self, force, sr, somega):
        return force

    def evolve_trajectory(self, state, dt, steps, params, sliding, direction, offset):
        self.calls.append((self.time, float(state[11]), params.reverse_push_acceleration))
        self.time += dt * steps
        result = state.copy()
        result[11] += dt * steps
        return result, sliding, direction


class ResetTests(unittest.TestCase):
    def schedule(self, events, **kwargs):
        module = Solver()
        params = SimpleNamespace(reverse_push_acceleration=0.)
        return module, BlowSchedule(module, params, .1, sr=1, somega=2,
                                   spin_resets=events, **kwargs)

    def test_inside_step_preserves_other_state_and_contact(self):
        module, schedule = self.schedule([(.25, -5)])
        initial = np.arange(15, dtype=float)
        direction = object()
        state, sliding, returned = schedule.evolve(initial, 10, True, direction)
        self.assertAlmostEqual(state[11], -4.25)
        np.testing.assert_array_equal(np.delete(state, 11), np.delete(initial, 11))
        self.assertEqual(initial[11], 11.)
        self.assertTrue(sliding)
        self.assertIs(returned, direction)
        self.assertTrue(any(abs(t-.25)<1e-12 and oz==-5 for t, oz, _ in module.calls))

    def test_zero_endpoints_and_no_repetition(self):
        _, schedule = self.schedule([(0., 2.), (.5, -3.)])
        state, _, _ = schedule.evolve(np.zeros(15), 5)
        self.assertEqual(state[11], -3.)
        state, _, _ = schedule.evolve(state, 5)
        self.assertAlmostEqual(state[11], -2.5)
        self.assertEqual(len(schedule.reset_samples), 2)

    def test_reset_during_blow_and_preevolution(self):
        module, schedule = self.schedule([(.35, -2), (.2, -5)], start=.2, duration=.3)
        state, _, _ = schedule.evolve(np.zeros(15), 10)
        self.assertAlmostEqual(state[11], -1.35)
        self.assertEqual([t for t, _, _ in schedule.reset_samples], [.2, .35])
        self.assertTrue(any(abs(t-.2)<1e-12 and oz==-5 and force==.7
                            for t, oz, force in module.calls))
        self.assertEqual(schedule.params.reverse_push_acceleration, 0.)

    def test_parser(self):
        self.assertEqual(parse_spin_reset('2000,-5'), (2000., -5.))
        for value in ('2000', 'a,-5', '-1,2', 'nan,2', '0,inf', '0,1,2'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_spin_reset(value)


if __name__ == '__main__':
    unittest.main()
