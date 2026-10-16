"""Verify C++ ownership of runtime mechanical parameters."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animate_one_trajectory import load_hysteresis


class EddyCurrentScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_hysteresis()

    def test_radius_and_disk_speed_scaling(self):
        self.assertAlmostEqual(self.module.SolverParams(sr=.5, somega=2).eddy_current_scale, .5)
        self.assertEqual(self.module.SolverParams().eddy_current_scale, 1)
        self.assertEqual(self.module.SolverParams(somega=2).eddy_current_scale, 2)

    def test_derived_parameters_are_readonly(self):
        params = self.module.SolverParams()
        for field in ("alpha", "delta", "gamma_g", "gamma", "gamma_m", "eddy_current_scale"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(params, field, 1)
        with self.assertRaises(TypeError):
            self.module.SolverParams(1, 2, 3, 4, 5)

    def test_invalid_scales(self):
        for value in (0, -1, np.nan, np.inf, 1e200, 1e-200):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.SolverParams(sr=value)

    def test_legacy_reference_values(self):
        p = self.module.SolverParams()
        np.testing.assert_allclose([p.alpha,p.delta,p.gamma_g,p.gamma,p.gamma_m],
            [.342857142857143,8.02199442675159e-5,5.111798459304,17302.6849037817,367982.448040497],
            rtol=1e-13)

    def test_scan_accepts_only_runtime_scales(self):
        zero = np.zeros(1, dtype=self.module.real_dtype)
        status, metrics, states = self.module.scan_initial_grid(zero, zero, .0005, 20, 10, 1, 1, 2, 1)
        self.assertEqual(status.shape, (1, 1))
        self.assertTrue(np.isfinite(states).all())
        with self.assertRaises(TypeError):
            self.module.scan_initial_grid(zero, zero, .0005, 20, 10, 1, 1, 2,
                                          self.module.SolverParams(), 1)

    def test_cpp_push_conversion(self):
        u = self.module.physical_units(sr=1, somega=2)
        expected = .7e-3 / u["ball_mass_kg"] / u["length_unit_m"] / u["disk_omega_rad_s"]**2
        self.assertAlmostEqual(self.module.push_acceleration(.7,1,2), expected)


if __name__ == "__main__":
    unittest.main()
