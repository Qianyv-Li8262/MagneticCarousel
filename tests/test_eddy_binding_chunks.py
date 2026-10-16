"""Run in a separate process to isolate the enabled pybind type registry."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


def check_binding(binary):
    spec = importlib.util.spec_from_file_location("hysteresis_eddy_enabled_test", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import animate_one_trajectory as animation

    assert module.eddy_currents_enabled
    assert module.eddy_mode_count == 12 and module.state_size == 51
    params = module.SolverParams(sr=1, somega=2)
    state = animation.make_initial_state(module, alpha=params.alpha, v0=-.05)
    state[0] = .58
    dt = .0005
    held = module.magnetize_held_ball(state, dt, 40, params)
    held_chunk = module.magnetize_held_ball(state, dt, 20, params)
    held_chunk = module.magnetize_held_ball(held_chunk, dt, 20, params)
    np.testing.assert_allclose(held, held_chunk, rtol=1e-13, atol=1e-13)
    assert np.any(held[15:] != 0)
    legacy = module.magnetize_held_ball(state[:15], dt, 40, params)
    np.testing.assert_array_equal(legacy, held)

    # Preserve the prepared magnetic histories when restoring release velocities.
    held[3:5], held[9:12] = state[3:5], state[9:12]
    full, _, _ = module.evolve_trajectory(held, dt, 40, params)
    chunk, sliding, direction = module.evolve_trajectory(held, dt, 20, params)
    chunk, _, _ = module.evolve_trajectory(chunk, dt, 20, params, sliding, direction, 20)
    np.testing.assert_allclose(full, chunk, rtol=1e-13, atol=1e-13)
    assert np.isfinite(full).all() and np.any(full[15:] != 0)
    assert np.isfinite(module.magnetic_power(full, params)).all()
    z = np.zeros(1, dtype=module.real_dtype)
    status, _, scanned = module.scan_initial_grid(z, z, dt, 20, 10, 1, 1, 2, 1)
    assert np.all(status == 0) and np.any(scanned[..., 15:] != 0)
    assert scanned.shape[-1] == 51 and np.isfinite(scanned).all()
    print("PASS: enabled binding, held history, legacy input, chunk persistence and scan output")


if __name__ == "__main__":
    check_binding(sys.argv[1])
