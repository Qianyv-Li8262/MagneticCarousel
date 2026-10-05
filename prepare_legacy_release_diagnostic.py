"""Create an isolated source snapshot with the original magnetic parameters."""
from pathlib import Path
import json
import re
import shutil

root = Path(__file__).resolve().parent
target = root / 'scratch' / 'legacy-release-source'
target.mkdir(parents=True, exist_ok=True)
shutil.copytree(root / 'kernels', target / 'kernels', dirs_exist_ok=True)
for name in ('CMakeLists.txt', 'CMakeUserPresets.json', 'write_build_config.py',
             'write_cuda_build_config.py', 'animate_one_trajectory.py',
             'project_paths.py', 'build_metadata.py', 'diagnose_release_stop.py'):
    shutil.copy2(root / name, target / name)
path = target / 'kernels' / 'headers' / 'hysteresis_config.hpp'
text = path.read_text(encoding='utf-8')
legacy = dict(Mu_r='2292.6667', Alpha_JA='3.2e-3', C_JA='0.38', K='1.0909091e-3')
for name, value in legacy.items():
    text, count = re.subn(rf'(CONFIG_CONST real_t {name}\s*=\s*)REAL_LITERAL\([^)]*\)',
                         rf'\g<1>REAL_LITERAL({value})', text)
    assert count == 1, (name, count)
path.write_text(text, encoding='utf-8')
(target / 'diagnostic_origin.json').write_text(json.dumps(dict(
    original_source=str(root), magnetic_parameters=legacy,
    unchanged='field geometry, tilt, contact friction and mechanical scaling'), indent=2), encoding='utf-8')
print(target)
