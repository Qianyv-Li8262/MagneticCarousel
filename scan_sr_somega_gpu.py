# SPDX-License-Identifier: AGPL-3.0-only
"""Run the current 2D (sr, somega) scan; edit scan_sr_somega_2d_gpu.py settings."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("scan_sr_somega_2d_gpu.py")), run_name="__main__")
