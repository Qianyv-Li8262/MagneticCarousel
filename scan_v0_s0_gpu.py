# SPDX-License-Identifier: AGPL-3.0-only
"""Run the current 2D initial-state scan (GPU by default); use --cpu for CPU."""
import sys
from scan_initial_2d import main

if __name__ == "__main__":
    if "--cpu" not in sys.argv and "--gpu" not in sys.argv:
        sys.argv.append("--gpu")
    main()
