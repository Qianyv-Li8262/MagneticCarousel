# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility launcher for the current initial-state scan with a fixed S0 line."""
import sys
from scan_initial_2d import main

if __name__ == "__main__":
    if "--cpu" not in sys.argv and "--gpu" not in sys.argv:
        sys.argv.append("--gpu")
    if not any(arg.startswith("--s0-lower") for arg in sys.argv):
        sys.argv.extend(("--s0-lower", "0"))
    if not any(arg.startswith("--s0-upper") for arg in sys.argv):
        sys.argv.extend(("--s0-upper", "0"))
    main()
