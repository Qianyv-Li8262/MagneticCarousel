# SPDX-License-Identifier: AGPL-3.0-only
"""Preserve the configured Windows CUDA host toolchain for ordinary builds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    mode, filename, *command = sys.argv[1:]
    path = Path(filename)
    if mode == "--capture":
        # Automatic CMake regeneration may run outside the preset environment.
        # Retain the working paths captured by the original configure step.
        if not all(shutil.which(tool) for tool in ("cl.exe", "mt.exe")):
            if path.is_file():
                return 0
            raise RuntimeError("Configure with an MSVC/Windows SDK environment first")
        # Only toolchain search paths belong in the build tree, not the full
        # shell environment (which may contain credentials).
        paths = {key: os.environ[key] for key in ("PATH", "INCLUDE", "LIB", "LIBPATH")
                 if key in os.environ}
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(paths, indent=2) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        return 0
    if mode != "--run" or not command:
        raise ValueError("expected --capture FILE or --run FILE COMPILER [ARGS...]")
    environment = os.environ.copy()
    environment.update(json.loads(path.read_text(encoding="utf-8")))
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
