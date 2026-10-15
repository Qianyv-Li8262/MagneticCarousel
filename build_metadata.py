# SPDX-License-Identifier: AGPL-3.0-only
"""Prominent informational summary of the verified build used by a run."""

import math
import os
import sys


def print_build_metadata(config: dict, *, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    parameters = config.get("field_parameters", {})
    models = {"point": "Point magnetic dipoles",
              "cylinder": "Uniformly axially magnetized solid cylinders"}
    modes = {"exact": "Exact point-dipole field", "lut": "Fourier-spline LUT"}
    unknown = "UNKNOWN (rebuild the build manifest)"
    rows = [
        ("Physical model", models.get(config.get("magnet_model"), unknown)),
        ("Magnet count", config.get("magnet_number", parameters.get("magnet_number", unknown))),
        ("Field calculation", modes.get(config.get("field_mode"), unknown)),
        ("Height", f"{parameters['height']} (normalized, magnet centre plane)"
         if "height" in parameters else unknown),
    ]
    tilt = config.get("inclined_disk", {})
    rolling = config.get("model_parameters", {}).get("kEnableRollingResistance")
    rows.append(("Rolling resistance", "Enabled" if rolling == "true" else "Disabled" if rolling == "false" else unknown))
    enabled = tilt.get("enabled")
    rows.append(("Inclined disk", "Enabled" if enabled is True else "Disabled" if enabled is False else unknown))
    if enabled is True:
        # The downhill force points 90 degrees clockwise from the positive tilt axis.
        # A negative tilt reverses that direction. Report the force, not the axis.
        angle = math.degrees(tilt["angle_rad"])
        direction = round(math.degrees(tilt["axis_azimuth_rad"]) - 90.0 +
                          (180.0 if angle < 0.0 else 0.0), 10) % 360.0
        rows.extend([
            ("Tilt force direction", f"{direction:g} deg (from +x, counterclockwise)"),
            ("Tilt angle", f"{angle:g} deg"),
        ])
    label_width = max(18, max(len(label) for label, _ in rows))
    lines = ["CURRENT COMPILED BUILD"] + [f"{label:<{label_width}}: {value}" for label, value in rows]
    width = max(map(len, lines)) + 4
    colour = ("NO_COLOR" not in os.environ
              and (bool(os.environ.get("FORCE_COLOR"))
                   or (stream.isatty() and os.environ.get("TERM") != "dumb")))
    # Informational blue background and bright white text, not warning colours.
    prefix, suffix = ("\033[1;97;48;2;40;100;200m", "\033[0m") if colour else ("", "")
    print(file=stream)
    for line in ["="*width, *[f"  {line:<{width-4}}  " for line in lines], "="*width]:
        print(f"{prefix}{line}{suffix}", file=stream)
    stream.flush()
