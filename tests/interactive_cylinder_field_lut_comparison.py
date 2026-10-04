# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262
"""Interactive finite-cylinder reference versus Fourier-spline comparison.

Like the point tool, this independently constructs SciPy splines; it does not
load a production binary LUT. Beta is factored out of both fields. Lengths are
in the same normalized units as the C++ generator, and thickness is full length.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.special import jv

from interactive_field_lut_comparison import (
    COMPONENTS, InteractiveComparison, Settings, _lut_components,
)


@dataclass(frozen=True)
class CylinderSettings(Settings):
    modes: int = 5
    nodes: int = 204
    height: float = 1.610066235
    magnet_radius: float = 0.5
    thickness: float = 0.5
    reference_order: int = 16
    reference_angles: int = 64


def _spectral_samples(settings: CylinderSettings, rho: np.ndarray, refinement=1):
    """Finite-radius and finite-thickness Hankel integral, independent of reference."""
    R = settings.magnets / 2
    gap = settings.height - settings.thickness / 2
    cutoff = 80 / gap
    width = 12 / (settings.upper_ratio*R + R + settings.magnet_radius + gap) / refinement
    panels = int(np.ceil(cutoff / width))
    if panels > 100000:
        raise ValueError("Spectral integral needs too many panels; check the height/thickness gap")
    gx, gw = leggauss(32)
    panel_width = cutoff / panels
    k = ((np.arange(panels)[:, None] + .5 + gx/2)*panel_width).ravel()
    weight = np.tile(gw*panel_width/2, panels)
    form = 2*jv(1, k*settings.magnet_radius)/(k*settings.magnet_radius)
    vertical = np.exp(-k*gap)*(-np.expm1(-k*settings.thickness))/settings.thickness
    weight *= 2*settings.magnets*form*vertical
    # Avoid allocating an unbounded radial-node x quadrature-node matrix.
    samples = np.empty((settings.modes, 4, rho.size))
    for mode in range(settings.modes):
        m = (2*mode+1)*settings.magnets//2
        w = weight*jv(m, k*R)
        for start in range(0, rho.size, 32):
            stop = min(start+32, rho.size)
            x = rho[start:stop, None]*k[None, :]
            J = jv(m, x)
            dJ = m*J/x-jv(m+1, x)
            samples[mode, :, start:stop] = np.stack((
                J@w, dJ@(k*w), -J@(k*w), -dJ@(k*k*w),
            ))
    return samples


def _build_cylinder_splines(settings: CylinderSettings):
    R = settings.magnets/2
    rho = np.linspace(settings.lower_ratio*R, settings.upper_ratio*R, settings.nodes)
    # Check quadrature separately from truncation and spline interpolation.
    probe = np.linspace(rho[0], rho[-1], 5)
    coarse = _spectral_samples(settings, probe)
    fine = _spectral_samples(settings, probe, refinement=2)
    if np.max(np.abs(coarse-fine)) > 1e-10*max(1, np.max(np.abs(fine))):
        raise RuntimeError("Cylinder spectral quadrature did not converge")
    samples = _spectral_samples(settings, rho)
    splines = []
    for mode, (A, dA, Z, dZ) in enumerate(samples):
        m = (2*mode+1)*settings.magnets//2
        splines.append((m,
            CubicSpline(rho, A, bc_type=((1, dA[0]), (1, dA[-1]))),
            CubicSpline(rho, Z, bc_type=((1, dZ[0]), (1, dZ[-1])))))
    return splines


def _surface_components(x, y, settings: CylinderSettings, *, refined=False):
    """Untruncated real-space field: two uniformly charged circular end faces.

    Field gradients differentiate the Coulomb kernel analytically, rather than
    taking finite differences of the field or reusing the Fourier coefficients.
    Disk weights integrate normalized area, leaving the moment-normalizing 1/L.
    """
    order = settings.reference_order*(2 if refined else 1)
    angles = settings.reference_angles*(2 if refined else 1)
    gx, gw = leggauss(order)
    radial = settings.magnet_radius*np.sqrt((gx+1)/2)
    theta = 2*np.pi*np.arange(angles)/angles
    disk_x = (radial[:, None]*np.cos(theta)).ravel()
    disk_y = (radial[:, None]*np.sin(theta)).ravel()
    weights = np.repeat(gw/(2*angles), angles)/settings.thickness
    flat_x, flat_y = x.ravel(), y.ravel()
    result = np.zeros((8, x.size))
    R = settings.magnets/2
    for start in range(0, x.size, 256):
        stop = min(start+256, x.size)
        out = result[:, start:stop]
        for magnet in range(settings.magnets):
            phi = magnet*2*np.pi/settings.magnets
            dx = flat_x[start:stop, None]-R*np.cos(phi)-disk_x
            dy = flat_y[start:stop, None]-R*np.sin(phi)-disk_y
            for face in (-1, 1):
                dz = settings.height-face*settings.thickness/2
                q = dx*dx+dy*dy+dz*dz
                inv3 = q**(-1.5)
                inv5 = inv3/q
                w = weights*face*(-1 if magnet % 2 else 1)
                out[0] += (dx*inv3)@w
                out[1] += (dy*inv3)@w
                out[2] += (dz*inv3)@w
                out[3] += (inv3-3*dx*dx*inv5)@w
                out[4] += (-3*dx*dy*inv5)@w
                out[5] += (-3*dx*dz*inv5)@w
                out[6] += (inv3-3*dy*dy*inv5)@w
                out[7] += (-3*dy*dz*inv5)@w
    return {name: values.reshape(x.shape) for (name, _), values in zip(COMPONENTS, result)}


def _check_reference(settings: CylinderSettings):
    R = settings.magnets/2
    rho = np.array([settings.lower_ratio*R, R-settings.magnet_radius, R,
                    R+settings.magnet_radius, settings.upper_ratio*R])[:, None]
    phi = np.array([0., np.pi/settings.magnets, .37])[None, :]
    x, y = rho*np.cos(phi), rho*np.sin(phi)
    coarse = _surface_components(x, y, settings)
    fine = _surface_components(x, y, settings, refined=True)
    for name, _ in COMPONENTS:
        if np.max(np.abs(coarse[name]-fine[name])) > 1e-9*max(1, np.max(np.abs(fine[name]))):
            raise RuntimeError("Surface reference did not converge; increase --reference-order/--reference-angles")


def _comparison(settings: CylinderSettings):
    _check_reference(settings)
    R = settings.magnets/2
    normalized = np.linspace(-settings.upper_ratio, settings.upper_ratio, settings.grid)
    nx, ny = np.meshgrid(normalized, normalized)
    x, y = R*nx, R*ny
    exact = _surface_components(x, y, settings)
    lut = _lut_components(x, y, settings, splines=_build_cylinder_splines(settings))
    difference = {name: lut[name]-exact[name] for name, _ in COMPONENTS}
    return nx, ny, exact, lut, difference


def main():
    defaults = CylinderSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("magnets", "modes", "nodes", "grid", "reference_order", "reference_angles"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("height", "lower_ratio", "upper_ratio", "magnet_radius", "thickness"):
        parser.add_argument("--"+name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--no-show", action="store_true", help="render once and exit without opening a window")
    args = parser.parse_args()
    settings = CylinderSettings(**{name: getattr(args, name) for name in defaults.__dataclass_fields__})
    if settings.magnets < 2 or settings.magnets % 2:
        parser.error("--magnets must be a positive even integer")
    if min(settings.modes, settings.nodes-1, settings.grid-15,
           settings.reference_order-1, settings.reference_angles-3) < 1:
        parser.error("modes >= 1, nodes >= 2, grid >= 16, reference order >= 2 and angles >= 4 required")
    if not (np.all(np.isfinite([settings.height, settings.magnet_radius, settings.thickness,
                               settings.lower_ratio, settings.upper_ratio]))
            and settings.magnet_radius > 0 and settings.thickness > 0
            and settings.height > settings.thickness/2
            and 0 < settings.lower_ratio < settings.upper_ratio):
        parser.error("positive radius/thickness, height > thickness/2, and ordered positive ratios required")
    comparison = InteractiveComparison(settings, args.save_dir, comparison=_comparison,
                                      title="Axially magnetized solid cylinders")
    if args.save_dir is not None:
        comparison.save(None)
    if args.no_show:
        plt.close(comparison.control_figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
