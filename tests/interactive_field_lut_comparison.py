# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Interactive comparison of the exact dipole field and the Fourier-spline LUT model.

This is a standalone visualization tool.  It does not read the production LUT
binary and does not invoke the C++ generator.  Instead, it independently builds
the same clamped cubic splines with SciPy so that the magnet count and retained
Fourier mode count can be changed interactively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Button, Slider
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gammaln, hyp2f1


COMPONENTS = (
    ("h_x", r"$h_x$"),
    ("h_y", r"$h_y$"),
    ("h_z", r"$h_z$"),
    ("G_xx", r"$\partial_x h_x$"),
    ("G_xy", r"$\partial_y h_x$"),
    ("G_xz", r"$\partial_z h_x$"),
    ("G_yy", r"$\partial_y h_y$"),
    ("G_yz", r"$\partial_z h_y$"),
)


@dataclass(frozen=True)
class Settings:
    magnets: int = 8
    modes: int = 2
    nodes: int = 128
    grid: int = 161
    height: float = 0.3
    lower_ratio: float = 0.5
    upper_ratio: float = 1.5


@dataclass
class ComponentFigure:
    figure: plt.Figure
    axes: tuple[plt.Axes, plt.Axes, plt.Axes]
    images: tuple
    colorbars: tuple


def _radial_function(harmonic: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return f_m(t), f'_m(t), and f''_m(t) from the analytic expansion."""
    m = float(harmonic)
    p = m + 1.5
    a = p
    b = 1.5
    c = m + 1.0
    x = t * t
    coefficient = np.exp(gammaln(m + 1.5) - gammaln(1.5) - gammaln(m + 1.0))
    f0 = hyp2f1(a, b, c, x)
    f1 = hyp2f1(a + 1.0, b + 1.0, c + 1.0, x)
    f2 = hyp2f1(a + 2.0, b + 2.0, c + 2.0, x)
    ab_over_c = a * b / c
    shifted_ab_over_c = (a + 1.0) * (b + 1.0) / (c + 1.0)
    f = coefficient * np.power(t, p) * f0
    df = coefficient * (
        p * np.power(t, p - 1.0) * f0
        + 2.0 * ab_over_c * np.power(t, p + 1.0) * f1
    )
    d2f = coefficient * (
        p * (p - 1.0) * np.power(t, p - 2.0) * f0
        + 2.0 * ab_over_c * (2.0 * p + 1.0) * np.power(t, p) * f1
        + 4.0 * ab_over_c * shifted_ab_over_c * np.power(t, p + 2.0) * f2
    )
    return f, df, d2f


def _mode_samples(
    harmonic: int, rho: np.ndarray, height: float, radius: float, magnets: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return A, dA/drho, Z=dA/dz, and dZ/drho on radial nodes."""
    z = height
    chi = (z * z + rho * rho + radius * radius) / (2.0 * rho * radius)
    discriminant = np.sqrt(chi * chi - 1.0)
    t = 1.0 / (chi + discriminant)
    dchi_drho = (rho * rho - z * z - radius * radius) / (2.0 * rho * rho * radius)
    dchi_dz = z / (rho * radius)
    d2chi_drho_dz = -z / (rho * rho * radius)
    dt_drho = -t * dchi_drho / discriminant
    dt_dz = -t * dchi_dz / discriminant
    d2t_drho_dz = (
        dchi_drho * dchi_dz / np.power(discriminant, 3)
        - t * d2chi_drho_dz / discriminant
    )
    f, df, d2f = _radial_function(harmonic, t)
    scale = 2.0 * magnets / (np.power(radius, 1.5) * np.power(rho, 1.5))
    A = scale * z * f
    dA_drho = scale * z * (df * dt_drho - 1.5 * f / rho)
    Z = scale * (f + z * df * dt_dz)
    dZ_drho = scale * (
        df * dt_drho
        - 1.5 * f / rho
        + z * (d2f * dt_drho * dt_dz + df * d2t_drho_dz - 1.5 * df * dt_dz / rho)
    )
    return A, dA_drho, Z, dZ_drho


def _build_splines(settings: Settings) -> list[tuple[int, CubicSpline, CubicSpline]]:
    radius = settings.magnets / 2.0
    rho_nodes = np.linspace(
        settings.lower_ratio * radius,
        settings.upper_ratio * radius,
        settings.nodes,
    )
    splines = []
    for mode in range(settings.modes):
        harmonic = (2 * mode + 1) * settings.magnets // 2
        A, dA, Z, dZ = _mode_samples(
            harmonic, rho_nodes, settings.height, radius, settings.magnets
        )
        A_spline = CubicSpline(
            rho_nodes,
            A,
            bc_type=((1, dA[0]), (1, dA[-1])),
        )
        Z_spline = CubicSpline(
            rho_nodes,
            Z,
            bc_type=((1, dZ[0]), (1, dZ[-1])),
        )
        splines.append((harmonic, A_spline, Z_spline))
    return splines


def _exact_components(x: np.ndarray, y: np.ndarray, settings: Settings) -> dict[str, np.ndarray]:
    radius = settings.magnets / 2.0
    z = settings.height
    shape = x.shape
    flat_x = x.ravel()[:, None]
    flat_y = y.ravel()[:, None]
    index = np.arange(settings.magnets)
    angles = 2.0 * np.pi * index / settings.magnets
    signs = np.where(index % 2 == 0, 1.0, -1.0)[None, :]
    dx = flat_x - radius * np.cos(angles)[None, :]
    dy = flat_y - radius * np.sin(angles)[None, :]
    rho_squared = dx * dx + dy * dy
    q = rho_squared + z * z
    q_to_minus_five_halves = np.power(q, -2.5)
    q_to_minus_seven_halves = np.power(q, -3.5)
    overall = 3.0 * signs * q_to_minus_seven_halves

    values = {
        "h_x": np.sum(signs * 3.0 * z * dx * q_to_minus_five_halves, axis=1),
        "h_y": np.sum(signs * 3.0 * z * dy * q_to_minus_five_halves, axis=1),
        "h_z": np.sum(signs * (2.0 * z * z - rho_squared) * q_to_minus_five_halves, axis=1),
        "G_xx": np.sum(overall * z * (q - 5.0 * dx * dx), axis=1),
        "G_xy": np.sum(-5.0 * overall * z * dx * dy, axis=1),
        "G_xz": np.sum(overall * dx * (rho_squared - 4.0 * z * z), axis=1),
        "G_yy": np.sum(overall * z * (q - 5.0 * dy * dy), axis=1),
        "G_yz": np.sum(overall * dy * (rho_squared - 4.0 * z * z), axis=1),
    }
    return {name: value.reshape(shape) for name, value in values.items()}


def _lut_components(x: np.ndarray, y: np.ndarray, settings: Settings, *, splines=None) -> dict[str, np.ndarray]:
    radius = settings.magnets / 2.0
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    valid = (rho >= settings.lower_ratio * radius) & (rho <= settings.upper_ratio * radius)
    radial = rho[valid]
    angle = phi[valid]

    h_rho = np.zeros_like(radial)
    h_phi = np.zeros_like(radial)
    h_z = np.zeros_like(radial)
    G_rr = np.zeros_like(radial)
    G_rp = np.zeros_like(radial)
    G_rz = np.zeros_like(radial)
    G_pp = np.zeros_like(radial)
    G_pz = np.zeros_like(radial)

    if splines is None:
        splines = _build_splines(settings)
    for harmonic, A_spline, Z_spline in splines:
        cosine = np.cos(harmonic * angle)
        sine = np.sin(harmonic * angle)
        A = A_spline(radial)
        dA = A_spline(radial, 1)
        d2A = A_spline(radial, 2)
        Z = Z_spline(radial)
        dZ = Z_spline(radial, 1)
        h_rho -= dA * cosine
        h_phi += harmonic * A / radial * sine
        h_z -= Z * cosine
        G_rr -= d2A * cosine
        G_rp += harmonic * (dA / radial - A / (radial * radial)) * sine
        G_rz -= dZ * cosine
        G_pp += (harmonic * harmonic * A / (radial * radial) - dA / radial) * cosine
        G_pz += harmonic * Z / radial * sine

    cosine_phi = np.cos(angle)
    sine_phi = np.sin(angle)
    computed = {
        "h_x": cosine_phi * h_rho - sine_phi * h_phi,
        "h_y": sine_phi * h_rho + cosine_phi * h_phi,
        "h_z": h_z,
        "G_xx": cosine_phi**2 * G_rr - 2.0 * cosine_phi * sine_phi * G_rp + sine_phi**2 * G_pp,
        "G_xy": cosine_phi * sine_phi * (G_rr - G_pp) + (cosine_phi**2 - sine_phi**2) * G_rp,
        "G_xz": cosine_phi * G_rz - sine_phi * G_pz,
        "G_yy": sine_phi**2 * G_rr + 2.0 * cosine_phi * sine_phi * G_rp + cosine_phi**2 * G_pp,
        "G_yz": sine_phi * G_rz + cosine_phi * G_pz,
    }
    result = {}
    for name, values in computed.items():
        field = np.full_like(rho, np.nan)
        field[valid] = values
        result[name] = field
    return result


def _comparison(settings: Settings):
    radius = settings.magnets / 2.0
    normalized = np.linspace(-settings.upper_ratio, settings.upper_ratio, settings.grid)
    normalized_x, normalized_y = np.meshgrid(normalized, normalized)
    x = radius * normalized_x
    y = radius * normalized_y
    exact = _exact_components(x, y, settings)
    lut = _lut_components(x, y, settings)
    difference = {name: lut[name] - exact[name] for name, _ in COMPONENTS}
    return normalized_x, normalized_y, exact, lut, difference


class InteractiveComparison:
    def __init__(self, settings: Settings, save_dir: Path | None = None,
                 comparison=_comparison, title="Point dipoles"):
        self.settings = settings
        self.save_dir = save_dir
        self.comparison = comparison
        self.title = title
        self.component_figures: dict[str, ComponentFigure] = {}
        self.control_figure = plt.figure(f"{title}: field LUT comparison", figsize=(18, 12))
        self.control_figure.subplots_adjust(left=0.045, right=0.98, bottom=0.07, top=0.87,
                                           wspace=0.5, hspace=0.55)
        self.plot_axes = self.control_figure.subplots(4, 6)
        magnet_axis = self.control_figure.add_axes((0.10, 0.925, 0.27, 0.018))
        mode_axis = self.control_figure.add_axes((0.49, 0.925, 0.25, 0.018))
        update_axis = self.control_figure.add_axes((0.79, 0.914, 0.08, 0.035))
        save_axis = self.control_figure.add_axes((0.89, 0.914, 0.08, 0.035))
        self.magnet_slider = Slider(
            magnet_axis, "Magnets", 2, max(32, settings.magnets), valinit=settings.magnets, valstep=2
        )
        self.mode_slider = Slider(
            mode_axis, "Modes", 1, max(16, settings.modes), valinit=settings.modes, valstep=1
        )
        self.update_button = Button(update_axis, "Update")
        self.save_button = Button(save_axis, "Save PNG")
        self.status = self.control_figure.text(0.5, 0.015, "", ha="center", va="bottom")
        self.update_button.on_clicked(self.update)
        self.save_button.on_clicked(self.save)
        self.update(None)

    def _current_settings(self) -> Settings:
        return replace(self.settings, magnets=int(self.magnet_slider.val),
                       modes=int(self.mode_slider.val))


    def update(self, _event) -> None:
        settings = self._current_settings()
        self.status.set_text("Computing …")
        self.control_figure.canvas.draw_idle()
        try:
            _, _, exact, lut, difference = self.comparison(settings)
        except (ValueError, RuntimeError) as error:
            self.status.set_text(str(error))
            self.control_figure.canvas.draw_idle()
            if _event is None:
                raise
            return
        extent = (
            -settings.upper_ratio,
            settings.upper_ratio,
            -settings.upper_ratio,
            settings.upper_ratio,
        )
        for index, (name, label) in enumerate(COMPONENTS):
            exact_values = exact[name]
            lut_values = lut[name]
            error_values = difference[name]
            shared_limit = max(np.nanmax(np.abs(exact_values)), np.nanmax(np.abs(lut_values)), 1e-15)
            error_limit = max(np.nanmax(np.abs(error_values)), 1e-15)
            shared_norm = TwoSlopeNorm(vmin=-shared_limit, vcenter=0.0, vmax=shared_limit)
            error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
            if name not in self.component_figures:
                figure = self.control_figure
                row, group = divmod(index, 2)
                axes = tuple(self.plot_axes[row, group*3:group*3+3])
                arrays = (exact_values, lut_values, error_values)
                norms = (shared_norm, shared_norm, error_norm)
                titles = (f"{label} Reference", f"{label} LUT", "LUT − Reference")
                images = []
                colorbars = []
                for axis, values, norm, title in zip(axes, arrays, norms, titles):
                    image = axis.imshow(
                        values,
                        origin="lower",
                        extent=extent,
                        cmap="coolwarm",
                        norm=norm,
                        interpolation="nearest",
                    )
                    axis.set_title(title, fontsize=9)
                    axis.set_xlabel(r"$x/R$", fontsize=8)
                    axis.set_ylabel(r"$y/R$", fontsize=8)
                    axis.tick_params(labelsize=7)
                    axis.set_aspect("equal")
                    images.append(image)
                    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.03)
                    colorbar.ax.tick_params(labelsize=6)
                    colorbars.append(colorbar)
                self.component_figures[name] = ComponentFigure(
                    figure, axes, tuple(images), tuple(colorbars)
                )
            component = self.component_figures[name]
            arrays = (exact_values, lut_values, error_values)
            norms = (shared_norm, shared_norm, error_norm)
            for image, colorbar, values, norm in zip(
                component.images, component.colorbars, arrays, norms
            ):
                image.set_data(values)
                image.set_extent(extent)
                image.set_norm(norm)
                colorbar.update_normal(image)
            valid = np.isfinite(lut_values)
            if not np.any(valid):
                raise ValueError("No grid pixels lie inside the LUT domain; increase --grid")
            reference_peak = max(np.max(np.abs(exact_values[valid])), 1e-15)
            relative = error_limit / reference_peak
            component.axes[2].set_title(f"max |Δ|/peak: {relative:.2e}", fontsize=9)
        self.settings = settings
        geometry = (f", a={settings.magnet_radius:g}, L={settings.thickness:g}"
                    if hasattr(settings, "magnet_radius") else "")
        self.control_figure.suptitle(f"{self.title}: N={settings.magnets}, modes={settings.modes}, "
                                   f"nodes={settings.nodes}, h={settings.height:g}{geometry}", y=0.99)
        self.status.set_text(
            "Reference and LUT share a colour scale; errors have their own scale. "
            "Blank LUT pixels are outside the table domain."
        )
        self.control_figure.canvas.draw_idle()

    def save(self, _event) -> None:
        directory = self.save_dir or Path(__file__).resolve().parents[1] / "build" / "field-lut-comparison"
        directory.mkdir(parents=True, exist_ok=True)
        name = "cylinder_comparison.png" if hasattr(self.settings, "magnet_radius") else "point_comparison.png"
        self.control_figure.savefig(directory / name, dpi=160)
        self.status.set_text(f"Saved {directory / name}")
        self.control_figure.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnets", type=int, default=8)
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--grid", type=int, default=161)
    parser.add_argument("--height", type=float, default=0.3)
    parser.add_argument("--lower-ratio", type=float, default=0.5)
    parser.add_argument("--upper-ratio", type=float, default=1.5)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--no-show", action="store_true", help="render once and exit without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.magnets < 2 or args.magnets % 2:
        raise SystemExit("--magnets must be a positive even integer")
    if args.modes < 1 or args.nodes < 2 or args.grid < 16:
        raise SystemExit("modes >= 1, nodes >= 2, and grid >= 16 are required")
    if (not np.all(np.isfinite([args.lower_ratio, args.upper_ratio, args.height]))
            or not (0.0 < args.lower_ratio < args.upper_ratio) or args.height <= 0.0):
        raise SystemExit("invalid radial ratios or height")
    settings = Settings(
        magnets=args.magnets,
        modes=args.modes,
        nodes=args.nodes,
        grid=args.grid,
        height=args.height,
        lower_ratio=args.lower_ratio,
        upper_ratio=args.upper_ratio,
    )
    comparison = InteractiveComparison(settings, args.save_dir)
    if args.save_dir is not None:
        comparison.save(None)
    if args.no_show:
        plt.close(comparison.control_figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
