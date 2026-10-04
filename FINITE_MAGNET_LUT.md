# Finite cylindrical magnets

Select the source geometry at CMake configure time:

```powershell
cmake --preset local-cpu -DIYPT16_MAGNET_MODEL=cylinder "-DIYPT16_MAGNET_RADIUS=0.5" "-DIYPT16_MAGNET_THICKNESS=0.5"
cmake --build --preset local-cpu-release
ctest --test-dir build/release --output-on-failure
```

The dimensions above are examples, not measured apparatus parameters. Both are
full physical lengths multiplied by the model's inverse length scale `k`;
`IYPT16_MAGNET_RADIUS` is the radius and `IYPT16_MAGNET_THICKNESS` is the full
thickness. A production cylinder configuration requires both explicitly.
`height` remains the distance from the magnet centre plane to the observation
plane, configured in `hysteresis_config_field.hpp`. This generator requires
`height > thickness / 2`.

`IYPT16_MAGNET_MODEL=point` selects the original point source. Selecting
`cylinder` forces the existing `IYPT16_FIELD_MODE=lut`. CPU and GPU manifests
record the model, dimensions and selected table hash. Both scanning scripts
override their field evaluation setting to LUT for a cylinder build, including
when their original setting was `exact`. The CUDA kernels and dynamics keep
the existing field interface and interpolation code.

## Construction

Magnets are uniformly axially magnetized solid cylinders, centred on the ring
of radius `R=N/2`, with alternating axial polarity. For azimuthal harmonic
`m=(2j+1)N/2`, the potential coefficient, with `Beta` factored out, is

```text
A_m(rho,h) = 2N integral_0^infinity
    G(kappa,h,L) F(kappa,a) J_m(kappa*rho) J_m(kappa*R) d kappa
F = 2 J_1(kappa*a)/(kappa*a)
G = exp[-kappa*(h-L/2)] * [-expm1(-kappa*L)]/L
```

Here `a` and `L` are the normalized radius and full thickness. This is the
exact finite-radius, finite-thickness shape factor for this magnetization;
there is no stack-of-point-dipoles approximation. `Beta` retains the
normalization of the total magnetic moment and is applied once at runtime.

The new generator evaluates the spectral integrals with 50-digit arithmetic,
32-point composite Gauss quadrature, a conservative omitted-tail bound of
`1e-25`, and a panel-refinement check of `1e-20` at five radii across all
retained modes. Bessel values for all modes share a downward recurrence.
It samples `A`, `dA/drho`, `Z=dA/dh`, and `dZ/drho`; the shared 100-digit
clamped-spline solver then produces the existing headerless layout:
`mode -> radial node -> [A, M_A, Z, M_Z]`. The two `M` slots are spline
second derivatives. Runtime table names and interpolation remain unchanged.

The intentional radial domain remains `[0.5R,1.5R]`. Current configuration
uses five modes and 204 nodes: 32,640 bytes for FP64 and 16,320 bytes for FP32.
A compile-time assertion enforces the 32 KiB FP64 table budget. Changing
geometry, magnet count, modes or nodes requires rerunning the field tests;
the budget alone does not guarantee accuracy for every geometry.

## Verification

With `BUILD_TESTING=ON`, CMake generates both point and cylinder tables.
`test_field_lut_vs_point` compares to the original analytic point field.
`test_field_lut_vs_cylinder` compares to an independent real-space integral
of the two circular surface-charge faces and its analytically differentiated
field kernel. This reference has no Fourier truncation or Bessel evaluation;
it checks its own area quadrature convergence before comparing the LUT.
Both tests sample all radial cells and compare all eight field/gradient
components with the existing `1e-4` tolerance, normalized by each component's
reference peak. For a point production build with no cylinder dimensions,
only the cylinder test fixture uses `a=L=0.5`.

CuPy loads the CMake-built `.cubin` files. The default CUDA architecture
selection remains `native`; an explicit cubin architecture must use a single
real target, for example `-DCMAKE_CUDA_ARCHITECTURES=120-real`.

## Interactive field comparison

```powershell
python tests/interactive_field_lut_comparison.py --height 1.610066235 --modes 5 --nodes 204
python tests/interactive_cylinder_field_lut_comparison.py --magnet-radius 0.5 --thickness 0.5 --height 1.610066235
```

Both tools use one window containing eight component groups (reference, LUT,
and signed error: 24 panels), plus magnet-count/mode sliders and Update/Save
buttons. Each reference/LUT pair shares a colour scale; the error uses its own
scale and reports maximum absolute error divided by the reference peak within
the table domain. Save writes one combined PNG. `--no-show --save-dir PATH`
renders and exports without opening a window.

These interactive tools independently rebuild the clamped splines with SciPy,
so their sliders are independent of the compiled production LUT. The cylinder
version uses the exact finite-cylinder spectral shape factors and an independent
surface-charge integral reference. It checks convergence of both integrals;
`--reference-order` and `--reference-angles` can increase the reference resolution
for difficult geometries. Beta is factored out of both displayed fields.

## Runtime build summary and incremental metadata

Animation, both scans, and the CPU benchmark print the compiled magnetic source
model, magnet count, field evaluation mode and normalized centre-plane height
on four prominent lines before evolution. Interactive terminals use a medium
blue (`#2864C8`) background with bright white text; redirected logs keep a plain
bordered block. `NO_COLOR` disables colour and `FORCE_COLOR=1` enables it.
Summaries read build manifests rather than assuming the command-line/default
field mode; the GPU scan displays cylinder builds as LUT even in dry-run mode.

CPU and GPU manifests are independent CMake outputs. A normal build updates them
automatically; explicitly request their dependency graph with:

```powershell
cmake --build --preset local-cpu-release --target build_metadata
```

Changing only a manifest writer refreshes JSON without compiling, linking or
copying binaries. Unchanged JSON content preserves its timestamp. Changing
compiled geometry/constants or field macros still builds their affected code
before updating metadata; changing cylinder dimensions also regenerates the LUT.
