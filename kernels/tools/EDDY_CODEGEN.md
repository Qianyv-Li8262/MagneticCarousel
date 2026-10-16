# Eddy-current coefficient generation

The normal CMake build runs `generate_eddy_coefficients` before the CPU and CUDA
solver targets. No Python fitting or runtime lookup table is involved.

Inputs reuse `Mu_r` and the AVX group count from `hysteresis_config.hpp`, and
`Rref`, `omega_ref`, `mu0`, `sigma` from `hysteresis_physics.hpp`. `sigma` is the
user-approved estimate from a resistivity of 0.15 micro-ohm metre, not a
measurement. The code generator is compiled with `USE_FP64` regardless of the
runtime solver precision, then performs its calculations with Boost
`cpp_dec_float_100`. Physical inputs retain the precision of their source
definitions; the extra arithmetic precision protects root finding and fitting.

The generator retains the slowest two diffusion time constants and places the
remaining candidate time constants logarithmically. It uses nonnegative least
squares against the full analytic conducting-sphere response minus its static
response. It does not truncate the physical series after twelve terms.

`eddy_codegen_config.hpp` specifies the fit band, sampling and error threshold.
Validation uses a separate denser grid and checks the low-frequency slope after
rounding the output to FP64. A failure aborts code generation before replacing
the previous output. Zero-weight slots are valid and preserve the AVX layout.

Outputs:

- `kernels/generated/eddy_current_coefficients.generated.hpp`: aligned FP64 arrays `aj` and
  `inv_tau_ref_star`, plus material/reference metadata. CPU arrays are aligned
  to 32 bytes; CUDA arrays use device constant memory.
- `<build>/generated/eddy_current_coefficients.report.json`: precision, parameters and validation
  results. Errors describe sampled frequency response, not a bound on coupled
  trajectory error or arbitrary discontinuous transients.

Future runtime code can include `eddy_coefficients.hpp`. For mode `j`, use

```cpp
r_j = dt_star * eddy_coefficients::inv_tau_ref_star[j]
      / params.eddy_current_scale;
```

`eddy_current_scale = somega * sr * sr`. It must not be compiled into the
generated coefficients. Four adjacent entries form one AVX group.

The fit band describes the reference radius. With unchanged material and
`R = sr * Rref`, a physical input frequency `f` corresponds to `f * sr^2` in
the reference fit. Changes to disk speed and spin also change the input's
physical spectrum. Extend/revalidate the fit band when scanning beyond it;
neither a finite-band fit nor twelve modes guarantees arbitrary-frequency
accuracy. Advancement and magnetic-moment logic remain in `eddy_currents.hpp`
and are not changed by this generator.
