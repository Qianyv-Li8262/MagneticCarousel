// SPDX-License-Identifier: AGPL-3.0-only
#include "eddy_coefficients.hpp"

// Compile the generated arrays as device constant memory and index them at
// runtime. Production advancement is deliberately not connected by this test.
extern "C" __global__ void read_eddy_coefficients(double *weights, double *rates)
{
    const int j=static_cast<int>(threadIdx.x);
    if (j<eddy_coefficients::mode_count) {
        weights[j]=eddy_coefficients::aj[j];
        rates[j]=eddy_coefficients::inv_tau_ref_star[j];
    }
}
