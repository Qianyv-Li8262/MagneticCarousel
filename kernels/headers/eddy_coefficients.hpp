// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "hysteresis_config.hpp"
#include "../generated/eddy_current_coefficients.generated.hpp"

// Coefficients only. Advancement and magnetic-moment logic belong elsewhere.
static_assert(eddy_coefficients::mode_count == 4*kEddyCurrentModeCount,
              "eddy coefficients are stale; rebuild generate_eddy_coefficients");
