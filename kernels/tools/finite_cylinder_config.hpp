// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// These lengths use the same k-normalization as height and the radial LUT.
// CMake supplies them for the generator/reference, never for the dynamics.
#ifndef IYPT16_CYLINDER_RADIUS
#error "Specify IYPT16_CYLINDER_RADIUS (dimensionless magnet radius)"
#endif
#ifndef IYPT16_CYLINDER_THICKNESS
#error "Specify IYPT16_CYLINDER_THICKNESS (dimensionless magnet thickness)"
#endif
