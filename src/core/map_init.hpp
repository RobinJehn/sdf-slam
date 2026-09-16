#pragma once

#include <vector>

#include "core/grid_map.hpp"
#include "core/pose.hpp"
#include "core/scan.hpp"

namespace sdf_slam {

/// Fills every map node with the signed distance to the nearest scan point.
///
/// The solver's residuals only state what the map should be within the
/// hallucination band around observed surfaces (DEC-0007), so a map started
/// at a constant carries no distance information anywhere else. Seeding from
/// the scans supplies that level everywhere, leaving the eikonal residual to
/// maintain the field instead of inventing it.
///
/// With `signed_distance`, the sign follows the hallucination convention:
/// positive between the sensor and the surface, negative behind it, taken from
/// the scan that owns the nearest point. That rule is only projective, so it
/// breaks where an occluded region meets free space seen from elsewhere: the
/// field jumps from a large negative to a large positive value across that
/// seam, which no eikonal solution can hold. Without `signed_distance` every
/// node gets the unsigned distance, which has a kink at surfaces but no jump.
/// An empty scan list leaves the map unchanged.
void InitializeFromScans(GridMap& map, const std::vector<Scan>& scans,
                         const std::vector<Pose2>& poses, bool signed_distance);

}  // namespace sdf_slam
