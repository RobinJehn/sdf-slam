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
/// The sign follows the hallucination convention: positive between the sensor
/// and the surface, negative behind it. A node takes the sign of the scan that
/// owns its nearest point. Scans with no points and an empty scan list leave
/// the map unchanged.
void InitializeFromScans(GridMap& map, const std::vector<Scan>& scans,
                         const std::vector<Pose2>& poses);

}  // namespace sdf_slam
