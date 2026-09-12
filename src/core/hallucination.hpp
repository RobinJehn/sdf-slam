#pragma once

#include <Eigen/Core>
#include <vector>

#include "core/scan.hpp"

namespace sdf_slam {

/// A hallucinated point on a sensor beam with its expected SDF value
/// (dissertation section 3.2.2).
struct HallucinatedPoint {
  Eigen::Vector2d point_sensor;
  double expected_sdf{0.0};
};

struct HallucinationOptions {
  /// Total hallucinated points per scan point. With both_directions the count
  /// splits evenly between the free-space side and the occluded side.
  int points_per_scan_point{6};
  double step_size{0.1};
  bool both_directions{true};
};

/// Generates hallucinated points along each beam of a scan. Points toward the
/// sensor origin get expected SDF +j*step (free space); points beyond the
/// surface get -j*step (inside the object).
std::vector<HallucinatedPoint> GenerateHallucinatedPoints(const Scan& scan,
                                                          const HallucinationOptions& options);

}  // namespace sdf_slam
