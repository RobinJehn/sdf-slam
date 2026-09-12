#include "core/hallucination.hpp"

namespace sdf_slam {

std::vector<HallucinatedPoint> GenerateHallucinatedPoints(const Scan& scan,
                                                          const HallucinationOptions& options) {
  std::vector<HallucinatedPoint> result;
  if (options.points_per_scan_point <= 0) {
    return result;
  }
  const int per_side = options.points_per_scan_point / (options.both_directions ? 2 : 1);
  result.reserve(scan.points.size() * static_cast<size_t>(options.points_per_scan_point));

  for (const auto& point : scan.points) {
    const double norm = point.norm();
    if (norm < 1e-9) {
      continue;  // A beam of length zero has no direction.
    }
    const Eigen::Vector2d toward_origin = -point / norm * options.step_size;
    for (int j = 1; j <= per_side; ++j) {
      result.push_back({point + toward_origin * j, options.step_size * j});
      if (options.both_directions) {
        result.push_back({point - toward_origin * j, -options.step_size * j});
      }
    }
  }
  return result;
}

}  // namespace sdf_slam
