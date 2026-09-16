#include "core/map_init.hpp"

#include <cstddef>
#include <limits>
#include <nanoflann.hpp>

namespace sdf_slam {

namespace {

/// nanoflann adaptor over a flat vector of 2D points.
struct PointCloudAdaptor {
  const std::vector<Eigen::Vector2d>* points;

  [[nodiscard]] size_t kdtree_get_point_count() const { return points->size(); }
  [[nodiscard]] double kdtree_get_pt(size_t idx, size_t dim) const {
    return (*points)[idx][static_cast<Eigen::Index>(dim)];
  }
  template <class Bbox>
  bool kdtree_get_bbox(Bbox& /*bbox*/) const {
    return false;
  }
};

using KdTree =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
                                        PointCloudAdaptor, 2>;

}  // namespace

void InitializeFromScans(GridMap& map, const std::vector<Scan>& scans,
                         const std::vector<Pose2>& poses, bool signed_distance) {
  std::vector<Eigen::Vector2d> points;
  std::vector<Eigen::Vector2d> origins;
  for (size_t i = 0; i < scans.size() && i < poses.size(); ++i) {
    const Eigen::Vector2d origin = poses[i].Translation();
    for (const auto& p : scans[i].points) {
      points.push_back(poses[i].Apply(p));
      origins.push_back(origin);
    }
  }
  if (points.empty()) {
    return;
  }

  const PointCloudAdaptor adaptor{&points};
  const KdTree tree(2, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  for (int h = 0; h < map.ny(); ++h) {
    for (int w = 0; w < map.nx(); ++w) {
      const Eigen::Vector2d node = map.NodePosition(w, h);
      const std::array<double, 2> query{node.x(), node.y()};
      size_t index = 0;
      double squared = std::numeric_limits<double>::max();
      nanoflann::KNNResultSet<double> result(1);
      result.init(&index, &squared);
      tree.findNeighbors(result, query.data());

      double sign = 1.0;
      if (signed_distance) {
        // Positive between the sensor and its surface point, negative behind.
        const double to_node = (node - origins[index]).norm();
        const double to_surface = (points[index] - origins[index]).norm();
        sign = to_node <= to_surface ? 1.0 : -1.0;
      }
      map.Value(w, h) = sign * std::sqrt(squared);
    }
  }
}

}  // namespace sdf_slam
