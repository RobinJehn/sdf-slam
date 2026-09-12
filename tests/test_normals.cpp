#include <gtest/gtest.h>

#include "core/normals.hpp"

namespace sdf_slam {
namespace {

/// A scan of a horizontal wall at y = 5 seen from the origin.
Scan WallScan() {
  Scan scan;
  for (int i = -20; i <= 20; ++i) {
    scan.points.push_back({0.2 * i, 5.0});
  }
  return scan;
}

TEST(Normals, WallNormalsPointTowardScanOrigin) {
  const std::vector<Scan> scans = {WallScan()};
  const std::vector<Pose2> poses = {{0.0, 0.0, 0.0}};
  const ScanNormals normals = ComputeScanNormals(scans, poses, 8);

  ASSERT_EQ(normals.normals.size(), scans[0].points.size());
  for (size_t i = 0; i < normals.normals.size(); ++i) {
    // Wall along x: normal must be (0, -1), toward the origin below the wall.
    EXPECT_NEAR(std::abs(normals.normals[i].x()), 0.0, 1e-9) << i;
    EXPECT_NEAR(normals.normals[i].y(), -1.0, 1e-9) << i;
    EXPECT_LT(normals.cornerness[i], 1e-6) << i;
  }
}

TEST(Normals, CornerPointsHaveHighCornerness) {
  // Two walls meeting at (5, 5): one along x, one along y.
  Scan scan;
  for (int i = 0; i <= 20; ++i) {
    scan.points.push_back({5.0 - 0.2 * i, 5.0});
    scan.points.push_back({5.0, 5.0 - 0.2 * i});
  }
  const ScanNormals normals = ComputeScanNormals({scan}, {{0.0, 0.0, 0.0}}, 10);

  // The corner point at (5, 5) appears twice; its neighborhood spans both
  // walls and PCA sees spread in both directions.
  double corner_max = 0.0;
  for (size_t i = 0; i < scan.points.size(); ++i) {
    if ((scan.points[i] - Eigen::Vector2d(5.0, 5.0)).norm() < 0.3) {
      corner_max = std::max(corner_max, normals.cornerness[i]);
    }
  }
  EXPECT_GT(corner_max, 0.1);
}

TEST(Normals, WeightedMethodScalesEikonalDownAtCorners) {
  Scan scan;
  for (int i = 0; i <= 20; ++i) {
    scan.points.push_back({5.0 - 0.2 * i, 5.0});
    scan.points.push_back({5.0, 5.0 - 0.2 * i});
  }
  const ScanNormals scan_normals = ComputeScanNormals({scan}, {{0.0, 0.0, 0.0}}, 10);
  const GridMap map(11, 11, {0.0, 0.0}, {10.0, 10.0});

  NormalOptions options;
  options.method = NormalMethod::kWeighted;
  options.corner_threshold = 0.2;
  const std::vector<NodeNormal> node_normals = AssignNodeNormals(map, scan_normals, options);

  // Node at (5, 5) sits on the corner; a node at (2, 5) sits on a clean wall.
  const NodeNormal& corner_node = node_normals[static_cast<size_t>(map.NodeId(5, 5))];
  const NodeNormal& wall_node = node_normals[static_cast<size_t>(map.NodeId(2, 5))];
  EXPECT_LT(corner_node.eikonal_scale, wall_node.eikonal_scale);
  EXPECT_NEAR(wall_node.eikonal_scale, 1.0, 1e-6);
}

TEST(Normals, HybridNormalStaysUnitLength) {
  Scan scan;
  for (int i = 0; i <= 20; ++i) {
    scan.points.push_back({5.0 - 0.2 * i, 5.0});
    scan.points.push_back({5.0, 5.0 - 0.2 * i});
  }
  const ScanNormals scan_normals = ComputeScanNormals({scan}, {{0.0, 0.0, 0.0}}, 10);
  const GridMap map(11, 11, {0.0, 0.0}, {10.0, 10.0});

  NormalOptions options;
  options.method = NormalMethod::kHybrid;
  const std::vector<NodeNormal> node_normals = AssignNodeNormals(map, scan_normals, options);
  for (const auto& node : node_normals) {
    EXPECT_NEAR(node.normal.norm(), 1.0, 1e-9);
  }
}

}  // namespace
}  // namespace sdf_slam
