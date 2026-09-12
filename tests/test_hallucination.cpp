#include <gtest/gtest.h>

#include "core/hallucination.hpp"

namespace sdf_slam {
namespace {

TEST(Hallucination, GeneratesPointsAlongBeamBothDirections) {
  Scan scan;
  scan.points.push_back({4.0, 0.0});  // beam along +x

  HallucinationOptions options;
  options.points_per_scan_point = 6;
  options.step_size = 0.1;
  options.both_directions = true;

  const auto points = GenerateHallucinatedPoints(scan, options);
  ASSERT_EQ(points.size(), 6U);

  // Alternating: toward origin (+delta), beyond surface (-delta).
  EXPECT_NEAR(points[0].point_sensor.x(), 3.9, 1e-12);
  EXPECT_NEAR(points[0].expected_sdf, 0.1, 1e-12);
  EXPECT_NEAR(points[1].point_sensor.x(), 4.1, 1e-12);
  EXPECT_NEAR(points[1].expected_sdf, -0.1, 1e-12);
  EXPECT_NEAR(points[4].point_sensor.x(), 3.7, 1e-12);
  EXPECT_NEAR(points[4].expected_sdf, 0.3, 1e-12);
  for (const auto& p : points) {
    EXPECT_NEAR(p.point_sensor.y(), 0.0, 1e-12);
  }
}

TEST(Hallucination, OneDirectionOnly) {
  Scan scan;
  scan.points.push_back({0.0, 2.0});

  HallucinationOptions options;
  options.points_per_scan_point = 3;
  options.step_size = 0.5;
  options.both_directions = false;

  const auto points = GenerateHallucinatedPoints(scan, options);
  ASSERT_EQ(points.size(), 3U);
  for (const auto& p : points) {
    EXPECT_GT(p.expected_sdf, 0.0);
    EXPECT_LT(p.point_sensor.y(), 2.0);
  }
}

TEST(Hallucination, SkipsZeroLengthBeams) {
  Scan scan;
  scan.points.push_back({0.0, 0.0});
  const auto points = GenerateHallucinatedPoints(scan, {});
  EXPECT_TRUE(points.empty());
}

}  // namespace
}  // namespace sdf_slam
