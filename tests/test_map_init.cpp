#include <gtest/gtest.h>

#include <cmath>

#include "core/map_init.hpp"

namespace sdf_slam {
namespace {

/// One scan of a wall along x = 2, seen from a sensor at the origin.
Scan WallAtX2() {
  Scan scan;
  for (int i = -20; i <= 20; ++i) {
    scan.points.emplace_back(2.0, 0.1 * i);
  }
  return scan;
}

TEST(MapInit, NodeOnTheSurfaceGetsZero) {
  GridMap map(11, 11, {0.0, -2.0}, {4.0, 2.0});
  InitializeFromScans(map, {WallAtX2()}, {Pose2{}});
  // node at x = 2.0, y = 0.0 sits on the wall
  EXPECT_NEAR(map.Value(5, 5), 0.0, 1e-9);
}

TEST(MapInit, SensorSideIsPositiveAndDistanceIsCorrect) {
  GridMap map(11, 11, {0.0, -2.0}, {4.0, 2.0});
  InitializeFromScans(map, {WallAtX2()}, {Pose2{}});
  // node at x = 1.2 lies 0.8 m in front of the wall, on the sensor side
  EXPECT_NEAR(map.Value(3, 5), 0.8, 1e-9);
}

TEST(MapInit, BehindTheSurfaceIsNegative) {
  GridMap map(11, 11, {0.0, -2.0}, {4.0, 2.0});
  InitializeFromScans(map, {WallAtX2()}, {Pose2{}});
  // node at x = 2.8 lies 0.8 m behind the wall, away from the sensor
  EXPECT_NEAR(map.Value(7, 5), -0.8, 1e-9);
}

TEST(MapInit, UsesThePoseToPlaceScanPoints) {
  GridMap map(11, 11, {0.0, -2.0}, {4.0, 2.0});
  // shifting the sensor +1 in x moves the wall to x = 3
  InitializeFromScans(map, {WallAtX2()}, {Pose2{1.0, 0.0, 0.0}});
  EXPECT_NEAR(map.Value(7, 5), 0.2, 1e-9);   // x = 2.8, in front of x = 3
  EXPECT_NEAR(map.Value(9, 5), -0.6, 1e-9);  // x = 3.6, behind it
}

TEST(MapInit, TakesTheNearestSurfaceAcrossScans) {
  GridMap map(11, 11, {0.0, -2.0}, {4.0, 2.0});
  // a second wall at x = 3 is nearer to the node at x = 2.8 than the one at x = 2
  InitializeFromScans(map, {WallAtX2(), WallAtX2()}, {Pose2{}, Pose2{1.0, 0.0, 0.0}});
  EXPECT_NEAR(std::abs(map.Value(7, 5)), 0.2, 1e-9);
}

TEST(MapInit, EmptyScansLeaveTheMapUntouched) {
  GridMap map(5, 5, {0.0, 0.0}, {1.0, 1.0}, 0.25);
  InitializeFromScans(map, {}, {});
  EXPECT_EQ(map.Value(2, 2), 0.25);
}

}  // namespace
}  // namespace sdf_slam
