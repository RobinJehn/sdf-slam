#include <gtest/gtest.h>

#include "core/grid_map.hpp"

namespace sdf_slam {
namespace {

GridMap MakeMap() {
  GridMap map(3, 3, {0.0, 0.0}, {2.0, 2.0});
  // Values: node (w, h) = w + 10 * h, an asymmetric pattern.
  for (int h = 0; h < 3; ++h) {
    for (int w = 0; w < 3; ++w) {
      map.Value(w, h) = w + 10.0 * h;
    }
  }
  return map;
}

TEST(GridMap, InterpolationMatchesManualBilinear) {
  const GridMap map = MakeMap();
  GridMap::CellRef cell;
  ASSERT_TRUE(map.Locate({0.5, 0.25}, cell));
  // Manual: (1-a)(1-b)*0 + a(1-b)*1 + (1-a)b*10 + ab*11 with a=0.5, b=0.25.
  const double expected = 0.5 * 0.75 * 1.0 + 0.5 * 0.25 * 10.0 + 0.5 * 0.25 * 11.0;
  EXPECT_NEAR(map.Interpolate(cell), expected, 1e-12);
}

TEST(GridMap, GradientMatchesFiniteDifferenceOfInterpolant) {
  const GridMap map = MakeMap();
  const Eigen::Vector2d p(0.7, 1.3);
  GridMap::CellRef cell;
  ASSERT_TRUE(map.Locate(p, cell));
  const Eigen::Vector2d grad = map.Gradient(cell);

  const double eps = 1e-7;
  GridMap::CellRef cell_x_plus;
  GridMap::CellRef cell_x_minus;
  ASSERT_TRUE(map.Locate(p + Eigen::Vector2d(eps, 0.0), cell_x_plus));
  ASSERT_TRUE(map.Locate(p - Eigen::Vector2d(eps, 0.0), cell_x_minus));
  const double fd_x = (map.Interpolate(cell_x_plus) - map.Interpolate(cell_x_minus)) / (2 * eps);
  GridMap::CellRef cell_y_plus;
  GridMap::CellRef cell_y_minus;
  ASSERT_TRUE(map.Locate(p + Eigen::Vector2d(0.0, eps), cell_y_plus));
  ASSERT_TRUE(map.Locate(p - Eigen::Vector2d(0.0, eps), cell_y_minus));
  const double fd_y = (map.Interpolate(cell_y_plus) - map.Interpolate(cell_y_minus)) / (2 * eps);

  EXPECT_NEAR(grad.x(), fd_x, 1e-5);
  EXPECT_NEAR(grad.y(), fd_y, 1e-5);
}

TEST(GridMap, PointOnGridLineTakesRightAndTopCell) {
  const GridMap map = MakeMap();
  GridMap::CellRef cell;
  ASSERT_TRUE(map.Locate({1.0, 1.0}, cell));
  EXPECT_EQ(cell.w, 1);
  EXPECT_EQ(cell.h, 1);
  EXPECT_NEAR(cell.alpha, 0.0, 1e-12);
  EXPECT_NEAR(cell.beta, 0.0, 1e-12);
}

TEST(GridMap, UpperDomainBoundaryFallsBackToLastCell) {
  const GridMap map = MakeMap();
  GridMap::CellRef cell;
  ASSERT_TRUE(map.Locate({2.0, 2.0}, cell));
  EXPECT_EQ(cell.w, 1);
  EXPECT_EQ(cell.h, 1);
  EXPECT_NEAR(cell.alpha, 1.0, 1e-12);
  EXPECT_NEAR(cell.beta, 1.0, 1e-12);
}

TEST(GridMap, OutsideDomainReturnsFalse) {
  const GridMap map = MakeMap();
  GridMap::CellRef cell;
  EXPECT_FALSE(map.Locate({-0.1, 1.0}, cell));
  EXPECT_FALSE(map.Locate({1.0, 2.1}, cell));
}

}  // namespace
}  // namespace sdf_slam
