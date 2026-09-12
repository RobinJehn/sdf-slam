#include <gtest/gtest.h>

#include "core/pose.hpp"

namespace sdf_slam {
namespace {

TEST(Pose2, ApplyRotatesAndTranslates) {
  const Pose2 pose{1.0, 2.0, M_PI / 2.0};
  const Eigen::Vector2d p = pose.Apply({1.0, 0.0});
  EXPECT_NEAR(p.x(), 1.0, 1e-12);
  EXPECT_NEAR(p.y(), 3.0, 1e-12);
}

TEST(Pose2, ComposeRelativeRoundtrip) {
  const Pose2 a{0.3, -1.2, 0.7};
  const Pose2 b{-2.0, 4.5, -2.9};
  const Pose2 rel = a.RelativeTo(b);
  const Pose2 recomposed = a.Compose(rel);
  EXPECT_NEAR(recomposed.x, b.x, 1e-12);
  EXPECT_NEAR(recomposed.y, b.y, 1e-12);
  EXPECT_NEAR(WrapAngle(recomposed.theta - b.theta), 0.0, 1e-12);
}

TEST(Pose2, WrapAngleStaysInRange) {
  // +-pi maps to either seam end; only the magnitude is defined.
  EXPECT_NEAR(std::abs(WrapAngle(3.0 * M_PI)), M_PI, 1e-12);
  EXPECT_NEAR(std::abs(WrapAngle(-3.0 * M_PI)), M_PI, 1e-12);
  EXPECT_NEAR(WrapAngle(0.5), 0.5, 1e-12);
  EXPECT_NEAR(WrapAngle(2.0 * M_PI + 0.25), 0.25, 1e-12);
  EXPECT_NEAR(WrapAngle(-2.0 * M_PI - 0.25), -0.25, 1e-12);
}

}  // namespace
}  // namespace sdf_slam
