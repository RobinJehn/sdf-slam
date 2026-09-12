#pragma once

#include <Eigen/Core>
#include <cmath>

namespace sdf_slam {

/// Wraps an angle to the interval [-pi, pi].
inline double WrapAngle(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

/// A 2D rigid-body pose: translation (x, y) and rotation theta.
struct Pose2 {
  double x{0.0};
  double y{0.0};
  double theta{0.0};

  [[nodiscard]] Eigen::Vector2d Translation() const { return {x, y}; }

  [[nodiscard]] Eigen::Matrix2d Rotation() const {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    Eigen::Matrix2d r;
    r << c, -s, s, c;
    return r;
  }

  /// Maps a point from this pose's frame into the parent frame.
  [[nodiscard]] Eigen::Vector2d Apply(const Eigen::Vector2d& p) const {
    return Rotation() * p + Translation();
  }

  /// Composes this pose with a relative pose: result = this ∘ rel.
  [[nodiscard]] Pose2 Compose(const Pose2& rel) const {
    const Eigen::Vector2d t = Apply(rel.Translation());
    return {t.x(), t.y(), WrapAngle(theta + rel.theta)};
  }

  /// Relative pose from this to other, so that this.Compose(result) == other.
  /// Matches the odometry model: dt = R(-theta_i) (t_{i+1} - t_i).
  [[nodiscard]] Pose2 RelativeTo(const Pose2& other) const {
    const Eigen::Vector2d dt = Rotation().transpose() * (other.Translation() - Translation());
    return {dt.x(), dt.y(), WrapAngle(other.theta - theta)};
  }
};

}  // namespace sdf_slam
