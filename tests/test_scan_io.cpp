#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "core/scan.hpp"

namespace sdf_slam {
namespace {

TEST(ScanIo, ParsesAsciiPcd) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "sdf_slam_test_scan.pcd";
  {
    std::ofstream file(path);
    file << "# .PCD v0.7 - Point Cloud Data file format\n"
         << "VERSION 0.7\nFIELDS x y\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\nWIDTH 2\nHEIGHT 1\n"
         << "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA ascii\n"
         << "1.5 -2.25\n0.0 3.0\n";
  }
  const Scan scan = LoadPcd(path);
  ASSERT_EQ(scan.points.size(), 2U);
  EXPECT_DOUBLE_EQ(scan.points[0].x(), 1.5);
  EXPECT_DOUBLE_EQ(scan.points[0].y(), -2.25);
  EXPECT_DOUBLE_EQ(scan.points[1].y(), 3.0);
  std::filesystem::remove(path);
}

TEST(ScanIo, LoadsVendoredDataset) {
  const Dataset dataset = LoadDataset(std::filesystem::path(SDF_SLAM_DATA_DIR) / "simu_10");
  EXPECT_EQ(dataset.scans.size(), dataset.poses.size());
  ASSERT_GE(dataset.scans.size(), 2U);
  EXPECT_FALSE(dataset.scans[0].points.empty());
  // First pose of the simulated dataset is the origin.
  EXPECT_DOUBLE_EQ(dataset.poses[0].x, 0.0);
  EXPECT_DOUBLE_EQ(dataset.poses[0].y, 0.0);
}

}  // namespace

TEST(ScanIo, LoadsCommaSeparatedPosesWithHeader) {
  const auto path = std::filesystem::temp_directory_path() / "sdf_slam_poses_test.csv";
  {
    std::ofstream file(path);
    file << "x,y,theta\n1.5,-2.0,0.25\n3.0,4.0,-0.5\n";
  }
  const std::vector<Pose2> poses = LoadPoses(path);
  std::filesystem::remove(path);
  ASSERT_EQ(poses.size(), 2U);
  EXPECT_DOUBLE_EQ(poses[0].x, 1.5);
  EXPECT_DOUBLE_EQ(poses[0].y, -2.0);
  EXPECT_DOUBLE_EQ(poses[0].theta, 0.25);
  EXPECT_DOUBLE_EQ(poses[1].theta, -0.5);
}

}  // namespace sdf_slam
