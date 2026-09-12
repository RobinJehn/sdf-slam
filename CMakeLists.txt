cmake_minimum_required(VERSION 3.20)
project(sdf_slam VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SDF_SLAM_BUILD_TESTS "Build unit tests" ON)
option(SDF_SLAM_BUILD_BENCHMARKS "Build solver benchmarks" ON)

# see DEC-0005 lean-dependency-stack
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

include(FetchContent)

# Fetched third-party projects declare ancient cmake_minimum_required values;
# this floor keeps modern CMake configuring them.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(
  nanoflann
  GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
  GIT_TAG v1.6.2
)
set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nanoflann yaml-cpp)

add_library(sdf_slam_core
  src/core/scan.cpp
  src/core/normals.cpp
  src/core/hallucination.cpp
  src/problem/problem.cpp
  src/solvers/lm_solver.cpp
  src/solvers/ceres_solver.cpp
  src/pipeline/config.cpp
  src/pipeline/runner.cpp
)
target_include_directories(sdf_slam_core PUBLIC src)
target_link_libraries(sdf_slam_core
  PUBLIC Eigen3::Eigen
  PRIVATE Ceres::ceres nanoflann::nanoflann yaml-cpp::yaml-cpp
)
target_compile_options(sdf_slam_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(run_slam apps/run_slam.cpp)
target_link_libraries(run_slam PRIVATE sdf_slam_core)
target_compile_options(run_slam PRIVATE -Wall -Wextra -Wpedantic)

if(SDF_SLAM_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(sdf_slam_tests
    tests/test_pose.cpp
    tests/test_scan_io.cpp
    tests/test_grid_map.cpp
    tests/test_hallucination.cpp
    tests/test_normals.cpp
    tests/test_jacobians.cpp
    tests/test_e2e.cpp
  )
  target_link_libraries(sdf_slam_tests PRIVATE sdf_slam_core GTest::gtest_main)
  target_compile_definitions(sdf_slam_tests
    PRIVATE SDF_SLAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  target_compile_options(sdf_slam_tests PRIVATE -Wall -Wextra -Wpedantic)

  include(GoogleTest)
  gtest_discover_tests(sdf_slam_tests)
endif()

if(SDF_SLAM_BUILD_BENCHMARKS)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(bench_solvers apps/bench_solvers.cpp)
  target_link_libraries(bench_solvers PRIVATE sdf_slam_core benchmark::benchmark_main)
  target_compile_definitions(bench_solvers
    PRIVATE SDF_SLAM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  target_compile_options(bench_solvers PRIVATE -Wall -Wextra -Wpedantic)
endif()
