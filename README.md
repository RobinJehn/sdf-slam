# sdf-slam

A 2D SLAM system that jointly optimizes sensor poses and a continuous signed
distance function (SDF) map in a single nonlinear least-squares problem.
Improved C++ reimplementation of the MInf dissertation *SDF SLAM* (Robin Jehn,
University of Edinburgh, 2026).

![Estimated SDF map on the simulated dataset](docs/figures/simu_76_map.png)

## How it works

The state vector holds every (active) SDF grid value plus the poses of all
scans (frame 0 fixed). Four residual types form one objective:

- **Scan**: the interpolated SDF at each transformed scan point must be zero.
- **Hallucination**: points sampled along each beam around the surface carry
  known signed distances, preventing the degenerate all-zero map.
- **Eikonal**: the SDF gradient projected onto the estimated surface normal
  must have unit magnitude.
- **Odometry**: relative pose measurements between consecutive frames.

A Levenberg-Marquardt solver exploits the Jacobian's sparsity (7 nonzeros per
point row) via sparse Cholesky. A Ceres backend solves the same problem for
comparison; `bench_solvers` measures both.

## Improvements over the dissertation implementation

- Parallel residual/Jacobian assembly; 100 batch iterations on the 76-scan
  simulated dataset run in ~9 s.
- Optional active-region state: only observed grid nodes are optimized.
- Three normal-estimation methods (`pca`, `hybrid`, `weighted`) for the
  Eikonal residual. The ablation on the noisy simulated dataset picked
  `weighted` as the default: 3x lower relative rotation error than the PCA
  baseline (DEC-0003).
- Analytic bilinear map gradients (consistent with the interpolation model),
  angle-wrapped odometry residuals, Jacobians verified against finite
  differences in the test suite.
- Lean dependencies: no PCL, no OpenCV.

## Build

Requires CMake ≥ 3.20, Ninja, Eigen, and Ceres (`brew install eigen
ceres-solver` / `apt install libeigen3-dev libceres-dev`). Everything else is
fetched at configure time.

```sh
make build          # configure + compile
make test           # unit + end-to-end tests
make check          # format check + build + tests + clang-tidy (CI gate)
make bench          # LM vs Ceres solver benchmark
```

## Run

```sh
./build/run_slam configs/simu_76_batch.yaml     # dissertation experiment 4.1.1
./build/run_slam configs/simu_76_noise.yaml     # experiment 4.1.3
./build/run_slam configs/intel_incremental.yaml # experiment 4.2.3 (slow)
```

Outputs land in `out/<run>/`: `poses_*.csv`, `map.txt`, `metrics.json`.
Figures:

```sh
cd tools/viz
uv run plot_map.py ../../out/simu_76_batch
uv run plot_trajectory.py ../../out/simu_76_batch
```

## Results (simu_76, ground-truth odometry)

| Metric | This repo | Dissertation (table 4.2) |
| --- | --- | --- |
| Mean translation error | 0.039 | 0.022 |
| Mean rotation error (rad) | 0.00095 | 0.00037 |
| Mean rel. translation error | 0.0041 | 0.0031 |
| Mean rel. rotation error (rad) | 0.00016 | 0.00014 |

On the full 910-scan Intel dataset the incremental run (10 LM iterations per
new scan, active region on) finishes in 16.4 minutes on an M-series laptop
and stays globally consistent without loop closure:

![Estimated SDF map on the full Intel dataset](docs/figures/intel_full_map.png)

## Solver benchmark (simu_10, 5 LM iterations, M-series laptop)

| Case | Time |
| --- | --- |
| Hand-rolled LM, dense map | 127 ms |
| Hand-rolled LM, active region | 52 ms |
| Ceres, dense map | 194 ms |
| Ceres, active region | 125 ms |

## Layout

- `src/core` — poses, PCD/dataset loading, SDF grid, normals, hallucinated points
- `src/problem` — residual math with analytic Jacobians, problem assembly
- `src/solvers` — hand-rolled LM and Ceres backends
- `src/pipeline` — YAML config, batch/incremental runner, metrics
- `docs/product` — PRD and decision records (DEC-NNNN, referenced from code)
- `data` — vendored simulated + Intel Research Lab datasets

## License

MIT
