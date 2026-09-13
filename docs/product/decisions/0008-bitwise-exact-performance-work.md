---
id: 0008
title: Bitwise-exact performance work on the chaotic recipe path
type: architecture
owner: Robin Jehn
created: 2026-09-13
updated: 2026-09-13
requirements:
- ../PRD.md
---

# DEC-0008 — Bitwise-exact performance work on the chaotic recipe path

## Context

Profiling showed the flagship run spends 96 % of its time in the solver: sparse LDLT ~55-61 %, the J^T J product ~16-19 %, Jacobian triplet merging ~10-13 %, Cost() ~9-11 %. Faster linear solvers (CHOLMOD supernodal LLT) and any parallel reduction change the floating-point rounding of the LM steps.

An experiment settled how much rounding freedom exists: a 1e-9 relative change to lambda_init collapses the lap-1 revisit median from 0.013 m to 0.687 m. CHOLMOD (a different elimination ordering, same mathematics) gives 0.782 m. The incremental recipe (DEC-0007) runs a fixed 10-iteration budget per increment, never converges tightly, and amplifies last-ulp step differences into macroscopic trajectory changes.

## Decision

Performance work on the recipe path must be bitwise-exact. A change qualifies only when the optimized binary reproduces the reference artifacts bit for bit (poses_estimated.csv and map.txt on the lap-1 testbed, then on a full flagship run).

Techniques that qualify, all landed:

- Cache Cost() values in the LM loop instead of recomputing on bitwise-unchanged state (3 calls per iteration down to 1).
- Parallelize per-element loops whose outputs are independent slots: scan normals, node normals, point and Eikonal row assembly.
- Replicate Eigen kernels operation for operation and split them across threads along boundaries that do not reorder any accumulation: the J^T J product (per-column accumulation order preserved), the -J^T r product (Eigen's two-accumulator kernel replicated exactly), and the simplicial LDLT numeric factorization (`src/solvers/parallel_ldlt.hpp`) scheduled over independent elimination-tree subtrees.
- Pin floating-point contraction where the compiler has a choice: the LDLT kernel uses std::fma exactly where the reference build fused (shift term, y-updates) and an unfused product where it did not (diagonal update), with FP_CONTRACT OFF in the kernel.

Unit tests assert bitwise equality of ParallelSimplicialLdlt against Eigen's SimplicialLDLT across thread counts.

## Consequences

- The recipe's headline numbers (0.065 m full Intel, 0.013 m lap-1) are knife-edge samples, not robust properties. Any dissertation-comparison claim needs this caveat; a different compiler, BLAS, or machine will produce different (likely much worse) metrics from the same config.
- `solver.linear_solver: cholmod` exists for experiments but is out of spec for the recipe: it changes the numbers, so a cholmod run is a new experiment, not a faster reproduction.
- Symbolic-analysis reuse in the LDLT is near-useless here (the J^T J pattern changes in ~99 % of iterations because points cross cells), so the win comes from parallelizing the numeric factorization, not from caching the analysis.

## Alternatives considered

- CHOLMOD supernodal LLT as the default (tried, rejected 2026-09-13): 1.8x faster on lap-1 but metric collapses (0.782 m) — chaos, not a bug; verified against a clean matrix churn test.
- Metric-equivalent tolerance for perf changes (rejected): the perturbation experiment shows no numeric tolerance exists between "bitwise" and "broken".
- Per-increment problem caching (rejected as primary lever): the Problem constructor is only ~3 % of runtime; the pre-profiling hypothesis that per-increment rebuilds dominate was wrong.
