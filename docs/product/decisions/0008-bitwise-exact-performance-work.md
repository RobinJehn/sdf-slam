---
id: 0008
title: Performance work is bitwise-exact or ensemble-certified
type: architecture
owner: Robin Jehn
created: 2026-09-13
updated: 2026-09-16
requirements:
- ../PRD.md
---

# DEC-0008 — Performance work is bitwise-exact or ensemble-certified

## Context

Profiling showed the flagship run spends 96 % of its time in the solver: sparse LDLT ~55-61 %, the J^T J product ~16-19 %, Jacobian triplet merging ~10-13 %, Cost() ~9-11 %. Faster linear solvers (CHOLMOD supernodal LLT) and any parallel reduction change the floating-point rounding of the LM steps.

An experiment settled how much rounding freedom exists on a boundary-sitting config: a 1e-9 relative change to lambda_init collapses the lap-1 revisit median from 0.013 m to 0.687 m. CHOLMOD (a different elimination ordering, same mathematics) gives 0.782 m. The incremental recipe (DEC-0007) runs a fixed iteration budget per increment, never converges tightly, and amplifies last-ulp step differences into macroscopic trajectory changes.

A rounding change is therefore a basin redraw, not a small accuracy loss: no numeric tolerance separates "bitwise" from "broken" near a basin boundary. But the chaos is config-dependent (DEC-0007): anchored configs sit deep in their basins (8/8 jittered runs, spread 0.000), and DEC-0009 provides a statistical gate that certifies any perturbation — a config change or a binary change — on ensembles instead of single runs. On the 150x150 reference the bitwise constraint also caps the achievable speedup: the serial LDLT elimination-tree trunk dominates, and the bitwise-exact parallel kernels gain only ~5 %.

## Decision

Performance work on the recipe path follows a two-tier policy (revised 2026-09-14):

1. **Refactor tier.** A change that claims "same computation, faster" must reproduce the reference artifacts bit for bit (poses_estimated.csv and map.txt on the lap-1 testbed, then on a full reference run). The bitwise diff is the complete and cheap verification; no ensemble is needed. Run bitwise checks on the Eigen-LDLT path (the `linear_solver` default): the CHOLMOD path cannot reproduce itself (see Consequences), so it cannot anchor a bitwise comparison.
2. **Reordering tier.** A change that reorders floating-point arithmetic (different linear solver, parallel reductions, changed contraction) is admissible when the new binary re-passes the full DEC-0009 gate on each production config: an 8-jitter lap-1 ensemble with 8/8 convergence and near-zero spread, then at least 3 lambda-jittered full runs scored on the frozen ICP relations and the rendered map, with the median inside the reference band (0.052-0.062 m stage-1). Certification binds to the (binary, config) pair: a new production config or a further reordering change repeats the gate.

Blanket numeric tolerances stay forbidden: the gate is statistical, never "the metric moved by less than X on one run".

Refactor-tier techniques, all landed:

- Cache Cost() values in the LM loop instead of recomputing on bitwise-unchanged state (3 calls per iteration down to 1).
- Parallelize per-element loops whose outputs are independent slots: scan normals, node normals, point and Eikonal row assembly.
- Replicate Eigen kernels operation for operation and split them across threads along boundaries that do not reorder any accumulation: the J^T J product (per-column accumulation order preserved), the -J^T r product (Eigen's two-accumulator kernel replicated exactly), and the simplicial LDLT numeric factorization (`src/solvers/parallel_ldlt.hpp`) scheduled over independent elimination-tree subtrees.
- Pin floating-point contraction where the compiler has a choice: the LDLT kernel uses std::fma exactly where the reference build fused (shift term, y-updates) and an unfused product where it did not (diagonal update), with FP_CONTRACT OFF in the kernel.

Unit tests assert bitwise equality of ParallelSimplicialLdlt against Eigen's SimplicialLDLT across thread counts.

Reordering-tier results:

- CHOLMOD supernodal LLT (`solver.linear_solver: cholmod`) is CERTIFIED on the anchored reference (2026-09-14): lap-1 ensemble 8/8 at 0.011 with spread 0.000; three lambda-jittered full runs at ICP-relations median 0.053-0.055 m (reference band 0.052-0.062), means 0.115-0.129, healthy maps. Stage 1 drops from ~34 to ~17-18 min and the warm 200x200 polish from ~2 min to ~60 s (jitter-stable to the last digit, median 0.053). CHOLMOD is the reference solver since 2026-09-15 (`configs/intel_smooth_gradient_150.yaml`, `configs/intel_polish_200.yaml`); it also matches the original dissertation repo's solver. The 2026-09-13 rejection (0.782 m) measured a boundary-sitting config; deep in the anchored basin the reordering is harmless.
- Trunk-parallel LDLT factorization: open candidate; less urgent now that CHOLMOD removes the serial-trunk bottleneck.

## Consequences

- The CHOLMOD path is nondeterministic per invocation (measured 2026-09-16: identical binary and config produce trajectories that differ from frame 1; absolute poses differ up to 13 m through the soft anchor mode while held-out relation scores stay in band, means 0.106-0.111 over five draws). The threaded BLAS inside the supernodal factorization reorders reductions between runs. Every reference run is therefore an independent basin draw even without a lambda jitter; report reference results only as ensemble statistics, and never expect a rerun to reproduce a CSV. The perturbation itself is tiny — the amplification is what differs by run length. Two reruns of the 200-scan ACES gate slice diverge by at most 2.7 mm (mean 0.6 mm), while a 910-scan Intel run reaches metres: the soft anchor mode (DEC-0007) grows the deviation along the chain, so short slices stay reproducible in practice and full runs do not.
- The recipe's headline numbers are knife-edge samples, not robust properties. Any dissertation-comparison claim needs this caveat; a different compiler, BLAS, or machine will produce different metrics from the same config unless the config passes the gate there too.
- A reordering-tier change costs a gate run per production config (~20 min lap-1 ensemble plus ~2 h of jittered full runs). Prefer refactor-tier wins for quick iterations.
- Symbolic-analysis reuse in the LDLT is near-useless here (the J^T J pattern changes in ~99 % of iterations because points cross cells), so the win comes from parallelizing the numeric factorization, not from caching the analysis.

## Alternatives considered

- Bitwise-exactness as the only admissible standard (was the decision 2026-09-13, relaxed 2026-09-14): safe but caps the reference-config speedup at ~5 % because the LDLT trunk stays serial; the DEC-0009 gate now certifies reorderings statistically.
- Metric-equivalent tolerance for perf changes (rejected): no numeric tolerance exists between "bitwise" and "broken" near a basin boundary; single-run comparisons cannot certify a reordering. The reordering tier replaces the tolerance with an ensemble gate.
- CHOLMOD supernodal LLT as the default (tried, rejected 2026-09-13 on the old flagship config; certified and adopted 2026-09-15): the 0.782 m collapse came from a config the gate later classified as boundary-sitting.
- Per-increment problem caching (rejected as primary lever): the Problem constructor is only ~3 % of runtime; the pre-profiling hypothesis that per-increment rebuilds dominate was wrong.
