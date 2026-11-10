# Performance work and correctness requirements

This development branch adds documentation only. It does not implement a native Python backend. Historical cross-language timings and speedup estimates from private workstations are not reproducible from this branch and should not be used as release claims.

SCPC means **Spatial Correlation Principal Components**. SCPC and conditional SCPC are developed by Müller and Watson (2022, Econometrica 90(6): 2901–2935; 2023, Journal of Business & Economic Statistics 41(4): 1050–1064). SPUR methods concern spatial unit-root inference and transformations.

## Mathematical contracts

The exponential working covariance is `Sigma(c)[i,j] = exp(-c * d[i,j])`. Its scale parameter solves the requested average **off-diagonal** correlation. Parameter solvers must be invariant to coordinate units and must identify unattainable targets when co-located pairs impose a positive correlation floor.

For a basis `R` and residual-maker `M`, the identity

`R.T @ M @ Sigma @ M.T @ R = (M.T @ R).T @ Sigma @ (M.T @ R)`

permits small projected covariance calculations. A QR basis for the design matrix can apply the residual projection without forming a dense `M` or solving normal equations. Symmetry of the usual OLS residual-maker simplifies `M.T` to `M`.

For sampled off-diagonal covariance sums, the diagonal contribution is `W.T @ W`. Replacing it with an identity is valid only for an orthonormal basis; conditioning on regressors generally changes the basis.

A geographic Nyström extension must use the same great-circle metric as the covariance used to construct its subsample eigenvectors. Subsamples must be drawn without replacement while retaining the complete population for subsequent draws.

Spatial half-life is a distance. Euclidean bounds stay in the units of the coordinates; Earth-radius conversion applies only to geographic distances. Angular likelihood inference must be invariant to scaling the observed signal.

## Performance considerations

Low-rank residual projection, reused decompositions, tiled kernels, and removal of unused simulation summaries are candidates for measurement. These are algorithmic opportunities, not benchmark results for this Python branch.

NumPy and BLAS kernels can release the GIL. Python orchestration overhead and repeated dense matrix products are separate costs; a C++ extension alone does not establish which cost dominates. Profile representative workloads before selecting an implementation.

An exact method retains at least one `n x n` distance matrix. Approximate methods retain spatial bases and subsample buffers; their memory constants depend on basis size, subsample size, and concurrent workers. Report peak resident memory alongside elapsed time and constrain both worker and BLAS thread counts when benchmarking.

## Evidence required for future changes

Use analytic special cases such as the Student-t identity for SCPC probabilities, direct defining matrix equations on small examples, coordinate translation and unit invariance, outcome-scale invariance, and simulation with stated Monte Carlo uncertainty. Include geographic seams, antipodes, repeated locations, rank-deficient designs, and failed calibration brackets.

A tighter root tolerance does not eliminate Monte Carlo noise. A finite covariance grid does not by itself prove size control over a continuous parameter class. Compare against upstream code to diagnose differences, but accept or reject an implementation on theoretical and numerical grounds.

Any performance report should record the exact commit, data generation, random seed, hardware, compiler, numerical libraries, thread settings, repeated-run timings, and peak memory. Existing upstream benchmark numbers do not validate a corrected implementation.
