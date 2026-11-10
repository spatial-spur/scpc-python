# Spatial SPUR & SCPC Ecosystem: Comprehensive Performance Optimization & Handoff Summary

This document provides a comprehensive technical synthesis of all optimization work, algorithmic breakthroughs, bug fixes, benchmarks, and cross-language architecture across the **`spatial-spur`** ecosystem. It is structured as an exhaustive reference and project handoff for remote engineering and econometrics teams.

---

## 1. Project Background & Methodological Scope

The `spatial-spur` ecosystem implements modern econometric tools for spatial regression analysis, low-frequency spatial unit root diagnostics, and spatial correlation-robust inference developed by **Ulrich K. Müller and Mark W. Watson**, with practitioner guidelines and applied workflows by **Sascha O. Becker, P. David Boll, and Hans-Joachim Voth**:
- **Spatial Unit Roots & Transformations (`spuR`, `spur-stata`, `spur-python`):** Implements tests for low-frequency spatial persistence ($I(0)$ vs. $I(1)$ nulls), residual-based variants for regressions, half-life spatial confidence intervals, and spatial transformations (LBM-GLS, nearest-neighbor, isotropic, and spatial clustering) from Müller & Watson (2024, *Econometrica*) and Becker, Boll, & Voth (2026, *The Stata Journal*).
- **Spatial Correlation-Predictive Convexity (`scpcR`, `ukmueller/SCPC`, `scpc-python`):** Implements C-SCPC and SCPC methods for spatial inference and confidence intervals robust to general spatial dependence, including unclustered and clustered geographic coordinates, from Müller & Watson (2021, *J. Econometrics*; 2022/2023, *JBES*).

---

## 2. Global Repository & Directory Map

All active codebases, benchmarks, and baseline implementations are organized locally as follows:

```
/Users/pauldavidboll/.gemini/antigravity/scratch/
├── spuR/                    # Base R version of spuR (pure C/C++ via zero-dependency .Call)
├── spuR_rcpp/               # Multi-threaded R package (Rcpp + RcppArmadillo + RcppParallel)
├── scpcR/                   # Base R version of scpcR (pure C/C++ via zero-dependency .Call)
├── scpcR_rcpp/              # Multi-threaded R package (Rcpp + RcppArmadillo + RcppParallel)
├── spur-stata/              # Stata/Mata package (active branch: perf-and-fixes, commit a731355)
├── spur-stata-baseline/     # Unmodified upstream baseline of spur-stata for parity checking
├── scpc_benchmark/          # Stata SCPC optimization suite
│   ├── baseline/            # Unmodified upstream ukmueller/SCPC (v1.0.0)
│   ├── scpc_v2/             # Pure-Mata accelerated package (v2.1.0)
│   └── scpc_plugin/         # C-plugin C++ prototype for Stata
├── scpc-python/             # Python implementation of SCPC (uv-managed, pytest)
└── spur-python/             # Python implementation of SPUR (uv-managed, pytest)
```

---

## 3. Package-by-Package Deep-Dive

### 3.1. `spuR` (R Package for Spatial Unit Roots)
* **Referenced Conversation:** [`Performance Optimization For SpuR`](conversation://7a45aaf2-e89d-4097-a7e3-5572615d08c6)
* **Repositories:** [`spuR`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/spuR) & [`spuR_rcpp`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/spuR_rcpp)

#### Algorithmic & Mathematical Breakthroughs:
1. **Low-Rank Pre-Projection for Residual Tests (`spurtest_i0resid`, `spurtest_i1resid`):**
   - *Problem:* The naïve algorithm formed the dense $n \times n$ projection matrix $\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ and computed $\mathbf{M} \mathbf{\Sigma}(c) \mathbf{M}'$ repeatedly across 30 grid points. At $n = 3,000$, $\mathbf{M}$ is 72 MB, requiring $54\text{ GFlops}$ per step and taking >8 minutes.
   - *Solution:* Exploited low-rank associativity:
     $$\mathbf{R}' (\mathbf{M} \mathbf{\Sigma} \mathbf{M}') \mathbf{R} = (\mathbf{M} \mathbf{R})' \mathbf{\Sigma} (\mathbf{M} \mathbf{R}) = \tilde{\mathbf{R}}' \mathbf{\Sigma} \tilde{\mathbf{R}}$$
     Precomputing the $n \times q$ projected basis $\tilde{\mathbf{R}} = \mathbf{R} - \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'\mathbf{R})$ once in $O(n k q)$ reduces each grid point evaluation to $O(n^2 q)$ with $q=15$, completely eliminating dense $n \times n \times n$ matrix products.
2. **Convexity Sandwich Solver for `get_cbar`:**
   - Inverted the average correlation equation $\frac{2}{n(n-1)} \sum_{i < j} \exp(-c \cdot d_{ij}) = \bar{\rho}$ without sorting $4.5\times 10^6$ distance pairs or performing continuous bisection on raw pairwise arrays. Implemented a 1024-bin histogram with linear interpolation and Jensen's inequality convexity bounds, shrinking runtime from $>0.40\text{ s}$ to $<0.003\text{ s}$ per call.
3. **Tiled Covariance Accumulation (`rcpp_om_grid`):**
   - Evaluates $\mathbf{R}' \mathbf{\Sigma}(c) \mathbf{R}$ across the parameter grid without materializing the dense $n \times n$ matrix $\mathbf{\Sigma}(c)$. Computes distances on-the-fly in L2-cache friendly tiles ($256 \times 256$) and accumulates directly into $q \times q$ blocks.
4. **Lock-Free Multi-Threading (`RcppParallel`):**
   - The 131 null grid points in `spurhalflife` and 30 grid points in `spurtest_i0*` are embarrassingly parallel. Implemented with Intel TBB via `RcppParallel::parallelFor`. Thread writes are strictly disjoint, yielding **100% bitwise determinism** regardless of thread count.

#### Verification & Benchmarks:
- **Unit Tests:** 100/100 pass (`PASS 100 | FAIL 0 | SKIP 11`).
- **Precision:** Bitwise identical on `spurhalflife`; test statistics match baseline to $< 4.6 \times 10^{-16}$ (machine epsilon).
- **Runtimes across Sample Sizes:**
  | Sample Size ($n$) | Routine | Baseline R | Optimized `.Call` (1 thread) | `RcppParallel` (Multi-core) | Net Speedup vs Baseline |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | **$n = 1,000$** | `spurtest_i0resid` | 13.91 s | 0.20 s | **0.13 s** | **107.0×** |
  | | `spurhalflife` | 19.34 s | 1.77 s | **0.55 s** | **35.2×** |
  | **$n = 2,000$** | `spurtest_i0resid` | 148.97 s | 0.42 s | **0.26 s** | **568.6×** |
  | | `spurtest_i1resid` | 26.81 s | 0.25 s | **0.22 s** | **120.2×** |
  | | `spurhalflife` | 100.69 s | 2.28 s | **0.80 s** | **126.2×** |
  | **$n = 3,000$** | `spurtest_i0resid` | 486.89 s | 0.74 s | **0.47 s** | **1,035.9×** |
  | | `spurtest_i1resid` | 161.54 s | 0.55 s | **0.54 s** | **300.3×** |
  | | `spurhalflife` | — | 3.23 s | **1.22 s** | **>200×** |

---

### 3.2. `scpcR` (R Package for Spatial Correlation-Predictive Convexity)
* **Referenced Conversation:** [`Performance Gains For ScpcR`](conversation://e15c3391-d06c-47a4-9549-48eeaf0e295d)
* **Repositories:** [`scpcR`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/scpcR) & [`scpcR_rcpp`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/scpcR_rcpp)

#### Algorithmic & Engineering Highlights:
1. **CRAN Policy & Resource Compliance (`_R_CHECK_LIMIT_CORES_`):**
   - Implemented thread getters/setters (`scpc_set_num_threads()` / `spur_set_num_threads()`) mutating R package options (`options(scpcR.threads = N)`), avoiding forbidden `Sys.setenv()` process-wide mutations.
   - When running on CRAN check farms (`_R_CHECK_LIMIT_CORES_` detected), worker threads are automatically throttled to $\le 2$ cores, guaranteeing automated CRAN pass without disabling tests.
   - Single-thread bypass: passing `nthreads = 1` calls the worker sequentially on the calling thread, completely bypassing Intel TBB task allocation overhead.
2. **Preservation of Müller's Large-$N$ Approximation ($N \ge 4,500$):**
   - Replicated Ulrich Müller's exact Stata Nyström approximation:
     - PCA coordinate rotation, reflection, and lexicographical sorting (`.normalize_s`).
     - Stata-identical Linear Congruential PRNG ($64389 \pmod{2^{32}}$) in `.next_u`.
     - Subsampled basis evaluation ($N_{\text{sub}} = 20, m = 1,000$) to estimate spatial bandwidth $c_0, c_{\max}$ without $O(n^2)$ matrices.
     - Monte Carlo covariance integration ($M = 1,000,000$ pairwise evaluations).
3. **Memory Wall Justification for Retaining Large-$N$:**
   - Why not drop Large-$N$ in C++?
     - Dense pairwise distance matrices scale as $O(n^2)$: $N = 10,000 \implies 800\text{ MB}$; $N = 20,000 \implies 3.2\text{ GB}$; $N = 50,000 \implies 20\text{ GB}$; $N = 100,000 \implies 80\text{ GB}$.
     - Large-$N$ mode operates in $O(n)$ spatial memory ($\approx 20\text{–}40\text{ MB}$ even at $N = 100,000$), enabling analysis on large census/micro datasets without out-of-memory crashes.
   - Routing: `method = "auto"` automatically routes $n < 4,500$ to exact mode and $n \ge 4,500$ to approx mode (with clustered data using $n = N_{\text{clusters}}$). Users can override with `method = "exact"` or `method = "approx"`.

#### Verification:
- `scpcR_rcpp`: Full test suite passes 288 assertions (`PASS 288 | FAIL 0 | SKIP 6 | WARN 1`).
- Exact replication verified against Stata at $n = 4,600$ across unconditional and conditional IV models.

---

### 3.3. `ukmueller/SCPC` & `scpc_v2` (Stata/Mata Implementation)
* **Referenced Conversation:** [`SCPC Performance Gains Analysis`](conversation://33f97f2f-5499-4c95-a230-2e6d157ed345)
* **Repositories & Suites:**
  - Upstream baseline: [`scpc_benchmark/baseline`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/scpc_benchmark/baseline)
  - Pure-Mata v2.1.0: [`scpc_benchmark/scpc_v2`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/scpc_benchmark/scpc_v2)
  - C-Plugin prototype: [`scpc_benchmark/scpc_plugin`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/scpc_benchmark/scpc_plugin)

#### Key Insights & Algorithmic Optimizations:
1. **Mathematical Explanation of "Identical Critical Values":**
   - *Observation:* On synthetic normal regressors at $N = 10,000$, `scpc_v2` reported identical critical values for `x1`, `x2`, and `_cons`.
   - *Theoretical Basis:* In Müller & Watson (2021/2022, Eq. 14):
     $$c_{\text{SCPC}, j}(\alpha) = \max\left( c_{\text{uncond}}(\alpha),\; c_{\text{cond}, j}(\alpha) \right)$$
     When regressors are standard Gaussian noise without spatial dependence, conditioning on them produces $c_{\text{cond}, j} \le c_{\text{uncond}}$. The $\max$ operator enforces the theoretical unconditional lower bound.
   - *Why Baseline Hid This:* The upstream baseline ran a separate PRNG seed sequence per covariate with Monte Carlo sampling noise ($M = 10^6$) and a loose bisection bracket ($10^{-3}$), causing values to jitter around $2.698 \dots 2.706$. In exact mode at $N = 722$, the upstream baseline also produced identical critical values down to the 6th decimal place.
2. **Pure-Mata Numerical Acceleration in `scpc_v2`:**
   - **64-point Gauss-Legendre Quadrature + Brent's Root Finder ($10^{-8}$):** Replaced coarse 20-point bisection, eliminating numerical jitter and solving critical values to full precision.
   - **Unconditional Bound Caching:** Computes and caches $c_{\text{uncond}}$ once rather than re-solving per variable.
   - **Vectorized LCG PRNG in Mata:** Vectorized pseudo-random index generation in `raninds` and pairwise distance lookups.
   - **Eliminated Macro/Mata Overhead:** Replaced Stata command-line macro passing with direct Mata memory structures.

#### Benchmarks (StataNow 19.5 SE):
| Configuration / $N$ | Upstream Baseline | `scpc_v2` Pure Mata | Speedup | Numerical Difference |
| :--- | :---: | :---: | :---: | :---: |
| $N = 200$ (Exact) | 0.179 s | **0.094 s** | **1.90×** | $0.000$ (Exact) |
| $N = 722$ (Exact) | 0.580 s | **0.288 s** | **2.01×** | $0.000$ (Exact) |
| $N = 1,500$ (Exact) | 2.424 s | **1.146 s** | **2.12×** | $0.000$ (Exact) |
| $N = 3,000$ (Exact) | 11.718 s | **5.559 s** | **2.11×** | $0.000$ (Exact) |
| $N = 5,000$ (Approx) | 40.567 s | **21.161 s** | **1.92×** | $0.000$ (Exact) |
| $N = 10,000$ (Approx) | 42.866 s | **20.654 s** | **2.08×** | Within MC tolerance |
| $N = 5,000$ (Exact Forced) | — (OOM/slow) | **35.796 s** | Feasible | Exact |

---

### 3.4. `spur-stata` (Stata/Mata Package for Spatial Unit Roots)
* **Referenced Conversation:** [`Evaluating Spur Stata Code Efficiency`](conversation://db6441e9-123a-4237-9e46-970dc4266972)
* **Repository:** [`spur-stata`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/spur-stata) (Branch: `perf-and-fixes`, Commit `a731355`)

#### Critical Inaccuracies & Bugs Fixed:
1. **Pass-by-Reference In-Place Mutation Trap in [`demeanmat.mata`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/spur-stata/mata/demeanmat.mata):**
   - *Bug:* Mata passes all function arguments by reference. The statement `mat = mat :- rowsum(mat)/cols(mat)` in `demeanmat(real matrix mat)` directly altered the caller's matrix in memory. When applied to distance matrices, the zero diagonal was mutated to negative numbers (e.g. $-0.55$). Subsequent $\exp(-c \cdot \mathbf{D})$ calls evaluated $\exp(c \cdot 0.55)$ up to $10^{163}$, causing numerical overflow and corrupted test statistics.
   - *Fix:* Replaced with local matrix allocation: `real matrix res = mat :- ...; return(res)`.
2. **Unconditional Earth Great-Circle Scaling in [`spurhalflife.ado`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/spur-stata/spurhalflife.ado):**
   - *Bug:* Lines 42–47 unconditionally scaled `max_dist` by $2\pi R_{\text{earth}} \approx 40,030,173\text{ m}$ even when coordinates were Euclidean (i.e. `latlong` was omitted), inflating Euclidean half-life confidence intervals by a factor of 40 million.
   - *Fix:* Wrapped with conditional check: `if "\`latlong'" != ""`.
3. **Dimension Safety & Vector Orientation:**
   - In `get_R.mata`: Ensured eigenvalues assigned to row vectors are consistently transposed to avoid Mata type/dimension errors.
   - In `lvech.mata`: Standardized lower-triangular vectorization.
4. **Single-Pass Operator Caching in `spurtransform.ado`:**
   - When transforming multiple variables (e.g., $y, x_1, x_2, \dots$), the baseline recomputed the entire distance matrix and eigensolver for every single variable. Cached the transformation operator $\mathbf{H}$ across all variables in the run.
5. **Recompiled Mata Library:**
   - Recompiled all 33 Mata source files into [`lspur.mlib`](file:///Users/pauldavidboll/.gemini/antigravity/scratch/spur-stata/lspur.mlib) (342 KB).

#### Benchmarks ($N = 1,500, n_{\text{rep}} = 10,000$):
| Routine | Baseline Runtime | Optimized Runtime | Speedup | Parity Difference |
| :--- | :---: | :---: | :---: | :---: |
| `spurtest i0resid` | 38.37 s | **9.65 s** | **4.0×** | $< 4.3 \times 10^{-14}$ |
| `spurtransform` (3 variables) | 11.06 s | **3.28 s** | **3.4×** | $0.00 \times 10^{00}$ |
| `spurtest i1resid` | 8.64 s | **4.27 s** | **2.0×** | $< 7.5 \times 10^{-14}$ |
| `spurtest i0` | 11.03 s | **8.75 s** | **1.3×** | $< 1.0 \times 10^{-14}$ |
| `spurhalflife` | 17.91 s | **13.50 s** | **1.3×** | $0.00 \times 10^{00}$ |
| `spurtest i1` | 3.26 s | **3.20 s** | **1.0×** | $< 1.0 \times 10^{-14}$ |
| **Total Test Suite** | **90.26 s** | **42.64 s** | **2.1× overall** | Exact |

#### Pending / Identified Next Steps in `spur-stata` for $n_{\text{rep}} = 100,000$:
Profiling in `compute_spur_persistence.mata` revealed 3 high-impact wins for default $n_{\text{rep}} = 100,000$:
1. **Intermediate Matrix Associativity:**
   Replace `xc = ch_omi * (ch_null' * emat)` with `xc = (ch_omi * ch_null') * emat`. The product `(ch_omi * ch_null')` is only $15 \times 15$, avoiding a $15 \times 15 \times 100,000$ matrix multiplication.
2. **Eliminate 40 MB Transpose:**
   Replace `mean(den_ha_mat')'` ($100,000 \times 50$) with `rowsum(den_ha_mat) / cols(den_ha_mat)`.
3. **Remove Dead Quantile Computation:**
   In the 131-iteration loop of `c_ci`, `cvalue = mm_quantile(lr, 1, (1 :- sz_vec'))` is computed on 100,000 elements (costing ~9 seconds), but `cv_mat` is **never returned or used** (only `pv_mat` is returned). Removing this dead code saves ~9s per `spurhalflife` call.

---

### 3.5. `scpc-python` & `spur-python` (Python Implementations)
* **Referenced Conversation:** [`Python Version Performance Optimization`](conversation://98c475e3-b7e6-4e95-afff-c9f87e5318c2)
* **Strategy Document:** [`python_optimization_strategy.md`](file:///Users/pauldavidboll/.gemini/antigravity/brain/98c475e3-b7e6-4e95-afff-c9f87e5318c2/python_optimization_strategy.md)
* **Repositories:** [`scpc-python`](file:///Users/pauldavidboll/Github/scpc-python) & [`spur-python`](file:///Users/pauldavidboll/Github/spur-python)

#### Profiling Diagnostics & Bottlenecks:
1. **`spurhalflife` Severe Scaling Bottleneck:**
   - Current Python runtime: **17.64 s at $n = 500$**, **~75 s at $n = 1,000$**, and **~5 minutes at $n = 2,000$** (vs. **0.80 s in RcppParallel**).
   - *Cause:* Nested Python loops over $131 \times 50$ grid points, reallocating 181 separate full $n \times n$ covariance matrices, running under Python's single-threaded Global Interpreter Lock (GIL).
2. **Dense $O(n^3)$ Residual Projection Trap:**
   - `spurtest_i0resid` and `spurtest_i1resid` in `spur/core.py` construct dense $M = I - X(X'X)^{-1}X'$ and compute $M @ \Sigma(c) @ M^T$. At $n = 3,000$, this takes ~3 minutes.
3. **`get_cbar` Repeated Exponentiation:**
   - Bisection on $n(n-1)/2$ distances evaluates `np.exp(-c * dist)` on millions of pairs up to 30 times per call.
4. **`scpc-python` Large-$N$ Python Loop Overhead:**
   - Subsampling ($N=20, m=1,000$) and Monte Carlo evaluation ($M=1,000,000$) run serial Python `for` loops in `raninds` and distance lookups.

#### Recommended Two-Tier Implementation Architecture:
```
           Public Python API (scpc, spurtest, spurhalflife)
                                |
                 [Does C++ extension exist?]
                     /                    \
                   YES                     NO
                    v                       v
     Native C++ Acceleration         Optimized NumPy Fallback
   (nanobind + Eigen + std::thread)   (Vectorized + Low-Rank Basis)
        [100x - 1,000x Speedup]             [5x - 20x Speedup]
```

- **Phase 1: Pure-NumPy Algorithmic Wins (Immediate, No Compiler Required):**
  - Implement low-rank basis $\tilde{R} = MR$, cutting $O(n^3)$ to $O(n^2 q)$.
  - Pre-allocate arrays and vectorize quadratic-form evaluations in `spur/utils/inference.py`.
  - Vectorize `raninds` in `scpc/utils/spatial.py`.
- **Phase 2: Compiled C++ Acceleration via `nanobind`:**
  - Directly port verified C++ routines from `spuR_rcpp` and `scpcR_rcpp` (`cpp_c_ci_mc`, `cpp_om_grid`, `cpp_lnset_wc0`, `cpp_lnget_oms`).
  - Use `nb::gil_scoped_release` and modern C++17 `std::thread` pool (avoiding OpenMP / `libomp` deployment hurdles on macOS).
- **Phase 3: Automated Wheel Building (`cibuildwheel`):**
  - Use `scikit-build-core` in `pyproject.toml` with GitHub Actions for cross-platform wheels (Linux, macOS arm64/x86_64, Windows).

---

## 4. Cross-Language Equivalence & Benchmark Synthesis

Below is an overview of performance across Stata, R, and Python:

| Task / Routine | Scale ($N$) | Stata Baseline | Stata `v2` / `perf-fixes` | Base R (`.Call`) | R (`RcppParallel`) | Python Baseline | Python Phase 2 (Target) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`spurtest_i0resid`** | $n = 1,000$ | 16.5 s | **4.2 s** | 0.20 s | **0.13 s** | 2.8 s | **~0.15 s** |
| | $n = 2,000$ | ~75 s | **18.5 s** | 0.42 s | **0.26 s** | ~45 s | **~0.30 s** |
| | $n = 3,000$ | ~180 s | **42.0 s** | 0.74 s | **0.47 s** | ~180 s | **~0.50 s** |
| **`spurhalflife`** | $n = 500$ | 5.2 s | **3.8 s** | 0.85 s | **0.30 s** | 17.6 s | **~0.30 s** |
| | $n = 1,000$ | 12.1 s | **8.5 s** | 1.77 s | **0.55 s** | 75.0 s | **~0.55 s** |
| | $n = 2,000$ | 28.5 s | **18.2 s** | 2.28 s | **0.80 s** | ~300 s | **~0.80 s** |
| **`scpc` (Exact)** | $N = 722$ | 0.58 s | **0.29 s** | 0.04 s | **0.02 s** | 0.35 s | **~0.03 s** |
| | $N = 3,000$ | 11.72 s | **5.56 s** | 0.32 s | **0.11 s** | 2.10 s | **~0.12 s** |
| **`scpc` (Large-$N$)**| $N = 5,000$ | 40.57 s | **21.16 s** | 0.22 s | **0.09 s** | 14.5 s | **~0.10 s** |
| | $N = 10,000$ | 42.87 s | **20.65 s** | 0.38 s | **0.14 s** | ~28 s | **~0.15 s** |

---

## 5. Immediate Action Items & Recommendations for Remote Project

1. **`spur-stata`**:
   - Merge branch `perf-and-fixes` into `main` after reviewing commit `a731355`.
   - Apply the three pending optimizations in `compute_spur_persistence.mata` (associative matrix multiply, `rowsum` over transpose, and removing dead `mm_quantile` loop) to drop 100k-draw runtime by an additional ~30–40%.
2. **`scpc` (Stata)**:
   - Upstream the `scpc_v2` enhancements (64-point Gauss-Legendre quadrature, Brent's root-finder, unconditional caching) to `ukmueller/SCPC`.
3. **`spuR` & `scpcR` (R)**:
   - Prepare CRAN submission for `spuR` and `scpcR`. Verify that `_R_CHECK_LIMIT_CORES_` automatically restricts threads on test rigs.
   - Choose whether to publish the zero-dependency `.Call` version or the `RcppArmadillo` + `RcppParallel` version as the primary CRAN release.
4. **`scpc-python` & `spur-python`**:
   - Execute Phase 1 (pure NumPy algorithmic wins: $\tilde{R} = MR$ preprojection, distance bisection caching, vectorized `raninds`).
   - Implement Phase 2 with `nanobind`, reusing the C++ kernels already validated in `spuR_rcpp/src/spur_rcpp.cpp` and `scpcR_rcpp/src/scpc_rcpp.cpp`.
5. **Cross-Language CI Parity Test**:
   - Construct a shared validation dataset (e.g. Chetty commuting zones with synthetic outcomes) and verify that all three language implementations (Stata, R, Python) generate test statistics and critical values matching to within $10^{-6}$.
