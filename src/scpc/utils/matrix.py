from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd
from scipy.sparse.linalg import ArpackError, eigsh

from ..types import ArrayLike, MatrixLike


def has_tied_eigenvalues(values: ArrayLike) -> bool:
    """Return True if consecutive descending eigenvalues exhibit a numerical tie."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 2:
        return False
    tol = 1.6e-11 * max(abs(float(vals[0])), 1.0)
    return bool(np.any(vals[:-1] - vals[1:] <= tol))


def lvech(mat: MatrixLike) -> ArrayLike:
    """Collect the strict lower triangle of a matrix.

    This helper turns a square matrix into the vector of pairwise entries
    below the diagonal. In SCPC, that compact form is useful when the spatial
    engine needs a one-dimensional view of all pairwise distances.

    Args:
        mat: Matrix whose lower-triangular entries should be extracted.

    Returns:
        The strict lower-triangular entries as a vector-like object.
    """
    mat = np.asarray(mat, dtype=float)
    return mat[np.tril_indices(mat.shape[0], k=-1)]


def demeanmat(mat: MatrixLike) -> MatrixLike:
    """Remove row and column averages from a matrix.

    This helper converts a matrix into its double-demeaned version. In SCPC,
    that matters because the spatial covariance objects are centered before
    their principal directions are extracted, so the projection basis reflects
    spatial variation rather than a raw level shift.

    Args:
        mat: Matrix to double-demean.

    Returns:
        The double-demeaned matrix.
    """
    mat = np.asarray(mat, dtype=float)

    # remove row means first, then column means from the row-demeaned matrix
    mat = mat - np.mean(mat, axis=1, keepdims=True)
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    return mat


def get_w_with_ties(
    distmat: MatrixLike, c0: float, qmax: int
) -> tuple[MatrixLike, bool]:
    """Build the candidate spatial projection basis and detect leading ties."""
    distmat = np.asarray(distmat, dtype=np.float64)
    n = distmat.shape[0]
    if n < 2 or qmax < 1 or qmax >= n:
        raise ValueError("SCPC basis requires n >= 2 and 1 <= qmax < n.")

    m = n - 1
    root_n = math.sqrt(n)
    sig = np.exp(-c0 * distmat)
    row_sums = np.sum(sig, axis=1)
    proj = (row_sums / root_n + sig[:, m]) / (root_n + 1.0)
    common = (float(np.sum(proj)) / root_n + float(proj[m])) / (root_n + 1.0)
    cov = sig[:m, :m] + common - proj[:m, None] - proj[None, :m]

    k_check = min(qmax + 1, m)
    vecs_sub: np.ndarray | None = None
    evals_check: np.ndarray | None = None
    if n > 512 and k_check < m:
        try:
            eigvals, vecs = eigsh(cov, k=k_check, which="LA")
            order = np.argsort(eigvals)[::-1]
            evals_check = eigvals[order]
            if not has_tied_eigenvalues(evals_check):
                vecs_sub = vecs[:, order[:qmax]]
        except ArpackError:
            vecs_sub = None

    if vecs_sub is None or evals_check is None:
        eigvals, vecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        evals_check = eigvals[order[:k_check]]
        vecs_sub = vecs[:, order[:qmax]]

    tied = has_tied_eigenvalues(evals_check)
    col_sums = np.sum(vecs_sub, axis=0)
    w = np.empty((n, qmax + 1), dtype=np.float64)
    w[:, 0] = 1.0 / root_n
    w[:m, 1:] = vecs_sub - col_sums[None, :] / (root_n * (root_n + 1.0))
    w[m, 1:] = -col_sums / root_n
    return w, tied


def get_w(distmat: MatrixLike, c0: float, qmax: int) -> MatrixLike:
    """Build the candidate spatial projection basis.

    This helper converts a distance matrix and kernel scale into the matrix of
    spatial directions that SCPC can project onto. The resulting basis is the
    raw material that `set_final_w()` later trims down to the final size.

    Args:
        distmat: Pairwise distance matrix.
        c0: Kernel scale implied by the target average correlation bound.
        qmax: Largest projection dimension under consideration.

    Returns:
        The candidate spatial projection matrix.
    """
    return get_w_with_ties(distmat, c0, qmax)[0]


def get_tau(y: ArrayLike, w: MatrixLike) -> float:
    """Compute the SCPC t statistic for one projected direction.

    This helper turns a coefficient-specific direction into the normalized
    statistic that SCPC compares against its size-controlling critical value.

    Args:
        y: Coefficient-specific direction or influence summary.
        w: Spatial projection matrix used for the statistic.

    Returns:
        The SCPC t statistic.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    # match the SCPC normalization in the R implementation exactly
    return float(
        math.sqrt(w.shape[1] - 1)
        * np.dot(w[:, 0], y)
        / np.linalg.norm(w[:, 1:].T @ y, ord=2)
    )


def coerce_numeric_matrix(a: MatrixLike, arg: str) -> tuple[np.ndarray, bool]:
    """Convert a vector or matrix input into a numeric 2d array."""
    vec_input = np.ndim(a) == 1
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{arg} must be a vector or matrix.")
    if a.shape[0] == 0:
        raise ValueError(f"{arg} must have at least one row.")
    if not np.isfinite(a).all():
        raise ValueError(f"{arg} contains non-finite values.")
    return a, vec_input


def restore_residualizer_shape(a: np.ndarray, vec_input: bool) -> np.ndarray:
    """Restore vector inputs to a one-dimensional result."""
    return a[:, 0] if vec_input else a


def scale_design_columns(design: np.ndarray) -> np.ndarray:
    """Equilibrate nonzero design columns without changing their span."""
    design = np.asarray(design, dtype=float)
    scales = np.max(np.abs(design), axis=0)
    return np.divide(design, scales, out=np.zeros_like(design), where=scales > 0)


def orthonormal_design_span(
    design: np.ndarray, *, tol: float | None = None
) -> np.ndarray:
    """Return an orthonormal basis for the span of the design."""
    if design.shape[1] == 0:
        return design
    u, singular, _ = np.linalg.svd(scale_design_columns(design), full_matrices=False)
    relative_tol = np.finfo(float).eps * max(design.shape) if tol is None else tol
    return u[:, singular > relative_tol * singular[0]]


def iv_projector_factors(
    x_iv: MatrixLike, z_iv: MatrixLike, *, tol: float = 1e-10, demean=None
) -> tuple[np.ndarray, np.ndarray]:
    """Return equilibrated factors X and V for the IV residualizer I - XV'."""
    x, _ = coerce_numeric_matrix(x_iv, "IV residualizer X")
    z, _ = coerce_numeric_matrix(z_iv, "IV residualizer Z")
    if x.shape[0] != z.shape[0]:
        raise ValueError(
            "IV residualizer requires X and Z to have the same number of rows."
        )
    if demean is not None:
        x = demean(x, context="IV residualizer X", drop_absorbed=True)
        z = demean(z, context="IV residualizer Z", drop_absorbed=True)
    x = scale_design_columns(x)
    qz = orthonormal_design_span(z, tol=tol)
    left, singular, right = np.linalg.svd(qz.T @ x, full_matrices=False)
    if (
        singular.size < x.shape[1]
        or not singular.size
        or singular[-1] <= tol * max(1.0, singular[0])
    ):
        raise ValueError("IV residualizer is rank deficient: X'P_ZX is not full rank.")
    # X (Q_Z'X)^+ Q_Z' equals the 2SLS hat matrix without its normal equations.
    return x, (qz @ (left / singular)) @ right


def ols_design_span(
    design: MatrixLike, *, demean=None, require_full_rank: bool = False
) -> np.ndarray:
    """Prepare the OLS nuisance span for either backend."""
    design, _ = coerce_numeric_matrix(design, "OLS residualizer X")
    if demean is not None:
        design = demean(design, context="OLS residualizer X", drop_absorbed=True)
    qx = orthonormal_design_span(design, tol=1e-10 if require_full_rank else None)
    if require_full_rank and qx.shape[1] != design.shape[1]:
        raise ValueError("OLS residualizer is rank deficient.")
    return qx


def make_ols_residualizer(
    design: MatrixLike, *, demean=None, require_full_rank: bool = False
):
    """Build a conditional OLS residualizer from one equilibrated decomposition."""
    qx = ols_design_span(design, demean=demean, require_full_rank=require_full_rank)
    return make_orthogonal_residualizer(qx, demean=demean)


def make_orthogonal_residualizer(qx: np.ndarray, *, demean=None):
    """Apply an OLS nuisance basis to spatial directions."""

    def residualize(values: MatrixLike) -> np.ndarray:
        centered = np.asarray(values, dtype=float)
        if demean is not None:
            centered = demean(centered, context="OLS residualizer W")
        return centered - qx @ (qx.T @ centered)

    return residualize


def make_iv_residualizer(
    X: MatrixLike,
    Z: MatrixLike,
    tol: float = 1e-10,
    demean=None,
    transpose: bool = False,
):
    """Build the IV residualizer used by conditional SCPC.

    This mirrors `scpcR:::.make_iv_residualizer()`. It fills the same role as
    the direct QR-based 2SLS residualization step in the R code, but it takes
    already-aligned numeric matrices from pyfixest instead of formula-driven
    fixest objects.

    Args:
        X: Second-stage regressor matrix.
        Z: Instrument matrix.
        tol: Rank tolerance used in the linear solves.
        demean: Optional callable that applies the same fixed-effect demeaning
            used by the fitted model to a vector or matrix.
        transpose: Apply the transpose of the IV residual maker. Conditional
            denominator weights require this form for an oblique projection.

    Returns:
        A callable that residualizes a vector or matrix against the IV
        nuisance space.

    Raises:
        ValueError: If the IV design is malformed or rank deficient.
    """
    u, v = iv_projector_factors(X, Z, tol=tol, demean=demean)
    left, right = (v, u) if transpose else (u, v)

    def residualize(Y: MatrixLike) -> np.ndarray:
        Y, vec_input = coerce_numeric_matrix(Y, "IV residualizer Y")
        if Y.shape[0] != u.shape[0]:
            raise ValueError(
                "IV residualizer received Y with an incompatible row count."
            )

        if demean is not None:
            Y = np.asarray(demean(Y, context="IV residualizer Y"), dtype=float)

        return restore_residualizer_shape(Y - left @ (right.T @ Y), vec_input)

    return residualize


def validate_conditional_denominator(before: np.ndarray, after: np.ndarray) -> None:
    """Reject conditional spatial bases with no informative denominator direction."""
    pre_max = float(np.max(np.linalg.norm(before, axis=0)))
    post = np.linalg.norm(after, axis=0)
    if (
        not math.isfinite(pre_max)
        or pre_max == 0.0
        or not np.isfinite(post).all()
        or np.all(post <= 1e-12 * pre_max)
    ):
        raise ValueError("Conditional projection collapsed all non-constant W columns.")


def orthogonalize_w_iv(
    wfin: MatrixLike,
    xj: ArrayLike,
    xjs: ArrayLike,
    residualize,
) -> MatrixLike:
    """Build the conditional IV projection matrix for non-clustered SCPC.

    Apply the transposed IV residual maker to the regressor-weighted spatial
    directions, then scale by the normalized coefficient influence direction.

    Args:
        wfin: Spatial projection matrix.
        xj: Coefficient-specific influence direction.
        xjs: Normalized sign or scaling vector for `xj`.
        residualize: Residualizer returned by `make_iv_residualizer()`.

    Returns:
        The orthogonalized conditional IV projection matrix.
    """
    wfin = np.asarray(wfin, dtype=float)
    xj = np.asarray(xj, dtype=float)
    xjs = np.asarray(xjs, dtype=float)

    if len(xj) != wfin.shape[0]:
        raise ValueError(
            "Conditional IV projection received xj with an incompatible length."
        )
    if len(xjs) != wfin.shape[0]:
        raise ValueError(
            "Conditional IV projection received xjs with an incompatible length."
        )
    if not np.isfinite(xj).all() or not np.isfinite(xjs).all():
        raise ValueError(
            "Conditional IV projection received non-finite xj or xjs values."
        )

    wx = wfin.copy()
    wx[:, 0] = wfin[:, 0] * xj * xjs

    if wfin.shape[1] > 1:
        y = wfin[:, 1:] * xj[:, None]
        r = np.asarray(residualize(y), dtype=float)
        wx[:, 1:] = r * xjs[:, None]
        validate_conditional_denominator(y, wx[:, 1:])

    if not np.isfinite(wx).all():
        raise ValueError("Conditional IV projection produced non-finite W values.")

    return wx


def orthogonalize_w_cluster_iv(
    wfin: MatrixLike,
    cl_vec: ArrayLike,
    xj_indiv: ArrayLike,
    residualize,
) -> MatrixLike:
    """Build the conditional IV projection matrix for clustered SCPC.

    Normalize individual influence directions within clusters, apply the
    transposed IV residual maker to regressor-weighted spatial directions,
    and aggregate the resulting weights back to the cluster level.

    Args:
        wfin: Cluster-level spatial projection matrix.
        cl_vec: Cluster membership for each individual observation.
        xj_indiv: Individual-level influence direction.
        residualize: Residualizer returned by `make_iv_residualizer()`.

    Returns:
        The cluster-level conditional IV projection matrix.
    """
    wfin = np.asarray(wfin, dtype=float)
    cl_vec = np.asarray(cl_vec)
    xj_indiv = np.asarray(xj_indiv, dtype=float)

    nclust = wfin.shape[0]
    ncol_w = wfin.shape[1]
    if np.issubdtype(cl_vec.dtype, np.integer):
        cl_idx = cl_vec.astype(int, copy=False)
    else:
        _, cl_idx = np.unique(cl_vec, return_inverse=True)

    if len(xj_indiv) != len(cl_vec):
        raise ValueError(
            "Clustered IV projection received xj_indiv with an incompatible length."
        )
    if not np.isfinite(xj_indiv).all():
        raise ValueError("Clustered IV projection received non-finite xj_indiv values.")

    xj_sq_sum = np.bincount(cl_idx, weights=xj_indiv**2, minlength=nclust)
    xjs_indiv = xj_indiv / np.sqrt(xj_sq_sum[cl_idx])
    xjs_indiv[~np.isfinite(xjs_indiv)] = 0

    w_indiv = wfin[cl_idx, :]
    wx = np.zeros((nclust, ncol_w), dtype=float)
    wx[:, 0] = np.bincount(
        cl_idx,
        weights=w_indiv[:, 0] * xjs_indiv * xj_indiv,
        minlength=nclust,
    )

    if ncol_w > 1:
        y = w_indiv[:, 1:] * xj_indiv[:, None]
        r = np.asarray(residualize(y), dtype=float)
        wx[:, 1:] = np.column_stack(
            [
                np.bincount(cl_idx, weights=r[:, col] * xjs_indiv, minlength=nclust)
                for col in range(r.shape[1])
            ]
        )

        validate_conditional_denominator(y, wx[:, 1:])

    if not np.isfinite(wx).all():
        raise ValueError(
            "Clustered conditional IV projection produced non-finite W values."
        )

    return wx


def encode_fixef_id(fixef_id: ArrayLike, n: int) -> np.ndarray:
    """Validate and factorize 1D or multi-way fixed-effect identifiers."""
    if isinstance(fixef_id, (str, bytes)):
        raise ValueError(
            "`fixef_id` must be a 1D or 2D array-like of fixed-effect identifiers."
        )

    raw_cols: list[np.ndarray] = []
    if isinstance(fixef_id, Mapping):
        if len(fixef_id) == 0:
            raise ValueError("`fixef_id` must contain at least one grouping vector.")
        for col_val in fixef_id.values():
            col_arr = np.asarray(col_val, dtype=object)
            if col_arr.ndim != 1 or col_arr.shape[0] != n:
                raise ValueError(
                    "`fixef_id` grouping vectors must be 1D and match `w` rows."
                )
            raw_cols.append(col_arr)
    elif isinstance(fixef_id, pd.DataFrame):
        if fixef_id.shape[0] != n:
            raise ValueError("`fixef_id` row count does not match `w`.")
        if fixef_id.shape[1] < 1:
            raise ValueError("`fixef_id` must contain at least one grouping column.")
        for idx in range(fixef_id.shape[1]):
            raw_cols.append(fixef_id.iloc[:, idx].to_numpy())
    else:
        fe_arr = np.asarray(fixef_id, dtype=object)
        if fe_arr.ndim not in (1, 2):
            raise ValueError("`fixef_id` must be 1D or 2D.")
        if fe_arr.shape[0] != n:
            raise ValueError("`fixef_id` row count does not match `w`.")
        if fe_arr.ndim == 1:
            raw_cols.append(fe_arr)
        else:
            if fe_arr.shape[1] < 1:
                raise ValueError(
                    "`fixef_id` must contain at least one grouping column."
                )
            for idx in range(fe_arr.shape[1]):
                raw_cols.append(fe_arr[:, idx])

    encoded_cols: list[np.ndarray] = []
    for col_arr in raw_cols:
        if bool(np.any(pd.isna(col_arr))):
            raise ValueError("`fixef_id` contains missing or NaN values.")
        if np.issubdtype(col_arr.dtype, np.floating) and not np.isfinite(col_arr).all():
            raise ValueError("`fixef_id` contains non-finite values.")
        if col_arr.dtype == object:
            for item in col_arr:
                if isinstance(item, (float, np.floating)) and not math.isfinite(
                    float(item)
                ):
                    raise ValueError("`fixef_id` contains non-finite values.")
        codes, _ = pd.factorize(col_arr, sort=False)
        if bool(np.any(codes < 0)):
            raise ValueError("`fixef_id` contains missing or NaN values.")
        encoded_cols.append(codes.astype(np.int64, copy=False))

    return np.column_stack(encoded_cols)


def make_fixef_demeaner(fixef_id: ArrayLike, n: int):
    """Build a scale-aware unweighted demeaning callable using PyFixest."""
    from pyfixest.core.demean import demean_within

    fe_codes = encode_fixef_id(fixef_id, n)
    counts = [np.bincount(codes) for codes in fe_codes.T]
    anchors = []
    rows = np.arange(n)
    for codes, sizes in zip(fe_codes.T, counts, strict=True):
        first_rows = np.full(len(sizes), n, dtype=int)
        np.minimum.at(first_rows, codes, rows)
        anchors.append(first_rows[codes])
    weights = np.ones(n)
    groups = fe_codes.astype(np.uintp, copy=False)

    def group_means(values: np.ndarray, codes: np.ndarray, sizes: np.ndarray):
        sums = np.zeros((len(sizes), values.shape[1]))
        np.add.at(sums, codes, values)
        return sums / sizes[:, None]

    def demean(
        values: MatrixLike, *, context: str, drop_absorbed: bool = False
    ) -> np.ndarray:
        matrix, vector = coerce_numeric_matrix(values, context)
        if matrix.shape[0] != n:
            raise ValueError(f"{context} does not line up with the fixed effects.")
        if matrix.shape[1] == 0:
            return matrix.copy()

        # Compare shifts in the absorbed space before scaling. A large group
        # offset can occur in any effect, and in different effects by column.
        shifted = matrix.copy()
        original = np.max(np.abs(matrix), axis=0)
        scales = original.copy()
        for anchor in anchors:
            with np.errstate(over="ignore"):
                candidate = matrix - matrix[anchor]
            candidate_scales = np.max(np.abs(candidate), axis=0)
            better = candidate_scales < scales
            shifted[:, better] = candidate[:, better]
            scales[better] = candidate_scales[better]
        scales[scales == 0] = 1.0
        shifted /= scales
        # A difference smaller than one input spacing can be introduced by
        # rounding the absorbed levels. Retain contrasts at or above it.
        noise = (original - np.nextafter(original, 0.0)) / scales
        if len(anchors) == 1:
            codes = fe_codes[:, 0]
            out = shifted - group_means(shifted, codes, counts[0])[codes]
        else:

            def solve(x: np.ndarray) -> np.ndarray:
                out, converged = demean_within(
                    x=x, flist=groups, weights=weights, tol=1e-14, maxiter=10_000
                )
                if not converged or not np.isfinite(out).all():
                    raise ValueError(
                        f"{context} fixed-effect demeaning did not converge."
                    )
                return out

            out = solve(shifted)
            # Check the group moments against the achieved within scale.
            # An absolute solver tolerance can leave substantial leakage
            # when most of the variation was absorbed.
            for step in range(7):
                size = np.max(np.abs(out), axis=0)
                moment = np.zeros_like(size)
                for codes, sizes in zip(fe_codes.T, counts, strict=True):
                    np.maximum(
                        moment,
                        np.max(np.abs(group_means(out, codes, sizes)), axis=0),
                        out=moment,
                    )
                refine = (moment > 1e-11 * size) & (size >= noise)
                if not np.any(refine):
                    break
                if step == 6:
                    raise ValueError(
                        f"{context} fixed-effect demeaning did not reach the required "
                        "group-moment tolerance."
                    )
                out[:, refine] = solve(out[:, refine] / size[refine]) * size[refine]
        if drop_absorbed:
            out[:, np.max(np.abs(out), axis=0) < noise] = 0.0
        return restore_residualizer_shape(out * scales, vector)

    return demean


def orthogonalize_w(
    w: MatrixLike,
    xj: ArrayLike,
    xjs: ArrayLike,
    model_mat: MatrixLike,
    include_intercept: bool = True,
    fixef_id: ArrayLike | None = None,
) -> MatrixLike:
    """Build the conditional projection basis for non-clustered SCPC.

    Conditional SCPC removes the part of the spatial directions that is
    explained by the regression covariates. This helper performs that
    orthogonalization for the ordinary, non-clustered case.

    Args:
        w: Spatial projection matrix.
        xj: Coefficient-specific influence direction.
        xjs: Normalized sign or scaling vector for `xj`.
        model_mat: Regression design matrix used for orthogonalization.
        include_intercept: Whether an intercept should be included when
            `fixef_id` is `None`.
        fixef_id: Optional group IDs aligned positionally with `w`. Supply one
            vector, a matrix or DataFrame with one column per effect, or a
            mapping to group vectors. Pass only within-identifiable regressors
            in `model_mat` when absorbing fixed effects.

    Returns:
        The orthogonalized projection matrix for conditional inference.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] < 1 or w.shape[1] < 1:
        raise ValueError("`w` must be a non-empty 2D matrix.")
    if not np.isfinite(w).all():
        raise ValueError("`w` contains non-finite values.")
    n = w.shape[0]

    xj = np.asarray(xj, dtype=float)
    xjs = np.asarray(xjs, dtype=float)
    if xj.ndim != 1 or xj.shape[0] != n or not np.isfinite(xj).all():
        raise ValueError("`xj` must be a finite 1D vector matching `w` rows.")
    if xjs.ndim != 1 or xjs.shape[0] != n or not np.isfinite(xjs).all():
        raise ValueError("`xjs` must be a finite 1D vector matching `w` rows.")

    demean = None if fixef_id is None else make_fixef_demeaner(fixef_id, n)

    wx = w.copy()
    wx[:, 0] = wx[:, 0] * xj * xjs

    if wx.shape[1] > 1:
        has_intercept_col = (
            hasattr(model_mat, "columns") and "(Intercept)" in model_mat.columns
        )
        x = np.asarray(model_mat, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != n:
            raise ValueError("`model_mat` must be a 2D matrix matching `w` rows.")
        if not np.isfinite(x).all():
            raise ValueError("`model_mat` contains non-finite values.")
        if include_intercept and demean is None and not has_intercept_col:
            x = np.column_stack((np.ones(n), x))

        rx = wx[:, 1:] * xj[:, None]

        # Conditional OLS denominator weights are S_x M_V D_x B_q.
        residualize = make_ols_residualizer(
            x, demean=demean, require_full_rank=demean is not None
        )
        rr = residualize(rx)
        wx[:, 1:] = rr * xjs[:, None]
        validate_conditional_denominator(rx, wx[:, 1:])

    return wx


def orthogonalize_w_cluster(
    w: MatrixLike,
    cl_vec: ArrayLike,
    xj_indiv: ArrayLike,
    model_mat_indiv: MatrixLike,
    include_intercept: bool = True,
) -> MatrixLike:
    """Build the conditional projection basis for clustered SCPC.

    In the clustered case, SCPC has to move between individual observations
    and cluster-level spatial directions. This helper carries out that
    clustered orthogonalization and aggregation step.

    Args:
        w: Cluster-level spatial projection matrix.
        cl_vec: Cluster membership for each individual observation.
        xj_indiv: Individual-level influence direction.
        model_mat_indiv: Individual-level design matrix.
        include_intercept: Whether an intercept should be included.

    Returns:
        The cluster-level orthogonalized projection matrix.
    """
    w = np.asarray(w, dtype=float)
    cl_vec = np.asarray(cl_vec)
    xj_indiv = np.asarray(xj_indiv, dtype=float)

    nclust = w.shape[0]
    ncol_w = w.shape[1]
    # this lets the large-n branch pass in its permuted cluster coding directly.
    if np.issubdtype(cl_vec.dtype, np.integer):
        cl_idx = cl_vec.astype(int, copy=False)
    else:
        _, cl_idx = np.unique(cl_vec, return_inverse=True)

    # Normalize the influence direction for the final cluster aggregation.
    xj_sq_sum = np.bincount(cl_idx, weights=xj_indiv**2, minlength=nclust)
    xjs_indiv = xj_indiv / np.sqrt(xj_sq_sum[cl_idx])
    xjs_indiv[~np.isfinite(xjs_indiv)] = 0

    w_indiv = w[cl_idx, :]
    wx = np.zeros((nclust, ncol_w), dtype=float)
    wx[:, 0] = np.bincount(
        cl_idx,
        weights=w_indiv[:, 0] * xjs_indiv * xj_indiv,
        minlength=nclust,
    )

    if ncol_w > 1:
        # Conditional clustered OLS denominator weights are Xi_s' M_V Xi B_q.
        weighted = w_indiv[:, 1:] * xj_indiv[:, None]
        x = np.asarray(model_mat_indiv, dtype=float)
        has_intercept_col = (
            hasattr(model_mat_indiv, "columns")
            and "(Intercept)" in model_mat_indiv.columns
        )
        if include_intercept and not has_intercept_col:
            x = np.column_stack((np.ones(x.shape[0]), x))

        residualize = make_ols_residualizer(x)
        residual = residualize(weighted)
        for col in range(ncol_w - 1):
            wx[:, col + 1] = np.bincount(
                cl_idx,
                weights=residual[:, col] * xjs_indiv,
                minlength=nclust,
            )

        validate_conditional_denominator(weighted, wx[:, 1:])

    return wx
