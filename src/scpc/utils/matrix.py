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
    if a.shape[0] == 0:
        raise ValueError(f"{arg} must have at least one row.")
    if not np.isfinite(a).all():
        raise ValueError(f"{arg} contains non-finite values.")
    return a, vec_input


def restore_residualizer_shape(a: np.ndarray, vec_input: bool) -> np.ndarray:
    """Restore vector inputs to a one-dimensional result."""
    return a[:, 0] if vec_input else a


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
    X, _ = coerce_numeric_matrix(X, "IV residualizer X")
    Z, _ = coerce_numeric_matrix(Z, "IV residualizer Z")
    if X.shape[0] != Z.shape[0]:
        raise ValueError(
            "IV residualizer requires X and Z to have the same number of rows."
        )

    if demean is not None:
        # scpcr demeans the raw iv design inside the residualizer when fixed
        # effects are present, so we do the same here.
        X = np.asarray(demean(X, context="IV residualizer X"), dtype=float)
        Z = np.asarray(demean(Z, context="IV residualizer Z"), dtype=float)

    qz, _ = np.linalg.qr(Z, mode="reduced")
    pzx = qz @ (qz.T @ X)
    A = X.T @ pzx
    if np.linalg.matrix_rank(A, tol=tol) < A.shape[1]:
        raise ValueError("IV residualizer is rank deficient: X'P_ZX is not full rank.")

    def residualize(Y: MatrixLike) -> np.ndarray:
        Y, vec_input = coerce_numeric_matrix(Y, "IV residualizer Y")
        if Y.shape[0] != X.shape[0]:
            raise ValueError(
                "IV residualizer received Y with an incompatible row count."
            )

        if demean is not None:
            Y = np.asarray(demean(Y, context="IV residualizer Y"), dtype=float)

        # project y into the instrument space before taking out the x part.
        if transpose:
            coef = np.linalg.solve(A.T, X.T @ Y)
            return restore_residualizer_shape(Y - pzx @ coef, vec_input)
        pzy = qz @ (qz.T @ Y)
        coef = np.linalg.solve(A, X.T @ pzy)
        if not np.isfinite(coef).all():
            raise ValueError(
                "IV residualizer produced non-finite auxiliary coefficients."
            )

        return restore_residualizer_shape(Y - X @ coef, vec_input)

    return residualize


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

    if not np.isfinite(wx).all():
        raise ValueError(
            "Clustered conditional IV projection produced non-finite W values."
        )

    return wx


def _encode_fixef_id(fixef_id: ArrayLike, n: int) -> np.ndarray:
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
            col_arr = np.asarray(col_val)
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
        fe_arr = np.asarray(fixef_id)
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


def _project_fixef_factor(
    mat: np.ndarray, codes: np.ndarray, n_groups: int, counts: np.ndarray
) -> None:
    """Subtract group means for one fixed-effect factor in place."""
    sums = np.zeros((n_groups, mat.shape[1]), dtype=float)
    np.add.at(sums, codes, mat)
    mat -= (sums / counts[:, None])[codes, :]


def _demean_by_fixef(values: np.ndarray, fe_codes: np.ndarray) -> np.ndarray:
    """Demean columns of `values` within one-way or multi-way fixed effects."""
    if values.shape[1] == 0:
        return values.copy()

    factors: list[tuple[np.ndarray, int, np.ndarray]] = []
    for col in range(fe_codes.shape[1]):
        codes = fe_codes[:, col]
        n_groups = int(codes.max()) + 1
        counts = np.bincount(codes, minlength=n_groups).astype(float)
        factors.append((codes, n_groups, counts))

    out = values.astype(float, copy=True)
    if len(factors) == 1:
        codes, n_groups, counts = factors[0]
        _project_fixef_factor(out, codes, n_groups, counts)
        return out

    def _sweep(mat: np.ndarray) -> np.ndarray:
        res = mat.copy()
        for codes, n_groups, counts in factors:
            _project_fixef_factor(res, codes, n_groups, counts)
        for codes, n_groups, counts in reversed(factors[:-1]):
            _project_fixef_factor(res, codes, n_groups, counts)
        return res

    col_scale = np.max(np.abs(out), axis=0)
    col_scale[col_scale == 0.0] = 1.0
    tol = 1e-12
    max_iter = 2000

    for _ in range(max_iter):
        x1 = _sweep(out)
        dx1 = x1 - out
        if float(np.max(np.abs(dx1) / col_scale[None, :])) <= tol:
            return x1

        x2 = _sweep(x1)
        dx2 = x2 - x1
        if float(np.max(np.abs(dx2) / col_scale[None, :])) <= tol:
            return x2

        ddx = dx2 - dx1
        denom = np.sum(ddx * ddx, axis=0)
        numer = np.sum(dx2 * ddx, axis=0)
        out = x2
        active = denom > (1e-30 * (col_scale**2))
        if np.any(active):
            step = np.zeros_like(denom)
            step[active] = numer[active] / denom[active]
            out[:, active] -= dx2[:, active] * step[None, active]

    raise ValueError("Fixed-effect demeaning did not converge.")


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
        fixef_id: Optional fixed-effect identifiers for demeaning.

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

    fe_codes = None if fixef_id is None else _encode_fixef_id(fixef_id, n)

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
        if include_intercept and fe_codes is None and not has_intercept_col:
            x = np.column_stack((np.ones(n), x))

        rx = wx[:, 1:] * xj[:, None]

        # Conditional OLS denominator weights are S_x M_V D_x B_q.
        pre_max = float(np.max(np.linalg.norm(rx, axis=0))) if rx.size > 0 else 0.0
        if not math.isfinite(pre_max) or pre_max == 0.0:
            raise ValueError(
                "Conditional projection collapsed all non-constant W columns."
            )

        if fe_codes is not None:
            # Apply Frisch-Waugh-Lovell fixed-effect demeaning to both X and Rx.
            pre_x_norms = np.linalg.norm(x, axis=0)
            x = _demean_by_fixef(x, fe_codes)
            rx = _demean_by_fixef(rx, fe_codes)
            if x.shape[1] > 0:
                post_x_norms = np.linalg.norm(x, axis=0)
                if bool(
                    np.any(
                        (pre_x_norms == 0.0)
                        | (post_x_norms <= 1e-10 * pre_x_norms)
                    )
                ):
                    raise ValueError("OLS residualizer is rank deficient.")
                x_scaled = x / post_x_norms[None, :]
                if int(np.linalg.matrix_rank(x_scaled, tol=1e-10)) < x.shape[1]:
                    raise ValueError("OLS residualizer is rank deficient.")

        rr = rx - x @ np.linalg.lstsq(x, rx, rcond=None)[0]
        wx[:, 1:] = rr * xjs[:, None]
        post_norms = np.linalg.norm(wx[:, 1:], axis=0)
        if not np.isfinite(post_norms).all() or bool(
            np.all(post_norms <= 1e-12 * pre_max)
        ):
            raise ValueError(
                "Conditional projection collapsed all non-constant W columns."
            )

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
        pre_max = float(
            np.max(
                np.sqrt(
                    np.sum(
                        (w_indiv[:, 1:] * xj_indiv[:, None]) ** 2,
                        axis=0,
                    )
                )
            )
        )
        if not math.isfinite(pre_max) or pre_max == 0.0:
            raise ValueError(
                "Conditional projection collapsed all non-constant W columns."
            )

        x = np.asarray(model_mat_indiv, dtype=float)
        has_intercept_col = (
            hasattr(model_mat_indiv, "columns")
            and "(Intercept)" in model_mat_indiv.columns
        )
        if include_intercept and not has_intercept_col:
            x = np.column_stack((np.ones(x.shape[0]), x))

        for col in range(1, ncol_w):
            temp = w_indiv[:, col] * xj_indiv
            resid_col = temp - x @ np.linalg.lstsq(x, temp, rcond=None)[0]
            wx[:, col] = np.bincount(
                cl_idx,
                weights=resid_col * xjs_indiv,
                minlength=nclust,
            )

        post_norms = np.linalg.norm(wx[:, 1:], axis=0)
        if not np.isfinite(post_norms).all() or bool(
            np.all(post_norms <= 1e-12 * pre_max)
        ):
            raise ValueError(
                "Conditional projection collapsed all non-constant W columns."
            )

    return wx
