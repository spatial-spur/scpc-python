from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import eigsh

from ..types import ArrayLike, MatrixLike


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
    distmat = np.asarray(distmat, dtype=float)
    n = distmat.shape[0]
    sig = np.exp(-c0 * distmat)
    sig_d = demeanmat(sig)

    if qmax < n - 1:
        eigvals, v = eigsh(sig_d, k=qmax, which="LM")
        order = np.argsort(np.abs(eigvals))[::-1]
        v = v[:, order]
    else:
        eigvals, v = np.linalg.eigh(sig_d)
        order = np.argsort(eigvals)[::-1]
        v = v[:, order[:qmax]]

    # prepend the normalized constant direction, matching the R code
    return np.column_stack((np.full(n, 1 / math.sqrt(n)), v))


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
        include_intercept: Whether an intercept should be included.
        fixef_id: Optional fixed-effect identifiers for demeaning.

    Returns:
        The orthogonalized projection matrix for conditional inference.
    """
    w = np.asarray(w, dtype=float)
    xj = np.asarray(xj, dtype=float)
    xjs = np.asarray(xjs, dtype=float)

    wx = w.copy()
    wx[:, 0] = wx[:, 0] * xj * xjs

    if wx.shape[1] > 1:
        x = np.asarray(model_mat, dtype=float)
        has_intercept_col = (
            hasattr(model_mat, "columns") and "(Intercept)" in model_mat.columns
        )
        if include_intercept and not has_intercept_col:
            x = np.column_stack((np.ones(x.shape[0]), x))

        rx = wx[:, 1:] * xjs[:, None]
        if fixef_id is not None:
            raise ValueError("Fixed-effect demeaning is not implemented.")

        # remove the part of the spatial directions that lies in the regressor space
        rr = rx - x @ np.linalg.lstsq(x, rx, rcond=None)[0]
        wx[:, 1:] = rr * xj[:, None]

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

    # normalize the influence direction within each cluster before projection
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
        x = np.asarray(model_mat_indiv, dtype=float)
        has_intercept_col = (
            hasattr(model_mat_indiv, "columns")
            and "(Intercept)" in model_mat_indiv.columns
        )
        if include_intercept and not has_intercept_col:
            x = np.column_stack((np.ones(x.shape[0]), x))

        for col in range(1, ncol_w):
            temp = w_indiv[:, col] * xjs_indiv
            resid_col = temp - x @ np.linalg.lstsq(x, temp, rcond=None)[0]
            wx[:, col] = np.bincount(
                cl_idx,
                weights=resid_col * xj_indiv,
                minlength=nclust,
            )

    return wx
