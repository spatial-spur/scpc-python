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


def normalize_cluster_direction(
    xj_indiv: np.ndarray, cl_idx: np.ndarray, nclust: int
) -> np.ndarray:
    """Normalize `xj_indiv` within each cluster without square overflow or underflow."""
    cl_max = np.zeros(nclust, dtype=float)
    np.maximum.at(cl_max, cl_idx, np.abs(xj_indiv))
    row_scale = cl_max[cl_idx]
    xj_scaled = np.divide(
        xj_indiv,
        row_scale,
        out=np.zeros_like(xj_indiv, dtype=float),
        where=row_scale > 0.0,
    )
    xj_sq_sum = np.bincount(cl_idx, weights=xj_scaled**2, minlength=nclust)
    cl_denom = np.sqrt(xj_sq_sum[cl_idx])
    xjs_indiv = np.divide(
        xj_scaled,
        cl_denom,
        out=np.zeros_like(xj_scaled),
        where=cl_denom > 0.0,
    )
    xjs_indiv[~np.isfinite(xjs_indiv)] = 0.0
    # An exact zero in a singleton follows the unclustered sign convention.
    singleton = np.bincount(cl_idx, minlength=nclust)[cl_idx] == 1
    xjs_indiv[singleton & (row_scale == 0.0)] = 1.0
    return xjs_indiv


def validate_conditional_denominator(before: np.ndarray, after: np.ndarray) -> None:
    """Reject conditional spatial bases with no informative denominator direction."""
    scale = max(np.max(np.abs(before)), np.max(np.abs(after)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Conditional projection collapsed all non-constant W columns.")
    # A single common scale protects norms and preserves their relative cutoff.
    pre_max = np.max(np.linalg.norm(before / scale, axis=0))
    post_max = np.max(np.linalg.norm(after / scale, axis=0))
    if pre_max == 0 or post_max <= 1e-12 * pre_max:
        raise ValueError("Conditional projection collapsed all non-constant W columns.")


def project_conditional_weights(
    wfin: MatrixLike,
    xj: ArrayLike,
    xjs: ArrayLike,
    residualize,
    *,
    rescale: bool = False,
) -> MatrixLike:
    """Build the conditional weights for unclustered OLS or IV SCPC.

    Apply the fitted residual maker to the regressor-weighted spatial
    directions, then scale by the normalized coefficient influence direction.

    Args:
        wfin: Spatial projection matrix.
        xj: Coefficient-specific influence direction.
        xjs: Normalized sign or scaling vector for `xj`.
        residualize: OLS residual maker or the transposed IV residual maker.
        rescale: Remove the common direction scale before projection when only
            rejection probabilities are needed.

    Returns:
        The orthogonalized conditional projection matrix.
    """
    wfin = np.asarray(wfin, dtype=float)
    xj = np.asarray(xj, dtype=float)
    xjs = np.asarray(xjs, dtype=float)

    if len(xj) != wfin.shape[0]:
        raise ValueError(
            "Conditional projection received xj with an incompatible length."
        )
    if len(xjs) != wfin.shape[0]:
        raise ValueError(
            "Conditional projection received xjs with an incompatible length."
        )
    if not np.isfinite(xj).all() or not np.isfinite(xjs).all():
        raise ValueError("Conditional projection received non-finite xj or xjs values.")

    if rescale:
        scale = np.max(np.abs(xj))
        if scale > 0:
            xj = xj / scale
    wx = wfin.copy()
    wx[:, 0] = wfin[:, 0] * xj * xjs

    if wfin.shape[1] > 1:
        y = wfin[:, 1:] * xj[:, None]
        r = np.asarray(residualize(y), dtype=float)
        wx[:, 1:] = r * xjs[:, None]
        validate_conditional_denominator(y, wx[:, 1:])

    if not np.isfinite(wx).all():
        raise ValueError("Conditional projection produced non-finite W values.")

    return wx


def project_clustered_weights(
    wfin: MatrixLike,
    cl_vec: ArrayLike,
    xj_indiv: ArrayLike,
    residualize,
    *,
    rescale: bool = False,
) -> MatrixLike:
    """Build the conditional weights for clustered OLS or IV SCPC.

    Normalize individual influence directions within clusters, apply the
    fitted residual maker to regressor-weighted spatial directions,
    and aggregate the resulting weights back to the cluster level.

    Args:
        wfin: Cluster-level spatial projection matrix.
        cl_vec: Cluster membership for each individual observation.
        xj_indiv: Individual-level influence direction.
        residualize: OLS residual maker or the transposed IV residual maker.
        rescale: Remove the common direction scale after obtaining the
            within-cluster normalized weights.

    Returns:
        The cluster-level conditional projection matrix.
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
            "Clustered conditional projection received xj_indiv with an incompatible length."
        )
    if not np.isfinite(xj_indiv).all():
        raise ValueError(
            "Clustered conditional projection received non-finite xj_indiv values."
        )

    xjs_indiv = normalize_cluster_direction(xj_indiv, cl_idx, nclust)
    if rescale:
        scale = np.max(np.abs(xj_indiv))
        if scale > 0:
            xj_indiv = xj_indiv / scale

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
            "Clustered conditional projection produced non-finite W values."
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


def input_spacing(magnitude: np.ndarray) -> np.ndarray:
    """Binary64 spacing in the input's binade, including subnormals and DBL_MAX."""
    _, exponent = np.frexp(magnitude)
    power = np.where(magnitude == 0, -1074, np.maximum(exponent - 53, -1074))
    return np.ldexp(1.0, power)


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

        if len(anchors) == 1:
            codes = fe_codes[:, 0]
            shape = (len(counts[0]), matrix.shape[1])
            original = np.zeros(shape)
            np.maximum.at(original, codes, np.abs(matrix))
            with np.errstate(over="ignore"):
                candidate = matrix - matrix[anchors[0]]
            shifted_scale = np.zeros(shape)
            np.maximum.at(shifted_scale, codes, np.abs(candidate))
            use_shift = shifted_scale < original
            scale = np.where(use_shift, shifted_scale, original)
            scale[scale == 0] = 1.0
            values = np.where(use_shift[codes], candidate, matrix) / scale[codes]
            out = values - group_means(values, codes, counts[0])[codes]
            if drop_absorbed:
                amplitude = np.zeros(shape)
                np.maximum.at(amplitude, codes, np.abs(out))
                drop = 2.0 * amplitude < input_spacing(original) / scale
                out[drop[codes]] = 0.0
            return restore_residualizer_shape(out * scale[codes], vector)

        # Compare shifts in the absorbed space before scaling. A large group
        # offset can occur in any effect, and in different effects by column.
        shifted = matrix.copy()
        abs_matrix = np.abs(matrix)
        original = np.max(abs_matrix, axis=0)
        scales = original.copy()
        for anchor in anchors:
            with np.errstate(over="ignore"):
                candidate = shifted - shifted[anchor]
            candidate_scales = np.max(np.abs(candidate), axis=0)
            better = candidate_scales < scales
            shifted[:, better] = candidate[:, better]
            scales[better] = candidate_scales[better]
        scales[scales == 0] = 1.0
        shifted /= scales
        noise = 0.5 * (input_spacing(original) / scales)

        def max_group_moment(arr: np.ndarray) -> np.ndarray:
            moment = np.zeros(arr.shape[1])
            for codes, sizes in zip(fe_codes.T, counts, strict=True):
                np.maximum(
                    moment,
                    np.max(np.abs(group_means(arr, codes, sizes)), axis=0),
                    out=moment,
                )
            return moment

        def solve(x: np.ndarray) -> np.ndarray:
            out, converged = demean_within(
                x=x, flist=groups, weights=weights, tol=1e-14, maxiter=10_000
            )
            if not np.isfinite(out).all():
                raise ValueError(f"{context} fixed-effect demeaning did not converge.")
            if not converged:
                size = np.maximum(np.max(np.abs(out), axis=0), 1e-15)
                if np.any(max_group_moment(out) > 1e-11 * size):
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
            moment = max_group_moment(out)
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
