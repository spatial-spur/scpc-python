from __future__ import annotations

from collections.abc import Sequence
import math
import warnings

import numpy as np

from .types import DataFrameLike, ModelLike, SCPCResult
from .utils.data import (
    get_conditional_projection_setup,
    get_obs_index,
    get_scpc_model_matrix,
    resolve_coords_input,
)
from .utils.matrix import orthogonalize_w, orthogonalize_w_cluster
from .utils.spatial import get_cv, get_distmat, get_oms, max_rp, set_oms_wfin


def scpc(
    model: ModelLike,
    data: DataFrameLike,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    cluster: str | None = None,
    ncoef: int | None = None,
    avc: float = 0.03,
    uncond: bool = False,
    cvs: bool = False,
) -> SCPCResult:
    """Run spatial correlation-robust inference.

    This is the package entrypoint. It combines the data alignment helpers,
    the spatial setup helpers, and the matrix calculations into one inference
    routine that mirrors the public `scpc()` function in the R package.

    Args:
        model: Fitted model object to analyze.
        data: Data used to fit the model.
        lon: Longitude column name for geodesic coordinates.
        lat: Latitude column name for geodesic coordinates.
        coords_euclidean: Euclidean coordinate column names.
        cluster: Optional clustering column.
        ncoef: Number of coefficients to report.
        avc: Upper bound on average pairwise correlation.
        uncond: Whether to skip the conditional adjustment.
        cvs: Whether to return additional critical values.

    Returns:
        The fitted SCPC result object.

    Raises:
        ValueError: Raised later for invalid combinations of arguments.
    """

    if avc <= 0.001 or avc >= 0.99:
        raise ValueError("Option avc() must be in (0.001, 0.99).")

    resid = np.asarray(model.resid, dtype=float)
    model_mat = get_scpc_model_matrix(model)
    s = np.asarray(model.model.exog, dtype=float) * resid[:, None]
    n = s.shape[0]
    p = s.shape[1]
    neff = n

    if model_mat.shape[0] != n:
        raise ValueError(
            "Model matrix row count does not match score matrix row count."
        )

    cond_setup = get_conditional_projection_setup(model, model_mat, n=n, uncond=uncond)
    model_mat_cond = np.asarray(cond_setup.model_mat, dtype=float)
    cond_include_intercept = cond_setup.include_intercept
    cond_fixef_id = cond_setup.fixef_id

    if model_mat_cond.shape != (n, p):
        raise ValueError(
            "Conditional projection matrix dimensions are incompatible with model "
            f"coefficients (rows = {model_mat_cond.shape[0]}, cols = {model_mat_cond.shape[1]}, "
            f"expected {n} x {p})."
        )

    obs_index = get_obs_index(model, data)
    coord_info = resolve_coords_input(data, obs_index, lon, lat, coords_euclidean)
    coords = np.asarray(coord_info.coords, dtype=float)
    latlong = coord_info.latlong

    cl_idx: np.ndarray | None = None
    if cluster is not None:
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("`cluster` must be a single column name.")
        if cluster not in data.columns:
            raise ValueError(f"Cluster variable not found in data: {cluster}")
        if cond_fixef_id is not None and not uncond:
            raise ValueError(
                "Conditional SCPC with absorbed fixed effects and external "
                "clustering is not yet implemented; use `uncond = TRUE`."
            )

        cl_values = np.asarray(data.iloc[np.asarray(obs_index, dtype=int) - 1][cluster])
        _, cl_idx = np.unique(cl_values, return_inverse=True)

        coord_by_cl = np.zeros((cl_idx.max() + 1, coords.shape[1]), dtype=float)
        np.add.at(coord_by_cl, cl_idx, coords)
        n_per_cl = np.bincount(cl_idx)
        coord_means = coord_by_cl / n_per_cl[:, None]
        first_index = np.array(
            [np.flatnonzero(cl_idx == g)[0] for g in range(len(n_per_cl))]
        )
        coord_first = coords[first_index, :]
        if np.max(np.abs(coord_means - coord_first)) > 1e-8:
            warnings.warn(
                "Coordinates vary within clusters. The first observation's "
                "coordinates are used for each cluster; consider averaging "
                "coordinates within clusters before calling scpc().",
                stacklevel=2,
            )

        s_cluster = np.zeros((len(n_per_cl), s.shape[1]), dtype=float)
        np.add.at(s_cluster, cl_idx, s)
        s = s_cluster
        coords = coord_first
        neff = s.shape[0]

    if coords.shape[1] == 1:
        coords = np.column_stack((coords, np.zeros(coords.shape[0])))

    d = get_distmat(coords, latlong)
    spc = set_oms_wfin(d, avc)
    wfin = np.asarray(spc.wfin, dtype=float)
    cvfin = spc.cvfin
    omsfin = spc.omsfin
    q = wfin.shape[1] - 1

    bread_inv = np.asarray(model.normalized_cov_params, dtype=float)

    k_use = p if ncoef is None else min(ncoef, p)
    out = np.full((k_use, 6), np.nan)
    levs = np.array([0.32, 0.10, 0.05, 0.01], dtype=float)
    cvs_mat = np.full((k_use, 4), np.nan) if cvs else None
    cvs_uncond = (
        np.array([get_cv(omsfin, q, level) for level in levs], dtype=float)
        if cvs
        else None
    )

    coef = np.asarray(model.params, dtype=float)

    for j in range(k_use):
        wj = neff * (bread_inv[j, :] @ s.T) + coef[j]

        tau_u = (
            math.sqrt(q)
            * np.dot(wfin[:, 0], wj)
            / math.sqrt(np.sum((wfin[:, 1:].T @ wj) ** 2))
        )
        se = math.sqrt(np.sum((wfin[:, 1:].T @ wj) ** 2)) / (
            math.sqrt(q) * math.sqrt(neff)
        )
        p_u = max_rp(omsfin, q, abs(tau_u) / math.sqrt(q))[0]

        if not uncond:
            if cluster is not None and cl_idx is not None:
                xj_indiv = neff * (bread_inv[j, :] @ model_mat_cond.T)
                wx = orthogonalize_w_cluster(
                    wfin,
                    cl_values,
                    xj_indiv,
                    model_mat_cond,
                    include_intercept=cond_include_intercept,
                )
            else:
                xj = neff * (bread_inv[j, :] @ model_mat_cond.T)
                xjs = np.sign(xj)
                wx = orthogonalize_w(
                    wfin,
                    xj,
                    xjs,
                    model_mat_cond,
                    include_intercept=cond_include_intercept,
                    fixef_id=cond_fixef_id,
                )

            omsx = get_oms(d, spc.c0, spc.cmax, wx, 1.2)
            p_c = max_rp(omsx, q, abs(tau_u) / math.sqrt(q))[0]
            cvx = get_cv(omsx, q, 0.05)
            p_final = max(p_u, p_c)
            cv = max(cvfin, cvx)
        else:
            p_final = p_u
            cv = cvfin

        out[j, :] = np.array(
            [
                coef[j],
                se,
                tau_u,
                p_final,
                coef[j] - cv * se,
                coef[j] + cv * se,
            ],
            dtype=float,
        )

        if cvs and cvs_mat is not None and cvs_uncond is not None:
            cvs_vec = cvs_uncond.copy()
            if not uncond:
                cvs_cond = np.array(
                    [get_cv(omsx, q, level) for level in levs], dtype=float
                )
                cvs_vec = np.maximum(cvs_vec, cvs_cond)
            cvs_mat[j, :] = cvs_vec

    return SCPCResult(
        scpcstats=out,
        scpccvs=cvs_mat,
        w=wfin,
        avc=avc,
        c0=spc.c0,
        cv=cvfin,
        q=q,
        call=None,  # TODO: add string representation or remove from result
    )
