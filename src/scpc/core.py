from __future__ import annotations

from collections.abc import Sequence
import math
import warnings

import numpy as np

from .types import DataFrameLike, ModelLike, SCPCResult
from .utils.data import (
    get_coef_names,
    get_conditional_projection_setup,
    get_fixest_bread_inv,
    get_fixest_score_matrix,
    get_obs_index,
    get_scpc_model_matrix,
    is_pyfixest_model,
    is_pyfixest_multi,
    resolve_coords_input,
)
from .utils.matrix import (
    orthogonalize_w,
    orthogonalize_w_cluster,
    orthogonalize_w_cluster_iv,
    orthogonalize_w_iv,
)
from .utils.spatial import (
    get_cv,
    get_oms,
    lnget_oms,
    max_rp,
    set_oms_wfin,
    validate_large_n_seed,
    validate_scpc_method,
)


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
    method: str = "auto",
    large_n_seed: int = 1,
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
        method: Requested spatial method.
        large_n_seed: Seed for the large-n approximation branch.
        uncond: Whether to skip the conditional adjustment.
        cvs: Whether to return additional critical values.

    Returns:
        The fitted SCPC result object.

    Raises:
        ValueError: Raised later for invalid combinations of arguments.
    """

    if avc <= 0.001 or avc >= 0.99:
        raise ValueError("Option avc() must be in (0.001, 0.99).")
    method = validate_scpc_method(method)
    large_n_seed = validate_large_n_seed(large_n_seed)

    if is_pyfixest_multi(model):
        raise ValueError(
            "`scpc()` only accepts a single fitted pyfixest model, not FixestMulti."
        )

    model_mat = get_scpc_model_matrix(model)
    if is_pyfixest_model(model):
        s = np.asarray(get_fixest_score_matrix(model), dtype=float)
        bread_inv = np.asarray(get_fixest_bread_inv(model), dtype=float)
        coef = np.asarray(getattr(model, "_beta_hat", None), dtype=float).reshape(-1)
    else:
        resid = np.asarray(model.resid, dtype=float)
        s = np.asarray(model.model.exog, dtype=float) * resid[:, None]
        bread_inv = np.asarray(model.normalized_cov_params, dtype=float)
        coef = np.asarray(model.params, dtype=float)

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
    cond_is_iv = cond_setup.is_iv
    cond_residualize = cond_setup.residualize

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
    cl_levels: np.ndarray | None = None
    cl_values: np.ndarray | None = None
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
        cl_levels, cl_idx = np.unique(cl_values, return_inverse=True)

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

    spc = set_oms_wfin(
        coords,
        avc,
        latlong,
        method=method,
        large_n_seed=large_n_seed,
    )
    d = None
    if not spc.large_n:
        if spc.distmat is None:
            raise ValueError("Internal error: exact SCPC setup is missing `distmat`.")
        d = np.asarray(spc.distmat, dtype=float)

    wfin = np.asarray(spc.wfin, dtype=float)
    cvfin = spc.cvfin
    omsfin = spc.omsfin
    perm = np.asarray(spc.perm, dtype=int)
    q = wfin.shape[1] - 1
    large_n_random_state = spc.random_state

    raw_coef_names = get_coef_names(model)
    k_use = p if ncoef is None else min(ncoef, p)
    coef_names = raw_coef_names[:k_use]
    out = np.full((k_use, 6), np.nan)
    levs = np.array([0.32, 0.10, 0.05, 0.01], dtype=float)
    cvs_mat = np.full((k_use, 4), np.nan) if cvs else None
    cvs_uncond = (
        np.array([get_cv(omsfin, q, level) for level in levs], dtype=float)
        if cvs
        else None
    )

    for j in range(k_use):
        wj = neff * (bread_inv[j, :] @ s.T) + coef[j]
        wj_perm = wj[perm]

        tau_u = (
            math.sqrt(q)
            * np.dot(wfin[:, 0], wj_perm)
            / math.sqrt(np.sum((wfin[:, 1:].T @ wj_perm) ** 2))
        )
        se = math.sqrt(np.sum((wfin[:, 1:].T @ wj_perm) ** 2)) / (
            math.sqrt(q) * math.sqrt(neff)
        )
        p_u = max_rp(omsfin, q, abs(tau_u) / math.sqrt(q))[0]

        if not uncond:
            if cluster is not None and cl_idx is not None:
                if cl_values is None or cl_levels is None:
                    raise ValueError(
                        "Internal error: clustered SCPC is missing cluster labels."
                    )

                cl_idx_scpc = cl_idx
                if spc.large_n and not np.array_equal(
                    perm, np.arange(len(perm), dtype=int)
                ):
                    # the sampled large-n branch reorders clusters before building w.
                    level_to_code = {
                        level: i for i, level in enumerate(cl_levels[perm])
                    }
                    cl_idx_scpc = np.array(
                        [level_to_code[value] for value in cl_values], dtype=int
                    )

                xj_indiv = neff * (bread_inv[j, :] @ model_mat_cond.T)
                if cond_is_iv:
                    wx = orthogonalize_w_cluster_iv(
                        wfin,
                        cl_idx_scpc,
                        xj_indiv,
                        residualize=cond_residualize,
                    )
                else:
                    wx = orthogonalize_w_cluster(
                        wfin,
                        cl_idx_scpc,
                        xj_indiv,
                        model_mat_cond,
                        include_intercept=cond_include_intercept,
                    )
            else:
                xj = neff * (bread_inv[j, :] @ model_mat_cond.T)
                xj = xj[perm]
                xjs = np.sign(xj)
                if cond_is_iv:
                    wx = orthogonalize_w_iv(
                        wfin,
                        xj,
                        xjs,
                        residualize=cond_residualize,
                    )
                else:
                    wx = orthogonalize_w(
                        wfin,
                        xj,
                        xjs,
                        model_mat_cond[perm, :],
                        include_intercept=cond_include_intercept,
                        fixef_id=cond_fixef_id,
                    )

            if spc.large_n:
                # this reuses the sampled omega construction from the large-n setup.
                omsx, large_n_random_state = lnget_oms(
                    spc.coords,
                    spc.c0,
                    spc.cmax,
                    wx,
                    1.2,
                    capM=1_000_000,
                    random_t=large_n_random_state
                    if large_n_random_state is not None
                    else 1,
                    latlong=latlong,
                )
            else:
                if d is None:
                    raise ValueError(
                        "Internal error: exact SCPC conditional branch is missing `distmat`."
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
        coef_names=coef_names,
        method=spc.method,
        large_n_seed=large_n_seed,
    )
