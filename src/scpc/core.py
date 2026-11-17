from __future__ import annotations

from collections.abc import Sequence
import importlib
import math
import os
from typing import Any
import warnings

import numpy as np

from .types import DataFrameLike, MatrixLike, ModelLike, SCPCResult
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
    LARGE_N_THRESHOLD,
    get_cv,
    get_oms,
    lnget_oms,
    max_rp,
    set_oms_wfin,
    validate_large_n_seed,
    validate_scpc_method,
)

VALID_BACKENDS = {"auto", "rust", "fallback"}
RUST_API_VERSION = 2


def resolve_backend(backend: object) -> str:
    """Resolve the execution backend from argument and `SPATIAL_BACKEND`."""
    if not isinstance(backend, str) or backend not in VALID_BACKENDS:
        raise ValueError('`backend` must be one of "auto", "rust", or "fallback".')
    if backend == "auto":
        env_backend = os.environ.get("SPATIAL_BACKEND", "auto") or "auto"
        if env_backend not in VALID_BACKENDS:
            raise ValueError(
                '`SPATIAL_BACKEND` must be one of "auto", "rust", or "fallback".'
            )
        return env_backend
    return backend


def validate_nthreads(nthreads: object) -> int | None:
    """Validate optional thread budget (`None` or integer >= 1)."""
    if nthreads is None:
        return None
    if (
        isinstance(nthreads, bool)
        or not isinstance(nthreads, (int, np.integer))
        or int(nthreads) < 1
    ):
        raise ValueError("`nthreads` must be a positive integer or None.")
    return int(nthreads)


def load_rust_backend() -> Any | None:
    """Import the compiled Rust extension module if present and ABI-compatible."""
    try:
        rust_mod = importlib.import_module("spatial_spur_rust")
    except ImportError:
        return None
    if getattr(rust_mod, "API_VERSION", None) != RUST_API_VERSION:
        return None
    return rust_mod


def compute_orthogonal_projector(
    model_mat: MatrixLike,
    *,
    include_intercept: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute orthonormal `(U, V)` factors for `I - X X^+` via thin SVD."""
    x_design = np.asarray(model_mat, dtype=np.float64)
    columns = getattr(model_mat, "columns", None)
    has_intercept_col = columns is not None and "(Intercept)" in columns
    if include_intercept and not has_intercept_col:
        x_design = np.column_stack(
            (np.ones(x_design.shape[0], dtype=np.float64), x_design)
        )
    if x_design.shape[1] == 0:
        return None, None
    u, s, _ = np.linalg.svd(x_design, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return None, None
    tol = 1e-12 * max(x_design.shape) * s[0]
    keep = s > tol
    if not np.any(keep):
        return None, None
    proj = np.ascontiguousarray(u[:, keep], dtype=np.float64)
    return proj, proj


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
    backend: str = "auto",
    nthreads: int | None = None,
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
        backend: Execution backend (`"auto"`, `"rust"`, or `"fallback"`).
        nthreads: Optional thread budget for compiled execution (`None` or >= 1).

    Returns:
        The fitted SCPC result object.

    Raises:
        ValueError: Raised later for invalid combinations of arguments.
    """

    if avc <= 0.001 or avc >= 0.99:
        raise ValueError("Option avc() must be in (0.001, 0.99).")
    method = validate_scpc_method(method)
    large_n_seed = validate_large_n_seed(large_n_seed)
    resolved_backend = resolve_backend(backend)
    resolved_nthreads = validate_nthreads(nthreads)

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

    raw_coef_names = get_coef_names(model)
    k_use = p if ncoef is None else min(ncoef, p)
    coef_names = raw_coef_names[:k_use]

    is_m1_exact = (
        method != "approx"
        and (method == "exact" or neff < LARGE_N_THRESHOLD)
        and cluster is None
        and cond_fixef_id is None
        and not cond_is_iv
        and k_use >= 1
    )

    if resolved_backend != "fallback":
        rust_mod = load_rust_backend()
        if resolved_backend == "rust":
            if rust_mod is None:
                raise RuntimeError(
                    "Rust backend requested (`backend='rust'`), but "
                    f"`spatial_spur_rust` (API_VERSION={RUST_API_VERSION}) is not installed."
                )
            if not is_m1_exact:
                reasons: list[str] = []
                if method == "approx" or (
                    method == "auto" and neff >= LARGE_N_THRESHOLD
                ):
                    reasons.append("approximate large-n method")
                if cluster is not None:
                    reasons.append("clustered inference")
                if cond_fixef_id is not None:
                    reasons.append("absorbed fixed-effect demeaning")
                if cond_is_iv:
                    reasons.append("instrumental variables")
                if k_use < 1:
                    reasons.append("non-positive ncoef")
                raise NotImplementedError(
                    "Rust backend does not support "
                    + ", ".join(reasons)
                    + " in Milestone 1."
                )

        if rust_mod is not None and is_m1_exact:
            coords_mat = np.ascontiguousarray(coords, dtype=np.float64)
            coef_use = np.ascontiguousarray(coef[:k_use], dtype=np.float64)
            wj_mat = np.ascontiguousarray(
                neff * (s @ bread_inv[:k_use, :].T) + coef_use[None, :],
                dtype=np.float64,
            )
            if uncond:
                xj_mat = None
                xjs_mat = None
                proj_u = None
                proj_v = None
            else:
                xj_mat = np.ascontiguousarray(
                    neff * (model_mat_cond @ bread_inv[:k_use, :].T),
                    dtype=np.float64,
                )
                xjs_mat = np.ascontiguousarray(np.sign(xj_mat), dtype=np.float64)
                proj_u, proj_v = compute_orthogonal_projector(
                    model_mat_cond,
                    include_intercept=cond_include_intercept,
                )

            rust_res = rust_mod.run_exact_scpc(
                coords_mat,
                coef_use,
                wj_mat,
                float(avc),
                latlong=bool(latlong),
                uncond=bool(uncond),
                cvs=bool(cvs),
                xj=xj_mat,
                xjs=xjs_mat,
                proj_u=proj_u,
                proj_v=proj_v,
                nthreads=resolved_nthreads,
            )
            if not (rust_res["tied"] and resolved_backend == "auto"):
                return SCPCResult(
                    scpcstats=np.asarray(rust_res["scpcstats"], dtype=float),
                    scpccvs=None
                    if rust_res["scpccvs"] is None
                    else np.asarray(rust_res["scpccvs"], dtype=float),
                    w=np.asarray(rust_res["wfin"], dtype=float),
                    avc=avc,
                    c0=float(rust_res["c0"]),
                    cv=float(rust_res["cv"]),
                    q=int(rust_res["q"]),
                    coef_names=coef_names,
                    method="exact",
                    large_n_seed=large_n_seed,
                    backend="rust",
                )

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
        backend="fallback",
    )
