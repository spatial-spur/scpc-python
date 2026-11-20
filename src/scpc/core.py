from __future__ import annotations

from collections.abc import Sequence
import importlib
import math
import os
from typing import Any
import warnings

import numpy as np
from statsmodels.regression.linear_model import OLS as SmOLS

from .types import DataFrameLike, MatrixLike, ModelLike, SCPCResult
from .utils.data import (
    get_coef_names,
    get_conditional_projection_setup,
    get_fixest_bread_inv,
    get_fixest_iv_design,
    get_fixest_score_matrix,
    get_obs_index,
    get_scpc_model_matrix,
    has_fixest_fe,
    is_fixest_iv_second_stage,
    is_pyfixest_model,
    is_pyfixest_multi,
    make_pyfixest_demeaner,
    resolve_coords_input,
)
from .utils.matrix import (
    iv_projector_factors,
    orthonormal_design_span,
    orthogonalize_w,
    orthogonalize_w_cluster,
    orthogonalize_w_cluster_iv,
    orthogonalize_w_iv,
)
from .utils.spatial import (
    InvalidSampledCovarianceError,
    LARGE_N_CAPM,
    LARGE_N_CAPN,
    LARGE_N_M,
    LARGE_N_THRESHOLD,
    get_cv,
    get_oms,
    lnget_oms,
    max_rp,
    normalize_s,
    set_oms_wfin,
    validate_large_n_seed,
    validate_scpc_method,
)

VALID_BACKENDS = {"auto", "rust", "fallback"}
RUST_API_VERSION = 6


def resolve_backend(backend: object) -> str:
    """Resolve the execution backend from explicit argument or `SPATIAL_BACKEND`."""
    if backend is None:
        env_backend = os.environ.get("SPATIAL_BACKEND", "auto") or "auto"
        if env_backend not in VALID_BACKENDS:
            raise ValueError(
                '`SPATIAL_BACKEND` must be one of "auto", "rust", or "fallback".'
            )
        return env_backend
    if not isinstance(backend, str) or backend not in VALID_BACKENDS:
        raise ValueError('`backend` must be one of "auto", "rust", or "fallback".')
    return backend


def is_weighted_or_non_ols_model(model: ModelLike) -> bool:
    """Return True if model metadata indicates weights or a non-OLS estimator."""
    if is_pyfixest_model(model):
        if bool(getattr(model, "_has_weights", False)):
            return True
        if getattr(model, "_weights_name", None) is not None:
            return True
        if getattr(model, "_method", None) != "feols":
            return True
        weights = getattr(model, "_weights", None)
        if weights is not None and not bool(
            np.all(np.asarray(weights, dtype=float) == 1.0)
        ):
            return True
        return False

    inner = getattr(model, "model", None)
    if not isinstance(inner, SmOLS):
        return True
    for attr in ("weights", "freq_weights", "var_weights"):
        vals = getattr(inner, attr, None)
        if vals is not None and not bool(np.all(np.asarray(vals, dtype=float) == 1.0)):
            return True
    return False


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
    proj = orthonormal_design_span(x_design)
    if proj.shape[1] == 0:
        return None, None
    proj = np.ascontiguousarray(proj, dtype=np.float64)
    return proj, proj


def compute_iv_projector(
    x_iv: MatrixLike,
    z_iv: MatrixLike,
    *,
    tol: float = 1e-10,
    demean=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute oblique `(U, V)` factors for `A -> A - X (X' P_Z X)^{-1} X' P_Z A`."""
    u, v = iv_projector_factors(x_iv, z_iv, tol=tol, demean=demean)
    return np.ascontiguousarray(u), np.ascontiguousarray(v)


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
    backend: str | None = None,
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
        backend: Execution backend (`"auto"`, `"rust"`, `"fallback"`, or
            `None` to consult `SPATIAL_BACKEND`).
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

    model_has_fe = has_fixest_fe(model)
    model_is_iv = is_fixest_iv_second_stage(model)
    if is_weighted_or_non_ols_model(model):
        raise NotImplementedError(
            "SCPC does not support weighted or non-OLS/2SLS fits. "
            "Pass an unweighted OLS or supported PyFixest 2SLS fit."
        )

    rust_mod = load_rust_backend() if resolved_backend != "fallback" else None
    if resolved_backend == "rust" and rust_mod is None:
        raise RuntimeError(
            "Rust backend requested (`backend='rust'`), but a compatible "
            f"`spatial_spur_rust` (API_VERSION={RUST_API_VERSION}) is unavailable. "
            "Install spatial-spur-rust >= 0.2.3 or use backend='fallback'."
        )

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

    iv_design = None
    model_mat_cond = None
    cond_include_intercept = False
    cond_fixef_id = None
    cond_residualize = None
    cond_ols_basis = None
    if not uncond:
        iv_design = get_fixest_iv_design(model) if model_is_iv else None
        model_mat = (
            get_scpc_model_matrix(model)
            if iv_design is None
            else iv_design["model_mat"]
        )
        cond_setup = get_conditional_projection_setup(
            model, model_mat, n=n, uncond=False, iv_design=iv_design
        )
        model_mat_cond = np.asarray(cond_setup.model_mat, dtype=float)
        cond_include_intercept = cond_setup.include_intercept
        cond_fixef_id = cond_setup.fixef_id
        cond_residualize = cond_setup.residualize
        cond_ols_basis = cond_setup.ols_basis
        if model_mat_cond.shape != (n, p):
            raise ValueError(
                f"Conditional projection has shape {model_mat_cond.shape}, "
                f"but the score matrix has shape {(n, p)}."
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

    fe_id_0based: np.ndarray | None = None
    fe_multiway = False
    if model_has_fe and not uncond:
        fe_raw = getattr(model, "_fe", None)
        if fe_raw is None:
            fe_multiway = True
        else:
            fe_arr = np.asarray(fe_raw)
            if fe_arr.ndim == 1:
                _, fe_inv = np.unique(fe_arr, return_inverse=True)
                fe_id_0based = np.ascontiguousarray(fe_inv, dtype=np.int64)
            elif fe_arr.ndim == 2 and fe_arr.shape[1] == 1:
                _, fe_inv = np.unique(fe_arr[:, 0], return_inverse=True)
                fe_id_0based = np.ascontiguousarray(fe_inv, dtype=np.int64)
            elif fe_arr.ndim == 2 and fe_arr.shape[1] == 0:
                fe_id_0based = None
            else:
                fe_multiway = True

    method_actual = (
        "approx"
        if (method == "approx" or (method == "auto" and neff >= LARGE_N_THRESHOLD))
        else "exact"
    )
    is_rust_eligible = (
        method_actual in {"exact", "approx"} and not fe_multiway and k_use >= 1
    )

    fallback_reason = None
    if resolved_backend != "fallback":
        if rust_mod is None:
            fallback_reason = "missing_or_incompatible_native_extension"
        elif not is_rust_eligible:
            fallback_reason = (
                "multiway_fixed_effects" if fe_multiway else "non_positive_ncoef"
            )
            if resolved_backend == "rust":
                detail = (
                    "multi-way absorbed fixed-effect demeaning"
                    if fe_multiway
                    else "non-positive ncoef"
                )
                raise NotImplementedError(
                    f"Rust backend does not support {detail} yet."
                )

        if rust_mod is not None and is_rust_eligible:
            coef_use = np.ascontiguousarray(coef[:k_use], dtype=np.float64)
            wj_mat = np.ascontiguousarray(
                neff * (s @ bread_inv[:k_use, :].T) + coef_use[None, :],
                dtype=np.float64,
            )
            if model_mat_cond is None:
                rust_xj = None
                rust_xjs = None
                proj_u = None
                proj_v = None
                cl_arg = None
                fe_arg = None
            else:
                xj_mat = np.ascontiguousarray(
                    neff * (model_mat_cond @ bread_inv[:k_use, :].T),
                    dtype=np.float64,
                )
                if cluster is not None:
                    if cl_idx is None:
                        raise ValueError(
                            "Internal error: clustered SCPC is missing cluster indices."
                        )
                    xj_sq_sum = np.zeros((neff, k_use), dtype=np.float64)
                    np.add.at(xj_sq_sum, cl_idx, xj_mat**2)
                    cl_denom = np.sqrt(xj_sq_sum[cl_idx, :])
                    xjs_mat = np.ascontiguousarray(
                        np.divide(
                            xj_mat,
                            cl_denom,
                            out=np.zeros_like(xj_mat),
                            where=cl_denom > 0.0,
                        ),
                        dtype=np.float64,
                    )
                    cl_arg = np.ascontiguousarray(cl_idx, dtype=np.int64)
                else:
                    xjs_mat = np.where(xj_mat < 0.0, -1.0, 1.0)
                    cl_arg = None

                if iv_design is not None:
                    demean = make_pyfixest_demeaner(model) if model_has_fe else None
                    fe_arg = fe_id_0based
                    proj_v, proj_u = compute_iv_projector(
                        iv_design["X"], iv_design["Z"], demean=demean
                    )
                    rust_xj = xj_mat
                    rust_xjs = xjs_mat
                else:
                    fe_arg = fe_id_0based
                    if cond_ols_basis is not None:
                        proj_u = proj_v = np.ascontiguousarray(cond_ols_basis)
                    else:
                        proj_u, proj_v = compute_orthogonal_projector(
                            model_mat_cond,
                            include_intercept=cond_include_intercept,
                        )
                    rust_xj = xj_mat
                    rust_xjs = xjs_mat

            if method_actual == "approx":
                coords_norm, perm = normalize_s(coords, latlong)
                perm_idx = np.asarray(perm, dtype=np.int64)
                coords_mat = np.ascontiguousarray(coords_norm, dtype=np.float64)
                wj_mat = np.ascontiguousarray(wj_mat[perm_idx, :], dtype=np.float64)
                if not uncond:
                    if cluster is not None and cl_arg is not None:
                        inv_perm = np.empty(len(perm_idx), dtype=np.int64)
                        inv_perm[perm_idx] = np.arange(len(perm_idx), dtype=np.int64)
                        cl_arg = np.ascontiguousarray(inv_perm[cl_arg], dtype=np.int64)
                    else:
                        rust_xj = (
                            None
                            if rust_xj is None
                            else np.ascontiguousarray(
                                rust_xj[perm_idx, :], dtype=np.float64
                            )
                        )
                        rust_xjs = (
                            None
                            if rust_xjs is None
                            else np.ascontiguousarray(
                                rust_xjs[perm_idx, :], dtype=np.float64
                            )
                        )
                        if proj_u is not None:
                            proj_u = np.ascontiguousarray(
                                proj_u[perm_idx, :], dtype=np.float64
                            )
                        if proj_v is not None:
                            proj_v = np.ascontiguousarray(
                                proj_v[perm_idx, :], dtype=np.float64
                            )
                        if fe_arg is not None:
                            fe_arg = np.ascontiguousarray(
                                fe_arg[perm_idx], dtype=np.int64
                            )
                try:
                    rust_res = rust_mod.run_approx_scpc(
                        coords_mat,
                        coef_use,
                        wj_mat,
                        float(avc),
                        latlong=bool(latlong),
                        uncond=bool(uncond),
                        cvs=bool(cvs),
                        xj=rust_xj,
                        xjs=rust_xjs,
                        proj_u=proj_u,
                        proj_v=proj_v,
                        nthreads=resolved_nthreads,
                        cluster_id=cl_arg,
                        fe_id=fe_arg,
                        seed=int(large_n_seed),
                        landmarks=LARGE_N_M,
                        samples=LARGE_N_CAPN,
                        pair_count=LARGE_N_CAPM,
                    )
                except rust_mod.InvalidSampledCovarianceError as error:
                    raise InvalidSampledCovarianceError(
                        f"{error}. Try method='exact'."
                    ) from error
                except rust_mod.TiedEigenspaceError:
                    if resolved_backend != "auto":
                        raise
                    rust_res = None
                    fallback_reason = "tied_eigenspace"
            else:
                coords_mat = np.ascontiguousarray(coords, dtype=np.float64)
                rust_res = rust_mod.run_exact_scpc(
                    coords_mat,
                    coef_use,
                    wj_mat,
                    float(avc),
                    latlong=bool(latlong),
                    uncond=bool(uncond),
                    cvs=bool(cvs),
                    xj=rust_xj,
                    xjs=rust_xjs,
                    proj_u=proj_u,
                    proj_v=proj_v,
                    nthreads=resolved_nthreads,
                    cluster_id=cl_arg,
                    fe_id=fe_arg,
                )

            if rust_res is not None and not (
                rust_res["tied"] and resolved_backend == "auto"
            ):
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
                    method=method_actual,
                    large_n_seed=large_n_seed,
                    backend="rust",
                    tied=bool(rust_res["tied"]),
                )
            if rust_res is not None and rust_res["tied"]:
                fallback_reason = "tied_eigenspace"

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

    cond_residualize_use = cond_residualize
    cond_fixef_id_use = cond_fixef_id
    if (
        not uncond
        and cluster is None
        and not np.array_equal(perm, np.arange(len(perm), dtype=int))
    ):
        inv_perm = np.empty(len(perm), dtype=int)
        inv_perm[perm] = np.arange(len(perm), dtype=int)
        if cond_residualize is not None:
            base_resid = cond_residualize

            def _permuted_resid(mat: MatrixLike) -> np.ndarray:
                arr = np.asarray(mat, dtype=float)
                return np.asarray(base_resid(arr[inv_perm, ...]), dtype=float)[
                    perm, ...
                ]

            cond_residualize_use = _permuted_resid
        if cond_fixef_id is not None:
            fe_arr = np.asarray(cond_fixef_id)
            cond_fixef_id_use = fe_arr[perm, ...] if fe_arr.ndim > 1 else fe_arr[perm]

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

        wj_norm = float(np.linalg.norm(wj_perm))
        denom = math.sqrt(float(np.sum((wfin[:, 1:].T @ wj_perm) ** 2)))
        if (
            not math.isfinite(denom)
            or not math.isfinite(wj_norm)
            or wj_norm == 0.0
            or denom <= 1e-12 * wj_norm
        ):
            raise ValueError("SCPC variance projection is degenerate.")
        tau_u = math.sqrt(q) * float(np.dot(wfin[:, 0], wj_perm)) / denom
        se = denom / (math.sqrt(q) * math.sqrt(neff))
        p_u = max_rp(omsfin, q, abs(tau_u) / math.sqrt(q))[0]

        if model_mat_cond is not None:
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
                if cond_residualize is not None:
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
                xjs = np.where(xj < 0.0, -1.0, 1.0)
                if cond_residualize_use is not None:
                    wx = orthogonalize_w_iv(
                        wfin,
                        xj,
                        xjs,
                        residualize=cond_residualize_use,
                    )
                else:
                    wx = orthogonalize_w(
                        wfin,
                        xj,
                        xjs,
                        model_mat_cond[perm, :],
                        include_intercept=cond_include_intercept,
                        fixef_id=cond_fixef_id_use,
                    )

            if spc.large_n:
                # this reuses the sampled omega construction from the large-n setup.
                omsx, large_n_random_state = lnget_oms(
                    spc.coords,
                    spc.c0,
                    spc.cmax,
                    wx,
                    1.2,
                    capM=LARGE_N_CAPM,
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
        tied=spc.tied or fallback_reason == "tied_eigenspace",
        fallback_reason=fallback_reason,
    )
