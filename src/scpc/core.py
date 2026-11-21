from __future__ import annotations

from collections.abc import Sequence
import importlib
import math
import os
from typing import Any

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
    get_pyfixest_raw_ols_design,
    get_scpc_model_matrix,
    has_fixest_fe,
    is_fixest_iv_second_stage,
    is_pyfixest_model,
    is_pyfixest_multi,
    resolve_coords_input,
)
from .utils.matrix import (
    iv_projector_factors,
    make_ols_residualizer,
    ols_design_span,
    project_clustered_weights,
    project_conditional_weights,
)
from .utils.spatial import (
    InvalidSampledCovarianceError,
    LARGE_N_CAPM,
    LARGE_N_THRESHOLD,
    get_cv,
    get_oms,
    max_rp,
    sample_covariance_pairs,
    sampled_covariance_grid,
    set_oms_wfin,
    validate_large_n_seed,
    validate_scpc_method,
)

VALID_BACKENDS = {"auto", "rust", "fallback"}
RUST_API_VERSION = 7


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
    proj = ols_design_span(x_design, require_full_rank=True)
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

    if not 0.001 < avc < 0.99:
        raise ValueError("Option avc() must be in (0.001, 0.99).")
    if ncoef is not None and (
        isinstance(ncoef, bool) or not isinstance(ncoef, (int, np.integer)) or ncoef < 1
    ):
        raise ValueError("`ncoef` must be a positive integer or None.")
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
            "Build the matching development adapter or use backend='fallback'."
        )

    obs_index = get_obs_index(model, data)

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
    if p < 1:
        raise ValueError("SCPC requires at least one fitted coefficient.")
    neff = n

    coord_info = resolve_coords_input(data, obs_index, lon, lat, coords_euclidean)
    coords = np.asarray(coord_info.coords, dtype=float)
    latlong = coord_info.latlong

    cl_idx: np.ndarray | None = None
    if cluster is not None:
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("`cluster` must be a single column name.")
        if cluster not in data.columns:
            raise ValueError(f"Cluster variable not found in data: {cluster}")
        cl_values = data.iloc[np.asarray(obs_index, dtype=int) - 1][cluster]
        if cl_values.isna().any():
            raise ValueError("Cluster labels must be present in the estimation sample.")
        _, first_index, cl_idx = np.unique(
            cl_values, return_index=True, return_inverse=True
        )

        coord_first = coords[first_index, :]
        if not np.array_equal(coords, coord_first[cl_idx]):
            raise ValueError(
                "Coordinates vary within clusters. Supply one coordinate per cluster "
                "for all observations in the estimation sample."
            )

        coords = coord_first
        neff = len(coord_first)

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
    coef_use = np.ascontiguousarray(coef[:k_use], dtype=np.float64)
    iv_design = get_fixest_iv_design(model) if model_is_iv and not uncond else None
    design = None
    if not uncond:
        if iv_design is not None:
            design = iv_design["X"]
        elif model_has_fe:
            design = get_pyfixest_raw_ols_design(model, n)
        else:
            design = get_scpc_model_matrix(model)

    fallback_reason = None
    if resolved_backend != "fallback":
        if rust_mod is None:
            fallback_reason = "missing_or_incompatible_native_extension"
        elif fe_multiway:
            fallback_reason = "multiway_fixed_effects"
            if resolved_backend == "rust":
                raise NotImplementedError(
                    "Rust backend does not support multi-way absorbed fixed-effect demeaning yet."
                )

        if rust_mod is not None and not fe_multiway:
            try:
                rust_res = rust_mod.run_model_scpc(
                    coords,
                    coef_use,
                    bread_inv[:k_use],
                    s,
                    float(avc),
                    design=design,
                    instruments=None if iv_design is None else iv_design["Z"],
                    direction_design=None
                    if iv_design is None
                    else iv_design["model_mat"],
                    cluster_id=cl_idx,
                    fe_id=fe_id_0based,
                    latlong=bool(latlong),
                    cvs=bool(cvs),
                    nthreads=resolved_nthreads,
                    seed=int(large_n_seed) if method_actual == "approx" else None,
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

    if cl_idx is not None:
        s_cluster = np.zeros((neff, s.shape[1]), dtype=float)
        np.add.at(s_cluster, cl_idx, s)
        s = s_cluster
    wj_mat = neff * (s @ bread_inv[:k_use].T) + coef_use
    cond_residualize = None
    xj_mat = None
    if design is not None:
        direction_design = iv_design["model_mat"] if iv_design is not None else design
        cond_setup = get_conditional_projection_setup(
            model,
            direction_design,
            n=n,
            uncond=False,
            iv_design=iv_design,
            raw_ols_design=model_has_fe and iv_design is None,
        )
        model_mat_cond = np.asarray(cond_setup.model_mat, dtype=float)
        if model_mat_cond.shape != (n, p):
            raise ValueError(
                f"Conditional projection has shape {model_mat_cond.shape}, "
                f"but the score matrix has shape {(n, p)}."
            )
        cond_residualize = cond_setup.residualize
        if cond_residualize is None:
            cond_residualize = make_ols_residualizer(
                model_mat_cond, require_full_rank=True
            )
        xj_mat = neff * (model_mat_cond @ bread_inv[:k_use].T)

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
    conditional_pairs = None
    if spc.large_n and xj_mat is not None:
        assert spc.random_state is not None
        conditional_pairs, _ = sample_covariance_pairs(
            spc.coords, LARGE_N_CAPM, spc.random_state, latlong
        )

    # Nyström reorders sites. Keep observation-space factors in the fitted order.
    cl_idx_scpc = cl_idx
    site_order = np.arange(len(perm))
    if cond_residualize is not None and not np.array_equal(perm, site_order):
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = site_order
        if cl_idx is not None:
            cl_idx_scpc = inv_perm[cl_idx]
        else:
            base_resid = cond_residualize

            def _permuted_resid(mat: MatrixLike) -> np.ndarray:
                arr = np.asarray(mat, dtype=float)
                return np.asarray(base_resid(arr[inv_perm]), dtype=float)[perm]

            cond_residualize = _permuted_resid

    out = np.full((k_use, 6), np.nan)
    levs = np.array([0.32, 0.10, 0.05, 0.01], dtype=float)
    cvs_mat = np.full((k_use, 4), np.nan) if cvs else None
    cvs_uncond = (
        np.array([get_cv(omsfin, q, level) for level in levs], dtype=float)
        if cvs
        else None
    )

    for j in range(k_use):
        wj = wj_mat[:, j]
        wj_perm = wj[perm]

        scale = float(np.max(np.abs(wj_perm)))
        if not math.isfinite(scale) or scale == 0:
            raise ValueError("SCPC variance projection is degenerate.")
        scaled_wj = wj_perm / scale
        wj_norm = float(np.linalg.norm(scaled_wj))
        denom = float(np.linalg.norm(wfin[:, 1:].T @ scaled_wj))
        if (
            not math.isfinite(denom)
            or not math.isfinite(wj_norm)
            or wj_norm == 0.0
            or denom <= 1e-12 * wj_norm
        ):
            raise ValueError("SCPC variance projection is degenerate.")
        tau_u = math.sqrt(q) * float(np.dot(wfin[:, 0], scaled_wj)) / denom
        se = (denom / (math.sqrt(q) * math.sqrt(neff))) * scale
        p_u = max_rp(omsfin, q, abs(tau_u) / math.sqrt(q))[0]

        if xj_mat is not None and cond_residualize is not None:
            if cl_idx_scpc is not None:
                wx = project_clustered_weights(
                    wfin,
                    cl_idx_scpc,
                    xj_mat[:, j],
                    residualize=cond_residualize,
                    rescale=True,
                )
            else:
                xj = xj_mat[perm, j]
                xjs = np.where(xj < 0.0, -1.0, 1.0)
                wx = project_conditional_weights(
                    wfin,
                    xj,
                    xjs,
                    residualize=cond_residualize,
                    rescale=True,
                )

            # A common positive scale leaves all conditional rejection probabilities
            # unchanged and keeps the covariance products in floating-point range.
            wx /= np.max(np.abs(wx))
            if conditional_pairs is not None:
                omsx = sampled_covariance_grid(
                    wx, spc.c0, spc.cmax, 1.2, conditional_pairs
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
