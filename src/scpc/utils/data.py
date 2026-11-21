from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from formulaic import ModelSpec

from ..types import (
    ArrayLike,
    ConditionalProjectionSetup,
    CoordinateData,
    DataFrameLike,
    FixestSpec,
    MatrixLike,
    ModelLike,
)
from .matrix import (
    make_fixef_demeaner,
    make_iv_residualizer,
    make_orthogonal_residualizer,
    ols_design_span,
)


def is_pyfixest_model(model: ModelLike) -> bool:
    """Check whether a fitted model comes from pyfixest."""
    return type(model).__module__.startswith("pyfixest.")


def is_pyfixest_multi(model: ModelLike) -> bool:
    """Check whether a pyfixest object bundles multiple fitted models."""
    return is_pyfixest_model(model) and type(model).__name__ == "FixestMulti"


def get_pyfixest_coef_names(model: ModelLike) -> list[str]:
    """Return the stored pyfixest coefficient names as plain strings."""
    return [str(name) for name in getattr(model, "_coefnames", [])]


def get_coef_names(model: ModelLike) -> list[str]:
    """Return coefficient names in coefficient order."""
    if is_pyfixest_model(model):
        return get_pyfixest_coef_names(model)

    return [str(name) for name in model.model.exog_names]


def get_pyfixest_data(model: ModelLike) -> DataFrameLike:
    """Return the stored pyfixest estimation sample.

    The iv helpers need the original sample values because scpcr rebuilds the
    raw X and Z matrices before any fixed-effect demeaning happens.
    """
    data = getattr(model, "_data", None)
    if data is None:
        raise ValueError(
            "PyFixest models passed to `scpc()` must keep their estimation "
            "sample; fit with `store_data=True` and `lean=False`."
        )
    return data


def get_pyfixest_named_columns(
    data: DataFrameLike,
    names: list[str],
    *,
    n: int,
    context: str,
) -> np.ndarray:
    """Extract named columns from the stored pyfixest data sample."""
    if not names:
        return np.empty((n, 0), dtype=float)

    missing = [name for name in names if name not in data.columns]
    if missing:
        raise ValueError(
            f"{context} columns are missing from the stored pyfixest data."
        )

    values = np.asarray(data.loc[:, names], dtype=float)
    if values.ndim != 2 or values.shape != (n, len(names)):
        raise ValueError(f"{context} columns could not be aligned to the model sample.")
    return values


def make_pyfixest_demeaner(model: ModelLike):
    """Build the unweighted conditional projector for the fitted FE sample."""
    fe = model._fe
    return make_fixef_demeaner(fe, len(fe))


def _get_pyfixest_formula_context(model: ModelLike) -> dict[str, object]:
    """Build the formulaic evaluation context including PyFixest transforms."""
    from pyfixest.estimation.formula.factor_interaction import factor_interaction
    from pyfixest.estimation.formula.utils import _factorize, log

    context: dict[str, object] = {
        "log": log,
        "i": factor_interaction,
        "__fixed_effect__": _factorize,
    }
    user_context = getattr(model, "_context", None)
    if isinstance(user_context, Mapping):
        context.update(user_context)
    return context


def _eval_pyfixest_rhs_design(
    rhs_spec: ModelSpec, data: DataFrameLike, model: ModelLike
):
    """Evaluate a PyFixest RHS ModelSpec with a clean encoder state for i() terms."""
    context = _get_pyfixest_formula_context(model)
    formula = getattr(rhs_spec, "formula", None)
    if formula is not None:
        return formula.get_model_matrix(data, context=context)
    return rhs_spec.get_model_matrix(data, context=context)


def get_pyfixest_raw_ols_design(model: ModelLike, n: int) -> np.ndarray:
    """Recreate the fitted columns before PyFixest's iterative absorption."""
    data = get_pyfixest_data(model)
    design = _eval_pyfixest_rhs_design(model._model_spec.second_stage.rhs, data, model)
    if not design.index.equals(data.index):
        raise ValueError("PyFixest OLS design does not line up with the fitted sample.")
    return get_pyfixest_named_columns(
        design, get_pyfixest_coef_names(model), n=n, context="PyFixest OLS regressor"
    )


def is_fixest_iv_second_stage(model: ModelLike) -> bool:
    """Check whether a model is the fixest-style IV second stage.

    This mirrors `scpcR:::.is_fixest_iv_second_stage()`, which checks whether
    a fixest fit is the active second-stage IV model.

    Args:
        model: Fitted model object.

    Returns:
        Whether the fitted model is an IV second-stage regression.
    """
    return is_pyfixest_model(model) and bool(getattr(model, "_is_iv", False))


def get_obs_index(model: ModelLike, data: DataFrameLike) -> ArrayLike:
    """Recover which rows of the input data were used by the fitted model.

    Fitted models often drop observations because of missing data or internal
    preprocessing. SCPC needs to know exactly which rows survived so that
    score contributions, regressors, and coordinates are all aligned to the
    same observations before any spatial calculations are done.

    Args:
        model: Fitted model object.
        data: Original data passed to the model.

    Returns:
        One-based row positions locating the model observations in `data`.

    Raises:
        ValueError: If the model rows cannot be mapped back to the provided
            data.
    """
    if is_pyfixest_model(model):
        fit_data = getattr(model, "_data", None)
        if fit_data is None:
            raise ValueError(
                "PyFixest models passed to `scpc()` must keep their estimation "
                "sample; fit with `store_data=True` and `lean=False`."
            )

        obs_index = np.asarray(fit_data.index)
        if not np.issubdtype(obs_index.dtype, np.integer):
            raise ValueError(
                "PyFixest estimation sample indices must be integer row positions."
            )
        if np.any(obs_index < 0) or np.any(obs_index >= len(data)):
            raise ValueError(
                "PyFixest estimation sample indices are outside the row range "
                "of `data`."
            )

        # pyfixest stores zero-based row positions, so we shift to the
        # one-based convention used by the r helpers.
        return obs_index.astype(int, copy=False) + 1

    model_data = model.model.data
    row_labels = model_data.row_labels
    if row_labels is None:
        positions = np.arange(len(data))
        removed = getattr(model_data, "missing_row_idx", [])
        if len(removed):
            if max(removed) >= len(data):
                raise ValueError("Could not align the model rows with `data`.")
            positions = np.delete(positions, removed)
        if len(positions) != int(model.nobs):
            raise ValueError("Could not align the model rows with `data`.")
        return positions + 1

    if not data.index.is_unique:
        raise ValueError(
            "`data.index` must be unique to map model rows back to `data`."
        )

    # statsmodels gives us row labels, so we map them back to row positions here.
    obs_index = data.index.get_indexer(row_labels)
    if np.any(obs_index < 0):
        raise ValueError("Could not map model frame rows back to `data`.")

    return obs_index + 1


def get_scpc_model_matrix(model: ModelLike) -> MatrixLike:
    """Recover the regression design matrix used for SCPC adjustments.

    SCPC needs a matrix representation of the fitted regressors because the
    conditional adjustment works by removing covariate-driven variation from
    spatial directions. This helper extracts that aligned design matrix.

    Args:
        model: Fitted model object.

    Returns:
        The model matrix aligned to the fitted coefficients.
    """
    if is_pyfixest_model(model):
        if is_pyfixest_multi(model):
            raise ValueError(
                "`scpc()` only accepts a single fitted pyfixest model, not FixestMulti."
            )

        if is_fixest_iv_second_stage(model):
            return np.asarray(get_fixest_iv_design(model)["model_mat"], dtype=float)

        model_mat = np.asarray(getattr(model, "_X", None), dtype=float)
        coef_names = get_pyfixest_coef_names(model)
        if model_mat.ndim != 2 or model_mat.shape[1] != len(coef_names):
            raise ValueError(
                "PyFixest design matrix columns do not line up with the "
                "stored coefficients."
            )
        return model_mat

    return np.asarray(model.model.exog, dtype=float)


def get_fixest_score_matrix(model: ModelLike) -> MatrixLike:
    """Return the coefficient-space score matrix for a pyfixest fit.

    This fills the role of `sandwich::estfun(model)` for pyfixest models. For
    IV fits, pyfixest stores the raw `Z * u` scores, so this helper maps them
    into coefficient space before SCPC uses them.

    Args:
        model: Fitted pyfixest model object.

    Returns:
        The score matrix aligned to the fitted coefficients.

    Raises:
        ValueError: If the stored pyfixest score objects are missing or
            malformed.
    """
    if not is_pyfixest_model(model):
        raise ValueError("`get_fixest_score_matrix()` requires a pyfixest model.")

    scores = np.asarray(getattr(model, "_scores", None), dtype=float)
    if scores.ndim != 2:
        raise ValueError("Could not extract the pyfixest score matrix.")

    if is_fixest_iv_second_stage(model):
        tzzinv = np.asarray(getattr(model, "_tZZinv", None), dtype=float)
        tzx = np.asarray(getattr(model, "_tZX", None), dtype=float)
        if tzzinv.ndim != 2 or tzx.ndim != 2:
            raise ValueError("Could not extract the pyfixest IV score mapping.")

        # pyfixest stores z * u here, so we map it into coefficient space.
        scores = scores @ tzzinv @ tzx

    if not np.isfinite(scores).all():
        raise ValueError("The pyfixest score matrix contains non-finite values.")

    return scores


def get_fixest_bread_inv(model: ModelLike) -> MatrixLike:
    """Return the coefficient-space bread matrix for a pyfixest fit.

    This fills the role of `sandwich::bread(model) / nobs(model)` for
    pyfixest models.

    Args:
        model: Fitted pyfixest model object.

    Returns:
        The bread matrix aligned to the fitted coefficients.

    Raises:
        ValueError: If the stored pyfixest bread matrix is missing or malformed.
    """
    if not is_pyfixest_model(model):
        raise ValueError("`get_fixest_bread_inv()` requires a pyfixest model.")

    bread_inv = np.asarray(getattr(model, "_bread", None), dtype=float)
    if bread_inv.ndim != 2 or not np.isfinite(bread_inv).all():
        raise ValueError("Could not extract the pyfixest bread matrix.")

    return bread_inv


def has_fixest_fe(model: ModelLike) -> bool:
    """Check whether a fixest-style model uses absorbed fixed effects.

    Fixed effects matter here because they change how the regression space
    should be represented before conditional SCPC orthogonalization. This
    helper makes that branch explicit.

    Args:
        model: Fitted model object.

    Returns:
        Whether the model includes absorbed fixed effects.
    """
    if is_pyfixest_model(model):
        return bool(getattr(model, "_has_fixef", False))

    fixef_vars = getattr(model, "fixef_vars", None)
    return fixef_vars is not None and len(fixef_vars) > 0


def get_fixest_iv_design(
    model: ModelLike,
) -> FixestSpec:
    """Extract the stored IV design objects from a pyfixest fit.

    This mirrors `scpcR:::.get_fixest_iv_design()`. In the R code that helper
    rebuilds the design from `stats::model.matrix(..., type = ...)`; in
    Python, we rebuild the same raw X, Z, and second-stage matrices from the
    stored sample, the raw endogenous variable, and the fitted first stage.

    Args:
        model: Fitted pyfixest IV model.

    Returns:
        The aligned `X`, `Z`, and second-stage model matrix, together with the
        coefficient names and fixed-effect flag.

    Raises:
        ValueError: If the model is not an IV second-stage fit or its stored
            matrices are malformed.
    """
    if not is_fixest_iv_second_stage(model):
        raise ValueError(
            "`get_fixest_iv_design()` requires a pyfixest IV second-stage model."
        )

    data = get_pyfixest_data(model)
    coef_names = get_pyfixest_coef_names(model)
    z_names = [str(name) for name in getattr(model, "_coefnames_z", [])]
    raw_endog = np.asarray(getattr(model, "_endogvar", None), dtype=float)
    first_stage = getattr(model, "_model_1st_stage", None)
    if first_stage is None:
        raise ValueError("Could not extract the stored pyfixest first-stage fit.")
    fit_endog = np.asarray(getattr(first_stage, "_Y_hat_link", None), dtype=float)

    if raw_endog.ndim == 1:
        raw_endog = raw_endog[:, None]
    if fit_endog.ndim == 1:
        fit_endog = fit_endog[:, None]
    if raw_endog.ndim != 2 or fit_endog.ndim != 2:
        raise ValueError("Could not extract the stored pyfixest first-stage objects.")

    n = len(data)
    if raw_endog.shape[0] != n or fit_endog.shape[0] != n:
        raise ValueError(
            "PyFixest first-stage objects do not line up with the stored data sample."
        )

    exo_names = [name for name in coef_names if name in z_names]
    endo_names = [name for name in coef_names if name not in z_names]
    inst_names = [name for name in z_names if name not in coef_names]

    if len(endo_names) != raw_endog.shape[1] or len(endo_names) != fit_endog.shape[1]:
        raise ValueError(
            "PyFixest endogenous regressors do not line up with the stored "
            "first-stage objects."
        )

    model_spec = getattr(model, "_model_spec", None)
    exo_source = data
    if (
        exo_names
        and (
            any(name not in data.columns for name in exo_names)
            or "Intercept" in exo_names
        )
        and model_spec is not None
        and hasattr(model_spec, "second_stage")
    ):
        exo_design = _eval_pyfixest_rhs_design(model_spec.second_stage.rhs, data, model)
        if exo_design.index.equals(data.index):
            exo_source = exo_design
    inst_source = data
    if (
        inst_names
        and any(name not in data.columns for name in inst_names)
        and model_spec is not None
        and hasattr(model_spec, "first_stage")
    ):
        inst_design = _eval_pyfixest_rhs_design(model_spec.first_stage.rhs, data, model)
        if inst_design.index.equals(data.index):
            inst_source = inst_design

    def _source_col(source: DataFrameLike, name: str, context: str) -> np.ndarray:
        if name in source.columns:
            col = np.asarray(source[name], dtype=float)
            if col.shape != (n,):
                raise ValueError(f"{context} column '{name}' has unexpected shape.")
            return col
        if name == "Intercept":
            return np.ones(n, dtype=float)
        raise ValueError(f"{context} column '{name}' is missing from stored data.")

    raw_col_map: dict[str, np.ndarray] = {}
    fit_col_map: dict[str, np.ndarray] = {}
    for name in exo_names:
        col = _source_col(exo_source, name, "PyFixest IV exogenous regressor")
        raw_col_map[name] = col
        fit_col_map[name] = col
    for idx, name in enumerate(endo_names):
        raw_col_map[name] = raw_endog[:, idx]
        fit_col_map[name] = fit_endog[:, idx]

    inst = get_pyfixest_named_columns(
        inst_source,
        inst_names,
        n=n,
        context="PyFixest IV instrument",
    )
    exo_for_z = (
        np.column_stack([raw_col_map[name] for name in exo_names])
        if exo_names
        else np.empty((n, 0), dtype=float)
    )
    x = np.column_stack([raw_col_map[name] for name in coef_names])
    z = np.column_stack((exo_for_z, inst))
    model_mat = np.column_stack([fit_col_map[name] for name in coef_names])

    if x.shape[1] != len(coef_names):
        raise ValueError(
            "PyFixest second-stage design columns do not line up with the "
            "stored coefficients."
        )
    if z.shape[1] != len(z_names):
        raise ValueError(
            "PyFixest instrument design columns do not line up with the "
            "stored instrument names."
        )
    if (
        not np.isfinite(x).all()
        or not np.isfinite(z).all()
        or not np.isfinite(model_mat).all()
    ):
        raise ValueError("PyFixest IV design extraction produced non-finite values.")

    return {
        "X": x,
        "Z": z,
        "model_mat": model_mat,
        "coef_names": coef_names,
        "fixef_id": None,
        "has_fixef": has_fixest_fe(model),
    }


def get_conditional_projection_setup(
    model: ModelLike,
    model_mat: MatrixLike,
    n: int,
    uncond: bool,
    iv_design: FixestSpec | None = None,
    raw_ols_design: bool = False,
) -> ConditionalProjectionSetup:
    """Prepare the regression-side objects for conditional SCPC.

    Conditional SCPC needs a version of the regressor space that matches the
    fitted coefficient space exactly, including any demeaning implied by fixed
    effects. This helper standardizes that information before the spatial
    directions are orthogonalized against it.

    Args:
        model: Fitted model object.
        model_mat: Raw model matrix extracted from the model.
        n: Expected number of active observations.
        uncond: Whether unconditional inference was requested.
        iv_design: Previously extracted IV design, if available.
        raw_ols_design: Whether `model_mat` already contains the active raw
            PyFixest OLS design before fixed-effect absorption.

    Returns:
        The normalized conditional projection setup.

    Raises:
        ValueError: Raised later if the regression objects cannot be aligned.
    """
    # The fitted model matrix already contains any intercept.
    setup = ConditionalProjectionSetup(
        model_mat=np.asarray(model_mat, dtype=float),
        include_intercept=False,
        fixef_id=None,
    )
    if uncond:
        return setup

    if is_pyfixest_model(model):
        model_mat = np.asarray(model_mat, dtype=float)
        if model_mat.shape[0] != n:
            raise ValueError(
                "Internal error: pyfixest conditional regressors have an "
                "incompatible row count."
            )

        if is_fixest_iv_second_stage(model):
            design = get_fixest_iv_design(model) if iv_design is None else iv_design
            model_mat_iv = np.asarray(design["model_mat"], dtype=float)
            x_iv = np.asarray(design["X"], dtype=float)
            z_iv = np.asarray(design["Z"], dtype=float)

            if bool(design["has_fixef"]):
                demean = make_pyfixest_demeaner(model)
                model_mat_iv = np.asarray(
                    demean(
                        model_mat_iv, context="pyfixest IV second-stage model matrix"
                    ),
                    dtype=float,
                )
            else:
                demean = None

            return ConditionalProjectionSetup(
                model_mat=model_mat_iv,
                include_intercept=not bool(design["has_fixef"]),
                fixef_id=np.asarray(getattr(model, "_fe"))
                if demean is not None
                else None,
                residualize=defer_residualizer(
                    lambda: make_iv_residualizer(
                        x_iv, z_iv, demean=demean, transpose=True
                    )
                ),
                is_iv=True,
            )

        if has_fixest_fe(model):
            demean = make_pyfixest_demeaner(model)
            raw = model_mat if raw_ols_design else get_pyfixest_raw_ols_design(model, n)
            model_mat = demean(
                raw, context="PyFixest OLS regressor", drop_absorbed=True
            )
            ols_basis = ols_design_span(model_mat, require_full_rank=True)
            return ConditionalProjectionSetup(
                model_mat=model_mat,
                include_intercept=False,
                fixef_id=np.asarray(getattr(model, "_fe")),
                residualize=make_orthogonal_residualizer(ols_basis, demean=demean),
                ols_basis=ols_basis,
            )

        return setup

    if not has_fixest_fe(model):
        return setup

    raise ValueError(
        "Conditional setup for fixest-style fixed effects is not implemented."
    )


def defer_residualizer(
    build: Callable[[], Callable[[MatrixLike], np.ndarray]],
) -> Callable[[MatrixLike], np.ndarray]:
    """Build the fallback projection only if it is used, once per fit."""
    cached_build = cache(build)

    def residualize(values: MatrixLike) -> np.ndarray:
        return cached_build()(values)

    return residualize


def resolve_coords_input(
    data: DataFrameLike,
    obs_index: ArrayLike,
    lon: str | None,
    lat: str | None,
    coords_euclidean: Sequence[str] | None,
) -> CoordinateData:
    """Normalize the location information used for spatial distances.

    This helper turns the user's coordinate arguments into one clean numeric
    representation aligned to the active observations. It is the place where
    SCPC decides whether the problem is being described in longitude/latitude
    space or in ordinary Euclidean coordinates.

    Args:
        data: Original input data.
        obs_index: Indices of the observations used by the model.
        lon: Longitude column name for geodesic coordinates.
        lat: Latitude column name for geodesic coordinates.
        coords_euclidean: Euclidean coordinate column names.

    Returns:
        The normalized coordinate representation used by the spatial engine.

    Raises:
        ValueError: Raised later for invalid or inconsistent coordinate input.
    """
    use_geodesic = lon is not None or lat is not None
    use_euclidean = coords_euclidean is not None

    if use_geodesic and use_euclidean:
        raise ValueError("Specify either `lon`/`lat` or `coords_euclidean`, not both.")
    if not use_geodesic and not use_euclidean:
        raise ValueError("Specify coordinates via `lon`/`lat` or `coords_euclidean`.")

    # obs_index follows the R helper and is therefore 1-based.
    obs_index = np.asarray(obs_index, dtype=int) - 1

    if use_geodesic:
        if lon is None or lat is None:
            raise ValueError("For geodesic coordinates, provide both `lon` and `lat`.")
        if not isinstance(lon, str) or not lon or not isinstance(lat, str) or not lat:
            raise ValueError("`lon` and `lat` must each be a single column name.")

        miss = sorted(set([lon, lat]) - set(data.columns))
        if miss:
            raise ValueError(
                f"Coordinate variables not found in data: {', '.join(miss)}"
            )

        coords = data.iloc[obs_index][[lon, lat]]
        if not all(np.issubdtype(dtype, np.number) for dtype in coords.dtypes):
            raise ValueError("`lon` and `lat` must reference numeric columns.")
        if not np.isfinite(np.asarray(coords, dtype=float)).all():
            raise ValueError("Geodesic coordinates must be finite.")
        if (coords[lon] < -180).any() or (coords[lon] > 180).any():
            raise ValueError("Longitude values must be in [-180, 180].")
        if (coords[lat] < -90).any() or (coords[lat] > 90).any():
            raise ValueError("Latitude values must be in [-90, 90].")

        return CoordinateData(coords=np.asarray(coords, dtype=float), latlong=True)

    if (
        isinstance(coords_euclidean, str)
        or coords_euclidean is None
        or len(coords_euclidean) < 1
    ):
        raise ValueError(
            "`coords_euclidean` must be a list or tuple of one or more column names."
        )

    miss = sorted(set(coords_euclidean) - set(data.columns))
    if miss:
        raise ValueError(f"Coordinate variables not found in data: {', '.join(miss)}")

    coords = data.iloc[obs_index][list(coords_euclidean)]
    if not all(np.issubdtype(dtype, np.number) for dtype in coords.dtypes):
        raise ValueError("`coords_euclidean` columns must be numeric.")
    if not np.isfinite(np.asarray(coords, dtype=float)).all():
        raise ValueError("Euclidean coordinates must be finite.")

    return CoordinateData(coords=np.asarray(coords, dtype=float), latlong=False)
