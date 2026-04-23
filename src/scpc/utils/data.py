from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import pandas as pd

from ..types import (
    ArrayLike,
    ConditionalProjectionSetup,
    CoordinateData,
    DataFrameLike,
    MatrixLike,
    ModelLike,
)
from .matrix import make_iv_residualizer


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
    """Build the fixed-effect demeaning callable used by pyfixest.

    This mirrors the `fixest::demean(...)` calls in scpcr. pyfixest keeps the
    fixed-effect structure outside the numeric design matrices, so we rebuild
    the same demeaning step from the stored model metadata when conditional
    scpc needs it.
    """
    from pyfixest.estimation.internals.demean_ import demean_model

    fe = model._fe
    weights = model._weights
    na_index = model._na_index
    fixef_tol = model._fixef_tol
    fixef_maxiter = model._fixef_maxiter
    demean_func = model._demean_func

    def demean(values: MatrixLike, *, context: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        vec_input = values.ndim == 1
        if vec_input:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"{context} must be a vector or matrix.")
        if values.shape[0] != len(fe):
            raise ValueError(
                f"{context} does not line up with the stored fixed effects."
            )
        if values.shape[1] == 0:
            return values[:, 0] if vec_input else values

        col_names = [f"v{i}" for i in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=col_names, index=fe.index)
        y = frame.iloc[:, :1]
        x = frame.iloc[:, 1:]
        # pyfixest caches demeaned values by column name. we pass a fresh
        # cache here because scpc demeaning is for temporary arrays whose
        # names do not identify their contents across calls.
        y_demeaned, x_demeaned = demean_model(
            y,
            x,
            fe,
            weights,
            {},
            na_index,
            fixef_tol,
            fixef_maxiter,
            demean_func,
        )

        out = pd.concat([y_demeaned, x_demeaned], axis=1).loc[:, col_names]
        out_values = np.asarray(out, dtype=float)
        if out_values.shape != values.shape:
            raise ValueError(f"{context} changed shape during fixed-effect demeaning.")
        return out_values[:, 0] if vec_input else out_values

    return demean


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

    row_labels = np.asarray(model.model.data.row_labels)

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
) -> dict[str, MatrixLike | list[str] | None | bool]:
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

    has_intercept = "Intercept" in coef_names
    exo_no_intercept = [name for name in exo_names if name != "Intercept"]
    intercept = (
        np.ones((n, 1), dtype=float) if has_intercept else np.empty((n, 0), dtype=float)
    )
    exo = get_pyfixest_named_columns(
        data,
        exo_no_intercept,
        n=n,
        context="PyFixest IV exogenous regressor",
    )
    inst = get_pyfixest_named_columns(
        data,
        inst_names,
        n=n,
        context="PyFixest IV instrument",
    )

    x = np.column_stack((intercept, exo, raw_endog))
    z = np.column_stack((intercept, exo, inst))
    model_mat = np.column_stack((intercept, exo, fit_endog))

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

    Returns:
        The normalized conditional projection setup.

    Raises:
        ValueError: Raised later if the regression objects cannot be aligned.
    """
    setup = ConditionalProjectionSetup(
        model_mat=np.asarray(model_mat, dtype=float),
        include_intercept=True,
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
            design = get_fixest_iv_design(model)
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
                fixef_id=None,
                residualize=make_iv_residualizer(x_iv, z_iv, demean=demean),
                is_iv=True,
            )

        if has_fixest_fe(model):
            return ConditionalProjectionSetup(
                model_mat=model_mat,
                include_intercept=False,
                fixef_id=None,
            )

        return setup

    if not has_fixest_fe(model):
        return setup

    raise ValueError(
        "Conditional setup for fixest-style fixed effects is not implemented."
    )


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
