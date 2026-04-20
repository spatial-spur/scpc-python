from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import (
    ArrayLike,
    ConditionalProjectionSetup,
    CoordinateData,
    DataFrameLike,
    MatrixLike,
    ModelLike,
)


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
        Integer indices locating the model observations in `data`.

    Raises:
        ValueError: Raised later if the model rows cannot be mapped back to
            the provided data.
    """
    row_labels = np.asarray(model.model.data.row_labels)

    try:
        obs_index = row_labels.astype(int)
    except (TypeError, ValueError):
        obs_index = data.index.get_indexer(row_labels) + 1
        if np.any(obs_index == 0):
            raise ValueError("Could not map model frame rows back to `data`.")

    if np.any((obs_index < 1) | (obs_index > len(data))):
        raise ValueError(
            "Model observation indices are outside the row range of `data`."
        )

    return obs_index


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
    return np.asarray(model.model.exog, dtype=float)


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
    fixef_vars = getattr(model, "fixef_vars", None)
    return fixef_vars is not None and len(fixef_vars) > 0


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
    if uncond or not has_fixest_fe(model):
        return setup

    raise ValueError(
        "Conditional setup for fixest-style fixed effects is not implemented."
    )


def resolve_coords_input(
    data: DataFrameLike,
    obs_index: ArrayLike,
    lon: str | None,
    lat: str | None,
    coord_euclidean: Sequence[str] | None,
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
        coord_euclidean: Euclidean coordinate column names.

    Returns:
        The normalized coordinate representation used by the spatial engine.

    Raises:
        ValueError: Raised later for invalid or inconsistent coordinate input.
    """
    use_geodesic = lon is not None or lat is not None
    use_euclidean = coord_euclidean is not None

    if use_geodesic and use_euclidean:
        raise ValueError("Specify either `lon`/`lat` or `coord_euclidean`, not both.")
    if not use_geodesic and not use_euclidean:
        raise ValueError("Specify coordinates via `lon`/`lat` or `coord_euclidean`.")

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
        isinstance(coord_euclidean, str)
        or coord_euclidean is None
        or len(coord_euclidean) < 1
    ):
        raise ValueError(
            "`coord_euclidean` must be a character vector with at least one column name."
        )

    miss = sorted(set(coord_euclidean) - set(data.columns))
    if miss:
        raise ValueError(f"Coordinate variables not found in data: {', '.join(miss)}")

    coords = data.iloc[obs_index][list(coord_euclidean)]
    if not all(np.issubdtype(dtype, np.number) for dtype in coords.dtypes):
        raise ValueError("`coord_euclidean` columns must be numeric.")
    if not np.isfinite(np.asarray(coords, dtype=float)).all():
        raise ValueError("Euclidean coordinates must be finite.")

    return CoordinateData(coords=np.asarray(coords, dtype=float), latlong=False)
