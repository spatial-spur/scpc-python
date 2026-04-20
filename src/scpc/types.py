from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

ArrayLike: TypeAlias = Any
MatrixLike: TypeAlias = Any
ModelLike: TypeAlias = Any
DataFrameLike: TypeAlias = Any


@dataclass(slots=True)
class CoordinateData:
    """Normalized coordinates for spatial distance calculations.

    This record holds the observation-aligned coordinate matrix together with
    the information needed to interpret those coordinates correctly. In this
    package, locations can come in either as longitude/latitude pairs or as
    ordinary Euclidean coordinates. `CoordinateData` gives the rest of the
    code one consistent representation of that information.

    Attributes:
        coords: Numeric coordinate matrix aligned to the model observations.
        latlong: Whether the coordinates should be treated as longitude and
            latitude rather than Euclidean coordinates.
    """

    coords: MatrixLike
    latlong: bool


@dataclass(slots=True)
class ConditionalProjectionSetup:
    """Regressor information for the conditional SCPC adjustment.

    This record contains the regression-side objects needed to build the
    conditional projection basis. In higher-level terms, it represents the
    covariate space that spatial directions must be made orthogonal to before
    conditional SCPC inference can be computed.

    Attributes:
        model_mat: Regression design matrix aligned to the active observations.
        include_intercept: Whether the conditional projection should include an
            explicit intercept column.
        fixef_id: Optional fixed-effect identifiers used for demeaning.
    """

    model_mat: MatrixLike
    include_intercept: bool
    fixef_id: ArrayLike | None


@dataclass(slots=True)
class SpatialSetup:
    """Spatial reference objects derived from the distance matrix.

    This record gathers the main outputs of the spatial setup stage: the final
    projection basis, the critical value attached to that basis, the omega
    grid used for size control, and the kernel scales behind those objects.
    It exists so the main inference routine can pass around one coherent
    description of the spatial environment.

    Attributes:
        wfin: Final spatial projection matrix.
        cvfin: Critical value attached to the final projection.
        omsfin: Omega matrices over the spatial correlation grid.
        c0: Kernel scale matching the target average correlation bound.
        cmax: Largest kernel scale used in the spatial grid search.
    """

    wfin: MatrixLike
    cvfin: float
    omsfin: list[MatrixLike]
    c0: float
    cmax: float


@dataclass(slots=True)
class SCPCResult:
    """SCPC estimates, intervals, and projection metadata.

    This is the main user-facing result type returned by `scpc()`. It keeps
    together the estimated coefficient table, any stored critical values, and
    the spatial projection information that explains how the inference was
    constructed.

    Attributes:
        scpcstats: Main result table with estimates, standard errors, test
            statistics, p-values, and confidence interval endpoints.
        scpccvs: Optional table of stored critical values at supported levels.
        w: Final spatial projection matrix used for inference.
        avc: Average pairwise correlation bound supplied by the user.
        c0: Kernel scale implied by `avc`.
        cv: Unconditional 5 percent critical value.
        q: Number of non-constant spatial principal components retained.
        call: Optional textual representation of the original call.
    """

    scpcstats: MatrixLike
    scpccvs: MatrixLike | None
    w: MatrixLike
    avc: float
    c0: float
    cv: float
    q: int
    call: str | None = None

    def __repr__(self) -> str:
        """Return a developer-oriented representation of the result.

        This method provides a compact object representation for interactive
        work and debugging, where a short structural summary is more useful
        than the full printed results table.

        Returns:
            A representation string.
        """
        pass

    def __str__(self) -> str:
        """Return a user-facing summary string.

        This method fills the role of `print.scpc`. It is the quick, readable
        view of the SCPC result that a user sees when they inspect the object
        at the console.

        Returns:
            A formatted summary string.
        """
        pass

    def summary(self) -> str:
        """Return an extended formatted summary.

        This method fills the role of `summary.scpc`. It exists for the case
        where the user wants the fuller inference table, confidence intervals,
        and any stored critical values in one readable summary.

        Returns:
            A formatted summary string.
        """
        pass

    def coef(self) -> Any:
        """Return the coefficient estimates.

        This method fills the role of `coef.scpc`. It gives users the compact
        coefficient vector without making them manually pull it out of the full
        result table.

        Returns:
            The coefficient estimates.
        """
        pass

    def confint(
        self,
        parm: Sequence[str] | Sequence[int] | None = None,
        level: float = 0.95,
    ) -> Any:
        """Return confidence intervals for selected coefficients.

        This method fills the role of `confint.scpc`. It is the focused access
        path for interval extraction when a user wants selected coefficients or
        a supported confidence level without reading the full summary output.

        Args:
            parm: Coefficient names or positions to include.
            level: Confidence level to request.

        Returns:
            Confidence intervals for the selected coefficients.

        Raises:
            ValueError: Raised later for unknown coefficients or unsupported
                confidence levels.
        """
        pass
