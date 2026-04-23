from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypedDict

import numpy as np
import pandas as pd

from .utils.results import resolve_parm_indices

ArrayLike: TypeAlias = Any
MatrixLike: TypeAlias = Any
ModelLike: TypeAlias = Any
DataFrameLike: TypeAlias = Any

SCPC_STATS_COLUMNS = ["Coef", "Std_Err", "t", "P>|t|", "2.5 %", "97.5 %"]
SCPC_CV_COLUMNS = ["32%", "10%", "5%", "1%"]
SCPC_CV_LEVELS = {0.68: 0, 0.90: 1, 0.95: 2, 0.99: 3}


class FixestSpec(TypedDict):
    """Stored pyfixest IV design objects aligned to coefficient order."""

    X: MatrixLike
    Z: MatrixLike
    model_mat: MatrixLike
    coef_names: list[str]
    fixef_id: ArrayLike | None
    has_fixef: bool


@dataclass(slots=True)
class CoordinateData:
    """Coordinates aligned to the active observations."""

    coords: MatrixLike
    """Coordinate matrix for the rows used in the model."""
    latlong: bool
    """Whether the coordinates are longitude/latitude rather than Euclidean."""


@dataclass(slots=True)
class ConditionalProjectionSetup:
    """Regression-side inputs used for the conditional adjustment."""

    model_mat: MatrixLike
    """Design matrix aligned to the rows kept in the model."""
    include_intercept: bool
    """Whether the adjustment should include an intercept term."""
    fixef_id: ArrayLike | None
    """Fixed-effect identifiers used for demeaning, if any."""
    residualize: Any | None = None
    """Residualizer used by the conditional IV branch, if any."""
    is_iv: bool = False
    """Whether the conditional adjustment should use the IV branch."""


@dataclass(slots=True)
class SpatialSetup:
    """Spatial objects used by the inference step."""

    wfin: MatrixLike
    """Final spatial projection matrix used for inference."""
    cvfin: float
    """Critical value attached to the final projection."""
    omsfin: list[MatrixLike]
    """Omega matrices evaluated over the spatial correlation grid."""
    c0: float
    """Kernel scale implied by the chosen correlation bound."""
    cmax: float
    """Largest kernel scale used in the spatial grid search."""
    coords: MatrixLike
    """Coordinates actually used by the spatial setup."""
    perm: ArrayLike
    """Permutation used to align the spatial branch with the data."""
    distmat: MatrixLike | None
    """Exact distance matrix, or `None` in the approximate branch."""
    method: str
    """Spatial method actually used: `exact` or `approx`."""
    large_n: bool
    """Whether the approximate large-n branch was used."""
    random_state: int | None
    """Random state carried forward by the approximate branch."""


@dataclass(slots=True)
class SCPCResult:
    """Main result returned by `scpc()`."""

    scpcstats: MatrixLike
    """Main output table with estimates, standard errors, tests, and intervals."""
    scpccvs: MatrixLike | None
    """Stored critical values, when requested."""
    w: MatrixLike
    """Final spatial projection matrix used for inference."""
    avc: float
    """Average correlation bound supplied by the user."""
    c0: float
    """Kernel scale implied by `avc`."""
    cv: float
    """Default 5 percent critical value used for intervals."""
    q: int
    """Number of spatial components kept in the final projection."""
    coef_names: list[str]
    """Coefficient names aligned to rows of `scpcstats` and `scpccvs`."""
    method: str = "exact"  # this is the actually used setting, so "auto" is missing
    """Spatial method actually used: `exact` or `approx`."""
    large_n_seed: int = 1
    """Seed used by the large-n approximation branch."""

    def __repr__(self) -> str:
        """Return a developer-oriented representation of the result.

        This method provides a compact object representation for interactive
        work and debugging, where a short structural summary is more useful
        than the full printed results table.

        Returns:
            A representation string.
        """
        return (
            f"SCPCResult(ncoef={len(self.coef_names)}, q={self.q}, "
            f"avc={self.avc!r}, method={self.method!r})"
        )

    def __str__(self) -> str:
        """Return a user-facing summary string.

        This method fills the role of `print.scpc`. It is the quick, readable
        view of the SCPC result that a user sees when they inspect the object
        at the console.

        Returns:
            A formatted summary string.
        """
        stats = pd.DataFrame(
            np.asarray(self.scpcstats, dtype=float),
            index=self.coef_names,
            columns=SCPC_STATS_COLUMNS,
        )
        lines = [
            f"SCPC Inference (ncoef = {len(self.coef_names)}, q = {self.q})",
            "",
            stats.iloc[:, :4].to_string(),
        ]
        if self.scpccvs is not None:
            cvs = pd.DataFrame(
                np.asarray(self.scpccvs, dtype=float),
                index=self.coef_names,
                columns=SCPC_CV_COLUMNS,
            )
            lines.extend(["", "Two-sided critical values:", cvs.to_string()])
        return "\n".join(lines)

    def summary(self) -> str:
        """Return an extended formatted summary.

        This method fills the role of `summary.scpc`. It exists for the case
        where the user wants the fuller inference table, confidence intervals,
        and any stored critical values in one readable summary.

        Returns:
            A formatted summary string.
        """
        stats = pd.DataFrame(
            np.asarray(self.scpcstats, dtype=float),
            index=self.coef_names,
            columns=SCPC_STATS_COLUMNS,
        )
        lines = [
            (
                f"SCPC Inference (ncoef = {len(self.coef_names)}, "
                f"q = {self.q}, avc = {self.avc})"
            ),
            "",
            stats.iloc[:, :4].to_string(),
            "",
            "95% Confidence Intervals:",
            self.confint().to_string(),
        ]
        if self.scpccvs is not None:
            cvs = pd.DataFrame(
                np.asarray(self.scpccvs, dtype=float),
                index=self.coef_names,
                columns=SCPC_CV_COLUMNS,
            )
            lines.extend(["", "Two-sided critical values:", cvs.to_string()])
        return "\n".join(lines)

    def coef(self) -> Any:
        """Return the coefficient estimates.

        This method fills the role of `coef.scpc`. It gives users the compact
        coefficient vector without making them manually pull it out of the full
        result table.

        Returns:
            The coefficient estimates.
        """
        stats = np.asarray(self.scpcstats, dtype=float)
        return pd.Series(stats[:, 0], index=self.coef_names, name="Coef")

    def confint(
        self,
        parm: str | int | Sequence[str] | Sequence[int] | None = None,
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
        idx = resolve_parm_indices(self.coef_names, parm)
        stats = np.asarray(self.scpcstats, dtype=float)
        names = [self.coef_names[i] for i in idx]

        if level == 0.95:
            values = stats[idx, 4:6]
        else:
            if self.scpccvs is None:
                raise ValueError(f"Confidence level {level} is not available.")
            level_idx = SCPC_CV_LEVELS[level]
            cvs = np.asarray(self.scpccvs, dtype=float)
            cv_vals = cvs[idx, level_idx]
            coef_vals = stats[idx, 0]
            se_vals = stats[idx, 1]
            values = np.column_stack(
                (coef_vals - cv_vals * se_vals, coef_vals + cv_vals * se_vals)
            )

        lower = 100 * (1 - level) / 2
        upper = 100 * (1 + level) / 2
        return pd.DataFrame(
            values,
            index=names,
            columns=[f"{lower:g} %", f"{upper:g} %"],
        )
