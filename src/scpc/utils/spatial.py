from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..types import ArrayLike, MatrixLike, SpatialSetup
from .matrix import get_w, lvech

GQX, GQW = np.polynomial.legendre.leggauss(40)
GQX = GQX * 0.5 + 0.5
GQW = GQW * 0.5


def get_avc(c: float, dist: ArrayLike) -> float:
    """Evaluate average pairwise correlation under an exponential kernel.

    This helper turns a kernel scale and a collection of pairwise distances
    into the average correlation summary that SCPC uses as its user-facing
    spatial sensitivity parameter.

    Args:
        c: Exponential kernel scale.
        dist: Pairwise distances.

    Returns:
        The implied average pairwise correlation.
    """
    dist = np.asarray(dist, dtype=float)
    return float(np.mean(np.exp(-c * dist)))


def get_distmat(s: MatrixLike, latlong: bool) -> MatrixLike:
    """Compute the pairwise distance matrix from observation coordinates.

    This helper is the boundary between raw location data and the rest of the
    spatial engine. It turns normalized coordinates into the distance matrix
    from which kernels, projections, and omega matrices are built.

    Args:
        s: Coordinate matrix aligned to the active observations.
        latlong: Whether to treat coordinates as longitude/latitude.

    Returns:
        The pairwise distance matrix.

    Raises:
        ValueError: Raised later for invalid coordinate shapes.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    if latlong:
        if s.shape[1] != 2:
            raise ValueError(
                "Internal error: geodesic coordinates must have exactly two columns."
            )

        # longitude and latitude are expected in degrees, matching the R code
        lon = np.radians(s[:, 0])
        lat = np.radians(s[:, 1])

        dlon = lon[:, None] - lon[None, :]
        dlat = lat[:, None] - lat[None, :]

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
        )
        d = 2 * 6378137 * np.arcsin(np.sqrt(a))
        d = d / (2 * math.pi * 6378137)
    else:
        d = np.sqrt(np.sum((s[:, None, :] - s[None, :, :]) ** 2, axis=2))

    return d


def get_c0_from_avc(dist: ArrayLike, avc0: float) -> float:
    """Solve for the kernel scale implied by a target average correlation.

    Users think in terms of an average correlation bound, but the spatial
    kernel itself is parameterized by a scale value. This helper bridges that
    gap by turning the bound into the kernel scale used internally.

    Args:
        dist: Pairwise distances.
        avc0: Target average pairwise correlation.

    Returns:
        The kernel scale matching the requested average correlation.
    """
    c0 = 10.0
    c1 = 10.0

    while get_avc(c0, dist) < avc0:
        c1 = c0
        c0 = 0.5 * c0

    while get_avc(c1, dist) > avc0 and c1 < 5000:
        c0 = c1
        c1 = 2 * c1

    while True:
        # search on the same multiplicative scale as the R implementation
        c = math.sqrt(c0 * c1)
        if get_avc(c, dist) > avc0:
            c0 = c
        else:
            c1 = c
        if c1 - c0 < 0.001:
            break

    return c


def get_rp(om: MatrixLike, cv: float) -> float:
    """Evaluate the rejection probability for one omega matrix.

    This is the low-level size-control calculation inside SCPC. It tells the
    package how likely a given covariance case is to reject at a proposed
    normalized critical value.

    Args:
        om: Omega matrix for one spatial covariance case.
        cv: Normalized critical value.

    Returns:
        The rejection probability for that case.
    """
    om = np.asarray(om, dtype=float)

    # replace the first row after scaling, just as in the R implementation
    omx = -(cv**2) * om
    omx[0, :] = om[0, :]

    evals_raw = np.real(np.linalg.eigvals(omx))
    evals = -evals_raw[evals_raw < 0]
    if len(evals) == 0:
        return 0.0

    denom = np.max(evals_raw)
    if not np.isfinite(denom) or denom <= 0:
        return 0.0

    evals = evals / denom
    tot = 0.0
    for j in range(len(GQX)):
        u = GQX[j]
        arg = (1 - u**2) * math.exp(np.sum(np.log1p(evals / (1 - u**2))))
        if not np.isfinite(arg) or arg <= 0:
            continue
        tot = tot + GQW[j] / math.sqrt(arg)

    return float(tot * 2 / math.pi)


def max_rp(oms: list[MatrixLike], q: int, cv: float) -> tuple[float, int]:
    """Find the most conservative rejection probability on the omega grid.

    SCPC controls size over a whole family of spatial covariance cases, not
    just one. This helper identifies which omega matrix is currently worst and
    reports its rejection probability.

    Args:
        oms: Omega matrices over the spatial correlation grid.
        q: Number of non-constant spatial principal components retained.
        cv: Normalized critical value.

    Returns:
        A pair containing the largest rejection probability and its index.
    """
    rps = np.array([get_rp(om[: q + 1, : q + 1], cv) for om in oms], dtype=float)
    i = int(np.argmax(rps))
    return float(np.max(rps)), i


def get_cv(oms: list[MatrixLike], q: int, level: float) -> float:
    """Find the size-controlling critical value for a projection dimension.

    This helper searches for the smallest critical value that keeps the SCPC
    rejection probability below the requested level across the omega grid.

    Args:
        oms: Omega matrices over the spatial correlation grid.
        q: Number of non-constant spatial principal components retained.
        level: Target two-sided rejection probability.

    Returns:
        The size-controlling critical value.
    """
    rp = 1.0
    i = 0
    cv0 = stats.t.ppf(1 - level / 2, df=q) / math.sqrt(q)

    while rp > level:
        cv1 = cv0
        while True:
            if get_rp(oms[i][: q + 1, : q + 1], cv1) > level:
                cv0 = cv1
                cv1 = cv1 + 1 / math.sqrt(q)
            else:
                break

        while cv1 - cv0 > 0.001 / math.sqrt(q):
            cv = 0.5 * (cv0 + cv1)
            if get_rp(oms[i][: q + 1, : q + 1], cv) > level:
                cv0 = cv
            else:
                cv1 = cv

        max_rp_value, max_rp_i = max_rp(oms, q, cv1)
        if max_rp_i == i:
            break

        i = max_rp_i
        rp = max_rp_value
        cv0 = cv1

    return cv1 * math.sqrt(q)


def set_final_w(
    oms: list[MatrixLike],
    w: MatrixLike,
    qmax: int,
) -> tuple[MatrixLike, float, int]:
    """Choose how many spatial principal components to keep.

    SCPC does not use an arbitrary projection size. It compares candidate
    projection dimensions and keeps the one that gives the shortest expected
    confidence interval under the package's reference criterion. This helper
    performs that selection and returns the projection that the main inference
    routine will actually use.

    Args:
        oms: Omega matrices over the spatial correlation grid.
        w: Candidate spatial projection basis.
        qmax: Largest projection dimension under consideration.

    Returns:
        The selected projection matrix, its matching critical value, and the
        selected number of spatial principal components.

    Raises:
        ValueError: Raised later if the inputs are dimensionally incompatible.
    """
    w = np.asarray(w, dtype=float)
    cvs = np.empty(qmax, dtype=float)
    lengths = np.empty(qmax, dtype=float)

    for q in range(1, qmax + 1):
        cvs[q - 1] = get_cv(oms, q, 0.05)
        lengths[q - 1] = (
            cvs[q - 1]
            * math.gamma(0.5 * (q + 1))
            / (math.sqrt(q) * math.gamma(0.5 * q))
        )

    q_opt = int(np.argmin(lengths)) + 1
    return w[:, : q_opt + 1], float(cvs[q_opt - 1]), q_opt


def get_nc(c0: float, cmax: float, cgridfac: float) -> int:
    """Determine the size of the multiplicative kernel grid.

    SCPC evaluates a sequence of spatial correlation strengths between a base
    kernel scale and a more weakly correlated upper end. This helper decides
    how many points are needed in that grid.

    Args:
        c0: Smallest kernel scale in the grid.
        cmax: Largest kernel scale in the grid.
        cgridfac: Multiplicative step size between grid points.

    Returns:
        The number of grid points to evaluate.
    """
    return max(2, math.ceil(math.log(cmax / c0) / math.log(cgridfac)))


def get_oms(
    distmat: MatrixLike,
    c0: float,
    cmax: float,
    w: MatrixLike,
    cgridfac: float,
) -> list[MatrixLike]:
    """Build omega matrices over the spatial correlation grid.

    These omega matrices summarize how each candidate spatial covariance
    structure interacts with the current projection basis. They are the core
    objects used for size control and critical value calculations in SCPC.

    Args:
        distmat: Pairwise distance matrix.
        c0: Smallest kernel scale in the grid.
        cmax: Largest kernel scale in the grid.
        w: Spatial projection matrix.
        cgridfac: Multiplicative step size between grid points.

    Returns:
        Omega matrices across the spatial correlation grid.
    """
    distmat = np.asarray(distmat, dtype=float)
    w = np.asarray(w, dtype=float)

    nc = get_nc(c0, cmax, cgridfac)
    oms: list[MatrixLike] = [np.empty((0, 0)) for _ in range(nc)]
    c = c0
    for i in range(nc):
        oms[i] = w.T @ (np.exp(-c * distmat) @ w)
        c = c * cgridfac

    return oms


def set_oms_wfin(distmat: MatrixLike, avc0: float) -> SpatialSetup:
    """Build the unconditional spatial setup from distances and an AVC bound.

    This helper is the spatial engine in one place. It converts the user-level
    average correlation bound into kernel scales, candidate projections, omega
    matrices, and finally the projection basis that `scpc()` uses for
    inference.

    Args:
        distmat: Pairwise distance matrix.
        avc0: Target average pairwise correlation bound.

    Returns:
        The complete spatial setup used by the main inference routine.
    """
    distmat = np.asarray(distmat, dtype=float)
    n = distmat.shape[0]
    distv = lvech(distmat)

    cgridfac = 1.2
    minavc = 0.00001

    if avc0 >= 0.05:
        qmax = 10
    elif avc0 >= 0.01:
        qmax = 20
    elif avc0 >= 0.005:
        qmax = 60
    else:
        qmax = 120

    c0 = get_c0_from_avc(distv, avc0)
    cmax = get_c0_from_avc(distv, minavc)

    while True:
        # grow the candidate projection dimension until the selected q stops
        # landing on the upper boundary.
        qmax = min(qmax, n - 1)
        w = get_w(distmat, c0, qmax)
        oms = get_oms(distmat, c0, cmax, w, cgridfac)
        wfin, cvfin, q = set_final_w(oms, w, qmax)
        if q < qmax or qmax == n - 1:
            break
        qmax = round(qmax + qmax / 2)

    return SpatialSetup(wfin=wfin, cvfin=cvfin, omsfin=oms, c0=c0, cmax=cmax)
