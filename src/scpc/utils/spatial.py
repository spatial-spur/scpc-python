from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.sparse.linalg import eigsh

from ..types import ArrayLike, MatrixLike, SpatialSetup
from .matrix import demeanmat, get_w, lvech

LARGE_N_THRESHOLD = 4500
LARGE_N_CAPN = 20
LARGE_N_CAPM = 1_000_000
LARGE_N_M = 1000
CGRIDFAC = 1.2
MINAVC = 0.00001
UINT32_MOD = 2**32

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


def get_distvec(s1: MatrixLike, s2: MatrixLike, latlong: bool) -> ArrayLike:
    """Compute paired distances between two coordinate matrices."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)

    if s1.shape != s2.shape:
        raise ValueError(
            "Internal error: paired distance inputs must have matching dimensions."
        )

    if latlong:
        lon1 = np.radians(s1[:, 0])
        lat1 = np.radians(s1[:, 1])
        lon2 = np.radians(s2[:, 0])
        lat2 = np.radians(s2[:, 1])
        dlon = 0.5 * (lon1 - lon2)
        dlat = 0.5 * (lat1 - lat2)
        return (
            np.arcsin(
                np.sqrt(
                    np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
                )
            )
            / math.pi
        )

    return np.sqrt(np.sum((s1 - s2) ** 2, axis=1))


def normalize_s(s: MatrixLike, latlong: bool) -> tuple[MatrixLike, ArrayLike]:
    """Normalize coordinates for the large-n approximation branch."""
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    if not latlong:
        s = s - np.mean(s, axis=0, keepdims=True)
        evals, rot = np.linalg.eigh(s.T @ s)
        rot = rot[:, np.argsort(evals)[::-1]]
        s = s @ rot

        # this keeps the randomized branch in a fixed orientation.
        if np.max(s[:, 0]) != np.max(np.abs(s[:, 0])):
            s = -s

        perm = np.lexsort(tuple(s[:, col] for col in range(s.shape[1] - 1, -1, -1)))
        s = s[perm, :]
        s = s - np.min(s, axis=0, keepdims=True)
        smax = np.max(s)
        if np.isfinite(smax) and smax > 0:
            s = s / smax
        return s, perm.astype(int)

    s = s.copy()
    s[:, 0] = s[:, 0] - np.mean(s[:, 0])
    s[:, 0] = ((s[:, 0] + 180.0) % 360.0) - 180.0
    perm = np.lexsort((s[:, 0], s[:, 1]))
    s = s[perm, :]
    return s, perm.astype(int)


def next_u(random_t: int) -> tuple[float, int]:
    """Advance the large-n pseudo-random state by one step."""
    random_t = (64389 * int(random_t) + 1) % UINT32_MOD
    return random_t / UINT32_MOD, random_t


def jumble_s(s: MatrixLike, m: int, random_t: int) -> tuple[MatrixLike, int]:
    """Randomly jumble the first `m` rows of the coordinate matrix."""
    s = np.array(s, dtype=float, copy=True)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    n = s.shape[0]
    for i in range(m):
        _, random_t = next_u(random_t)
        j = int((random_t * n) // UINT32_MOD)
        tmp = s[j, :].copy()
        s[j, :] = s[i, :]
        s[i, :] = tmp
    return s, random_t


def ln_subset_evecs(distmat: MatrixLike, c0: float, qmax: int) -> MatrixLike:
    """Compute subset eigenvectors for the large-n approximation branch."""
    distmat = np.asarray(distmat, dtype=float)
    sig_d = demeanmat(np.exp(-c0 * distmat))
    n = sig_d.shape[0]

    if qmax < n - 1:
        evals, vectors = eigsh(sig_d, k=qmax, which="LM")
        order = np.argsort(np.abs(evals))[::-1]
        return vectors[:, order]

    evals, vectors = np.linalg.eigh(sig_d)
    order = np.argsort(evals)[::-1]
    return vectors[:, order[:qmax]]


def lnset_wc0(
    s: MatrixLike,
    avc0: float,
    qmax: int,
    minavc: float,
    latlong: bool,
    capN: int = 20,
    m: int = 1000,
    random_t: int = 1,
) -> tuple[MatrixLike, float, float, int]:
    """Approximate the projection basis and kernel scales for large n."""
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    n = s.shape[0]
    m = min(m, n)
    block_len = m * (m - 1) // 2
    ms: list[tuple[np.ndarray, np.ndarray]] = []
    distvec = np.empty(capN * block_len, dtype=float)

    r = s.copy()
    for i in range(capN):
        r, random_t = jumble_s(r, m, random_t)
        subset = r[:m, :]
        distmat = get_distmat(subset, latlong)
        ms.append((subset, distmat))
        start = i * block_len
        stop = (i + 1) * block_len
        distvec[start:stop] = lvech(distmat)

    c0 = get_c0_from_avc(distvec, avc0)
    cmax = get_c0_from_avc(distvec, minavc)

    wall = np.zeros((n, capN * qmax), dtype=float)
    for i, (subset, distmat) in enumerate(ms):
        w0 = ln_subset_evecs(distmat, c0, qmax)
        wx = np.zeros((n, qmax), dtype=float)
        for j in range(m):
            # this is the same nystrom-style extension used in the r/stata branch.
            v = np.exp(-c0 * np.sqrt(np.sum((s - subset[j, :]) ** 2, axis=1)))
            wx = wx + np.outer(v, w0[j, :])
        wx = wx - np.mean(wx, axis=0, keepdims=True)
        norms = np.sqrt(np.sum(wx**2, axis=0))
        norms[~np.isfinite(norms) | (norms == 0)] = 1.0
        wx = wx / norms
        for j in range(qmax):
            wall[:, j * capN + i] = wx[:, j]

    w = np.zeros((n, qmax), dtype=float)
    for i in range(qmax):
        wx = wall[:, : capN * (i + 1)]
        evals, evecs = np.linalg.eigh(wx.T @ wx)
        evec = evecs[:, np.argmax(evals)]
        w[:, i] = wx @ evec
        w[:, i] = w[:, i] / math.sqrt(np.sum(w[:, i] ** 2))
        wall = wall - np.outer(w[:, i], w[:, i] @ wall)

    w = np.column_stack((np.full(n, 1 / math.sqrt(n)), w))
    return w, c0, cmax, random_t


def raninds(n: int, capM: int, random_t: int) -> tuple[ArrayLike, int]:
    """Generate distinct linked random indices for sampled distance pairs."""
    v = np.empty(capM + 1, dtype=int)
    _, random_t = next_u(random_t)
    j = int((random_t * n) // UINT32_MOD)

    for i in range(capM + 1):
        v[i] = j
        _, random_t = next_u(random_t)
        j = (j + 1 + int((random_t * (n - 1)) // UINT32_MOD)) % n

    return v, random_t


def lnget_oms(
    s: MatrixLike,
    c0: float,
    cmax: float,
    w: MatrixLike,
    cgridfac: float,
    capM: int = 1000000,
    random_t: int = 1,
    latlong: bool = False,
) -> tuple[list[MatrixLike], int]:
    """Approximate omega matrices for the large-n branch."""
    s = np.asarray(s, dtype=float)
    w = np.asarray(w, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    nc = get_nc(c0, cmax, cgridfac)
    oms: list[MatrixLike] = [np.empty((0, 0)) for _ in range(nc)]

    n = s.shape[0]
    inds, random_t = raninds(n, capM, random_t)
    dist = get_distvec(s[inds[:capM], :], s[inds[1 : capM + 1], :], latlong)
    w1 = w[inds[:capM], :]
    w2 = w[inds[1 : capM + 1], :]

    c = c0
    for i in range(nc):
        cd = np.exp(-c * dist)
        oms[i] = np.eye(w.shape[1]) + 0.5 * (n * (n - 1) / capM) * (
            w1.T @ (w2 * cd[:, None]) + w2.T @ (w1 * cd[:, None])
        )
        c = c * cgridfac

    return oms, random_t


def validate_large_n_seed(seed: object) -> int:
    """Validate the seed used by the large-n approximation branch."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
        or int(seed) >= UINT32_MOD
    ):
        raise ValueError(
            "`large_n_seed` must be a single integer-valued number in [0, 2^32)."
        )
    return int(seed)


def validate_scpc_method(method: object) -> str:
    """Validate the requested SCPC spatial method."""
    methods = {"auto", "exact", "approx"}
    if not isinstance(method, str) or not method or method not in methods:
        raise ValueError('`method` must be one of "auto", "exact", or "approx".')
    return method


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


def set_oms_wfin(
    coords: MatrixLike,
    avc0: float,
    latlong: bool,
    method: str = "auto",
    large_n_seed: int = 1,
) -> SpatialSetup:
    """Build the unconditional spatial setup from coordinates and an AVC bound.

    This helper now chooses between the exact and approximate spatial setup
    branches and returns the metadata the later conditional step needs.

    Args:
        coords: Observation coordinates.
        avc0: Target average pairwise correlation bound.
        latlong: Whether to treat coordinates as longitude/latitude.
        method: Requested spatial method.
        large_n_seed: Seed for the large-n approximation branch.

    Returns:
        The complete spatial setup used by the main inference routine.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    if method == "auto":
        method_actual = "exact" if n < LARGE_N_THRESHOLD else "approx"
    else:
        method_actual = method

    if avc0 >= 0.05:
        qmax = 10
    elif avc0 >= 0.01:
        qmax = 20
    elif avc0 >= 0.005:
        qmax = 60
    else:
        qmax = 120

    distmat: MatrixLike | None = None
    coords_use = coords
    perm = np.arange(n, dtype=int)
    random_t: int | None = None

    while True:
        qmax = min(qmax, n - 1)

        if method_actual == "exact":
            distmat = get_distmat(coords, latlong)
            distv = lvech(distmat)
            c0 = get_c0_from_avc(distv, avc0)
            cmax = get_c0_from_avc(distv, MINAVC)
            w = get_w(distmat, c0, qmax)
            oms = get_oms(distmat, c0, cmax, w, CGRIDFAC)
            coords_use = coords
            perm = np.arange(n, dtype=int)
            random_t = None
        else:
            random_t = large_n_seed
            coords_use, perm = normalize_s(coords, latlong)
            w, c0, cmax, random_t = lnset_wc0(
                coords_use,
                avc0,
                qmax,
                MINAVC,
                latlong,
                capN=LARGE_N_CAPN,
                m=LARGE_N_M,
                random_t=random_t,
            )
            oms, random_t = lnget_oms(
                coords_use,
                c0,
                cmax,
                w,
                CGRIDFAC,
                capM=LARGE_N_CAPM,
                random_t=random_t,
                latlong=latlong,
            )
            distmat = None

        wfin, cvfin, q = set_final_w(oms, w, qmax)
        if q < qmax or qmax == n - 1:
            break
        qmax = round(qmax + qmax / 2)

    return SpatialSetup(
        wfin=wfin,
        cvfin=cvfin,
        omsfin=oms,
        c0=c0,
        cmax=cmax,
        coords=coords_use,
        perm=perm,
        distmat=distmat,
        method=method_actual,
        large_n=method_actual == "approx",
        random_state=random_t,
    )
