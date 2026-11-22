from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy import optimize, stats
from scipy.sparse.linalg import eigsh

from ..types import ArrayLike, MatrixLike, SpatialSetup
from .matrix import demeanmat, get_w_with_ties, lvech

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


class InvalidSampledCovarianceError(ValueError):
    """An approximate covariance grid cannot define SCPC inference."""


def validate_sampled_covariance(omega: np.ndarray, grid_index: int) -> None:
    """Reject materially indefinite sampled matrices, including zero-diagonal cases."""
    message = (
        f"Invalid sampled SCPC covariance at grid point {grid_index + 1}: "
        "expected a finite positive-semidefinite matrix. Try method='exact'."
    )
    if (
        omega.ndim != 2
        or not omega.size
        or omega.shape[0] != omega.shape[1]
        or not np.isfinite(omega).all()
    ):
        raise InvalidSampledCovarianceError(message)
    scale = float(np.max(np.abs(omega)))
    if scale == 0.0:
        return
    tolerance = 32 * omega.shape[0] * np.finfo(float).eps
    scaled = omega / scale
    if (
        np.max(np.abs(scaled - scaled.T)) > tolerance
        or np.linalg.eigvalsh(scaled)[0] < -tolerance
    ):
        raise InvalidSampledCovarianceError(message)


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

        a = np.clip(
            np.sin(dlat / 2) ** 2
            + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2,
            0.0,
            1.0,
        )
        d = 2 * 6378137 * np.arcsin(np.sqrt(a))
        d = d / (2 * math.pi * 6378137)
    else:
        d = np.hypot.reduce(s[:, None, :] - s[None, :, :], axis=2)

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
        a = np.clip(
            np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2,
            0.0,
            1.0,
        )
        return np.arcsin(np.sqrt(a)) / math.pi

    return np.hypot.reduce(s1 - s2, axis=1)


def normalize_s(s: MatrixLike, latlong: bool) -> tuple[MatrixLike, ArrayLike]:
    """Normalize coordinates for the large-n approximation branch."""
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    if not latlong:
        s = s - np.mean(s, axis=0, keepdims=True)
        max_abs = float(np.max(np.abs(s))) if s.size > 0 else 0.0
        if max_abs > 0.0:
            s_scaled = s / max_abs
            evals, rot = np.linalg.eigh(s_scaled.T @ s_scaled)
            rot = rot[:, np.argsort(evals)[::-1]]
            s = s_scaled @ rot
        else:
            s = np.zeros_like(s)

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
    """Randomly jumble the first `m` rows of the coordinate matrix without replacement."""
    s = np.array(s, dtype=float, copy=True)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    n = s.shape[0]
    for i in range(m):
        _, random_t = next_u(random_t)
        j = i + int((random_t * (n - i)) // UINT32_MOD)
        if j != i:
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

    floor_corr = float(np.mean(distvec == 0.0))
    minavc_eff = floor_corr + (1.0 - floor_corr) * minavc
    c0 = get_c0_from_avc(distvec, avc0)
    cmax = get_c0_from_avc(distvec, minavc_eff)

    wall = np.zeros((n, capN * qmax), dtype=float)
    for i, (subset, distmat) in enumerate(ms):
        w0 = ln_subset_evecs(distmat, c0, qmax)
        wx = np.zeros((n, qmax), dtype=float)
        for j in range(m):
            d_cross = get_distvec(s, np.broadcast_to(subset[j, :], s.shape), latlong)
            v = np.exp(-c0 * d_cross)
            wx = wx + np.outer(v, w0[j, :])
        wx = wx - np.mean(wx, axis=0, keepdims=True)
        norms = np.sqrt(np.sum(wx**2, axis=0))
        norms[~np.isfinite(norms) | (norms == 0)] = 1.0
        wx = wx / norms
        for j in range(qmax):
            wall[:, j * capN + i] = wx[:, j]

    w = np.zeros((n, qmax), dtype=float)
    min_norm = math.sqrt(np.finfo(float).eps)
    for i in range(qmax):
        wx = wall[:, : capN * (i + 1)]
        evals, evecs = np.linalg.eigh(wx.T @ wx)
        evec = evecs[:, np.argmax(evals)]
        w[:, i] = wx @ evec
        norm = math.sqrt(float(np.sum(w[:, i] ** 2)))
        if not math.isfinite(norm) or norm <= min_norm:
            raise ValueError(
                "degenerate principal direction in Nystrom landmark consolidation"
            )
        w[:, i] = w[:, i] / norm
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


def sample_covariance_pairs(
    s: MatrixLike, capM: int, random_t: int, latlong: bool
) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """Draw a linked chain of distinct sites and measure each pair once."""
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    indices, state = raninds(s.shape[0], capM, random_t)
    distances = get_distvec(s[indices[:-1]], s[indices[1:]], latlong)
    return (indices, distances), state


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
    pairs, state = sample_covariance_pairs(s, capM, random_t, latlong)
    return sampled_covariance_grid(w, c0, cmax, cgridfac, pairs), state


def sampled_covariance_grid(
    w: MatrixLike,
    c0: float,
    cmax: float,
    cgridfac: float,
    pairs: tuple[np.ndarray, np.ndarray],
) -> list[MatrixLike]:
    """Evaluate the covariance grid using a fixed set of sampled site pairs."""
    w = np.asarray(w, dtype=float)
    nc = get_nc(c0, cmax, cgridfac)
    oms: list[MatrixLike] = [np.empty((0, 0)) for _ in range(nc)]

    inds, dist = pairs
    n = w.shape[0]
    multiplier = 0.5 * n * (n - 1) / len(dist)
    w1 = w[inds[:-1]]
    w2 = w[inds[1:]]

    gram = w.T @ w
    c = c0
    for i in range(nc):
        cd = np.exp(-c * dist)
        cross = w1.T @ (w2 * cd[:, None])
        omega = gram + multiplier * (cross + cross.T)
        validate_sampled_covariance(omega, i)
        oms[i] = omega
        c = c * cgridfac

    return oms


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

    Uses a scale-invariant initial guess, validates the co-location floor,
    brackets the root without fixed upper caps, and solves to high precision.

    Args:
        dist: Pairwise distances.
        avc0: Target average pairwise correlation.

    Returns:
        The kernel scale matching the requested average correlation.
    """
    dist_arr = np.asarray(dist, dtype=np.float64).ravel()
    if dist_arr.size == 0:
        raise ValueError("At least two locations are required.")
    if not np.isfinite(dist_arr).all() or bool(np.any(dist_arr < 0.0)):
        raise ValueError("Distances must be finite and nonnegative.")

    pos_mask = dist_arr > 0.0
    floor_corr = float(np.mean(dist_arr == 0.0))
    if (
        not bool(np.any(pos_mask))
        or not math.isfinite(avc0)
        or avc0 <= floor_corr
        or avc0 >= 1.0
    ):
        raise ValueError(
            "Target correlation must exceed the co-located-pair fraction and be below one."
        )

    pos_dist = dist_arr[pos_mask]
    mean_pos = float(np.mean(pos_dist))
    max_d = float(np.max(pos_dist))
    target_pos = (avc0 - floor_corr) / (1.0 - floor_corr)
    near_one = avc0 > 0.9999
    finfo = np.finfo(np.float64)
    tiny = float(finfo.tiny)
    huge = float(finfo.max)

    if math.isfinite(mean_pos) and mean_pos > 0.0:
        cand = -math.log(target_pos) / mean_pos
        if math.isfinite(cand) and cand > 0.0:
            raw_init = cand
        else:
            raw_init = min(max(1.0 / max_d, tiny), huge * 0.5)
    else:
        raw_init = min(max(1.0 / max_d, tiny), huge * 0.5)

    def eval_residual(c: float) -> float:
        if near_one:
            return (1.0 - avc0) + float(np.sum(np.expm1(-c * pos_dist))) / dist_arr.size
        return float(np.mean(np.exp(-c * dist_arr))) - avc0

    lo = raw_init
    hi = raw_init
    f_lo = eval_residual(lo)
    if f_lo == 0.0:
        return float(lo)

    if f_lo < 0.0:
        hi = lo
        for _ in range(1024):
            lo *= 0.5
            if lo == 0.0 or not math.isfinite(lo):
                raise ValueError(
                    "Decay calibration lower bracket underflowed; distances may be degenerate."
                )
            f_lo = eval_residual(lo)
            if f_lo == 0.0:
                return float(lo)
            if f_lo > 0.0:
                break
            hi = lo
        if f_lo < 0.0:
            raise ValueError("Decay calibration lower bracket did not enclose root.")
    else:
        f_hi = f_lo
        for _ in range(1024):
            lo = hi
            f_lo = f_hi
            if hi >= huge * 0.5:
                hi = huge
                f_hi = eval_residual(hi)
                break
            hi *= 2.0
            f_hi = eval_residual(hi)
            if f_hi == 0.0:
                return float(hi)
            if f_hi < 0.0:
                break
        if f_hi > 0.0:
            raise ValueError(
                "Spatial decay calibration failed to bracket upper decay scale."
            )

    sol = optimize.root_scalar(
        eval_residual,
        bracket=[lo, hi],
        method="brentq",
        rtol=1e-13,
        xtol=tiny,
        maxiter=200,
    )
    if sol.converged and math.isfinite(sol.root) and sol.root > 0.0:
        return float(sol.root)

    # Safeguarded fallback bisection in log-scale
    for _ in range(128):
        mid = math.sqrt(lo * hi) if lo > 0.0 and hi / lo > 4.0 else 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if eval_residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _log_one_plus_exp(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x + np.log1p(np.exp(-np.abs(x))), np.log1p(np.exp(x)))


def _angular_product(
    cos2: np.ndarray, ratios: np.ndarray, logs: np.ndarray
) -> np.ndarray:
    out = np.zeros_like(cos2)
    pos = cos2 > 0.0
    if not bool(np.any(pos)):
        return out
    c_pos = cos2[pos]
    prod = np.prod(c_pos[:, None] / (c_pos[:, None] + ratios[None, :]), axis=1)
    tiny = float(np.finfo(np.float64).tiny)
    use_sqrt = prod >= tiny
    res = np.empty_like(prod)
    if bool(np.any(use_sqrt)):
        res[use_sqrt] = np.sqrt(prod[use_sqrt])
    if bool(np.any(~use_sqrt)):
        log_cos = np.log(c_pos[~use_sqrt])
        res[~use_sqrt] = np.exp(
            -0.5 * np.sum(_log_one_plus_exp(logs[None, :] - log_cos[:, None]), axis=1)
        )
    out[pos] = res
    return out


def _complement_integrand(
    sin2: np.ndarray, denom: np.ndarray, ratios: np.ndarray, scale: float
) -> np.ndarray:
    product = np.ones_like(sin2)
    deficit = np.zeros_like(sin2)
    for ratio in ratios:
        extra = (float(ratio) / scale) * denom
        inv = 1.0 / (sin2 + extra)
        deficit = deficit + product * extra * inv
        product = product * (sin2 * inv)
    return deficit / ((1.0 + np.sqrt(product)) * denom)


def _quad_panel(fn, lo: float, hi: float) -> float:
    width = hi - lo
    theta = lo + width * GQX
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    vals = fn(sin_t * sin_t, cos_t * cos_t)
    return float(width * np.dot(GQW, vals))


def _quad_adaptive(
    fn, lo: float, hi: float, whole: float, tol: float, depth: int
) -> float:
    mid = 0.5 * (lo + hi)
    left = _quad_panel(fn, lo, mid)
    right = _quad_panel(fn, mid, hi)
    if depth == 20 or abs(left + right - whole) <= max(tol, 1e-11 * abs(whole)):
        return left + right
    return _quad_adaptive(fn, lo, mid, left, tol * 0.5, depth + 1) + _quad_adaptive(
        fn, mid, hi, right, tol * 0.5, depth + 1
    )


def _integrate_angular(fn) -> float:
    half_pi = 0.5 * math.pi
    whole = _quad_panel(fn, 0.0, half_pi)
    tol = 1e-10 * abs(whole) if abs(whole) < 1e-8 else 1e-11
    return _quad_adaptive(fn, 0.0, half_pi, whole, tol, 0) * (2.0 / math.pi)


def bivariate_rejection(a: float, b: float, d: float, cv: float) -> float:
    """Use the Gaussian quadrant probability for (X - cv Y)(X + cv Y) > 0."""
    tol = 64 * np.finfo(float).eps
    magnitude = max(abs(a), abs(b), abs(d))
    if magnitude == 0:
        return 0.0
    v0, v1 = a / magnitude + tol, d / magnitude + tol
    if min(v0, v1) < 0 or abs(b) / magnitude > math.sqrt(v0) * math.sqrt(v1):
        raise ValueError("SCPC covariance is not positive semidefinite.")
    a, d = max(a, 0.0), max(d, 0.0)
    if min(a, d) == 0.0:
        return float(a > 0.0)
    # Evaluate atan2(2 c sqrt(ad-b²), c²d-a) in independently scaled units.
    # Exact products distinguish a tiny positive determinant from a strict tie.
    ea, ed = (max(-1022, math.frexp(v)[1] - 1) for v in (a, d))
    eb, parity = divmod(ea + ed, 2)
    an, dn = math.ldexp(a, -ea), math.ldexp(d, -ed + parity)
    t = cv * math.ldexp(1.0, eb - ea)
    bn = Fraction.from_float(b) / Fraction(2) ** eb
    an, dn = Fraction.from_float(an), Fraction.from_float(dn)
    root = math.sqrt(float(max(0, an * dn - bn * bn)))
    if t > 2**500:
        spread, difference = 2 * root / t, float(dn)
    else:
        tf = Fraction.from_float(t)
        spread, difference = 2 * t * root, float(tf * tf * dn - an)
    return math.atan2(spread, difference) / math.pi if spread else float(difference < 0)


def one_factor_rejection(om: np.ndarray, cv: float) -> float | None:
    """An exactly rank-one covariance has a deterministic strict rejection decision."""
    first = float(om[0, 0])
    if first <= 0 or first * float(om[1, 1]) != float(om[1, 0]) * float(om[1, 0]):
        return None
    # Comparing the exact binary products keeps a genuinely positive, tiny variance.
    an, ad = first.as_integer_ratio()
    for i in range(1, len(om)):
        cn, cd = float(om[i, 0]).as_integer_ratio()
        for j in range(1, i + 1):
            bn, bd = float(om[i, j]).as_integer_ratio()
            dn, dd = float(om[j, 0]).as_integer_ratio()
            if an * bn * cd * dd != cn * dn * ad * bd:
                return None
    if not np.array_equal(om, om.T):
        return None
    # A deterministic tie can be exactly on the requested threshold.
    denominator = sum(map(Fraction.from_float, map(float, om.diagonal()[1:])))
    return float(
        Fraction.from_float(first) > Fraction.from_float(float(cv)) ** 2 * denominator
    )


def rejection_eigenvalues(om: np.ndarray, cv: float, tolerance: float) -> np.ndarray:
    """Symmetric spectrum of L' D L for a positive semidefinite covariance L L'."""
    variances = om.diagonal()
    if np.min(variances) < -tolerance:
        raise ValueError("SCPC covariance is not positive semidefinite.")
    std = np.full(len(om), math.sqrt(max(0.0, float(variances[1:].max()))))
    std[0] = math.sqrt(max(0.0, float(variances[0])))
    zero = std == 0
    divisor = np.where(zero, 1.0, std)
    corr = (om / divisor[:, None]) / divisor[None, :]
    corr[zero, :] = 0
    corr[:, zero] = 0
    if np.any(np.abs(om[zero]) > tolerance) or not np.isfinite(corr).all():
        scale = max(0.0, float(variances.max()))
        if scale == 0:
            raise ValueError("SCPC covariance is not positive semidefinite.")
        corr = om / scale
        std.fill(math.sqrt(scale))
    try:
        factor = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(corr)
        if values[0] < -32 * len(om) * np.finfo(float).eps:
            # The input tolerance is relative to the original covariance block.
            scale = max(float(variances.max()), 0.0)
            if scale == 0:
                raise ValueError(
                    "SCPC covariance is not positive semidefinite."
                ) from None
            values, vectors = np.linalg.eigh(om / scale)
            if values[0] < -32 * len(om) * np.finfo(float).eps:
                raise ValueError(
                    "SCPC covariance is not positive semidefinite."
                ) from None
            std.fill(math.sqrt(scale))
        keep = values > 0
        factor = vectors[:, keep] * np.sqrt(values[keep])
    if cv > 1:
        std[0] /= cv
    else:
        std[1:] *= cv
    largest = std.max()
    if largest == 0 or factor.shape[1] == 0:
        return np.zeros(1)
    weighted = factor * (std / largest)[:, None]
    return np.linalg.eigvalsh(
        np.outer(weighted[0], weighted[0]) - weighted[1:].T @ weighted[1:]
    )


def get_rp(om: MatrixLike, cv: float) -> float:
    """Evaluate the rejection probability for one omega matrix.

    Uses the exact Cauchy formula for a single negative eigenvalue, handles
    zero-denominator covariances by returning 1.0 when the numerator is
    positive, and uses tail-stable adaptive angular/complement quadrature for
    higher dimensions.

    Args:
        om: Omega matrix for one spatial covariance case.
        cv: Normalized critical value.

    Returns:
        The rejection probability for that case.
    """
    if not math.isfinite(cv):
        raise ValueError("SCPC critical value must be finite.")
    if cv < 0.0:
        return 1.0
    om = np.asarray(om, dtype=np.float64)
    if (
        om.ndim != 2
        or om.shape[0] != om.shape[1]
        or len(om) < 2
        or not np.isfinite(om).all()
    ):
        raise ValueError(
            "SCPC covariance must be finite and square with at least two rows."
        )
    tolerance = (
        32 * len(om) * np.finfo(float).eps * max(0.0, float(om.diagonal().max()))
    )
    if np.any(np.abs(om - om.T) > tolerance):
        raise ValueError("SCPC covariance is not symmetric.")
    if len(om) > 2:
        active = np.flatnonzero(om.diagonal()[1:]) + 1
        if len(active) <= 1:
            j = active[0] if len(active) else 1
            omitted = np.arange(len(om))
            omitted = omitted[(omitted != 0) & (omitted != j)]
            if not om[omitted, :].any() and not om[:, omitted].any():
                om = om[np.ix_([0, j], [0, j])]
    if len(om) == 2:
        a, b, d = float(om[0, 0]), float(om[0, 1]), float(om[1, 1])
        return bivariate_rejection(a, b + 0.5 * (float(om[1, 0]) - b), d, cv)
    atom = one_factor_rejection(om, cv)
    if atom is not None:
        return atom
    evals_raw = rejection_eigenvalues(om, cv, tolerance)
    max_positive = float(np.max(evals_raw)) if evals_raw.size > 0 else 0.0
    if not math.isfinite(max_positive) or max_positive <= 0.0:
        return 0.0

    neg_evals = -evals_raw[evals_raw < 0.0]
    if neg_evals.size == 0:
        return 1.0

    ratios = neg_evals / max_positive
    logs = np.log(neg_evals) - math.log(max_positive)
    if ratios.size == 1:
        return float((2.0 / math.pi) * math.atan(math.exp(-0.5 * float(logs[0]))))

    scale = float(np.sum(ratios))
    if scale == 0.0:
        return 1.0

    if scale < 0.0625:
        deficit = _integrate_angular(
            lambda sin2, cos2: _complement_integrand(
                sin2, cos2 + scale * sin2, ratios, scale
            )
        )
        probability = 1.0 - math.sqrt(scale) * deficit
    else:
        probability = _integrate_angular(
            lambda _sin2, cos2: _angular_product(cos2, ratios, logs)
        )

    return float(min(1.0, max(0.0, probability)))


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
    step = 1.0 / math.sqrt(q)
    cv0 = stats.t.ppf(1 - level / 2, df=q) * step
    cv1 = cv0

    for _ in range(len(oms) + 5):
        if rp <= level:
            break
        cv1 = cv0
        for iter_idx in range(1024):
            if get_rp(oms[i][: q + 1, : q + 1], cv1) > level:
                cv0 = cv1
                cv1 += step if iter_idx < 32 else cv1
            else:
                break
        if get_rp(oms[i][: q + 1, : q + 1], cv1) > level:
            raise ValueError("SCPC critical value could not be bracketed.")

        while cv1 - cv0 > 0.001 * step:
            cv = 0.5 * (cv0 + cv1)
            if cv == cv0 or cv == cv1:
                break
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
            * math.exp(math.lgamma(0.5 * (q + 1)) - math.lgamma(0.5 * q))
            / math.sqrt(q)
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
    ratio = cmax / c0
    log_ratio = (
        math.log(ratio)
        if math.isfinite(ratio) and ratio > 0
        else math.log(cmax) - math.log(c0)
    )
    return max(2, math.ceil(log_ratio / math.log(cgridfac)))


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
    tied = False

    q_limit = (min(LARGE_N_M, n) - 1) if method_actual == "approx" else (n - 1)
    while True:
        qmax = min(qmax, q_limit)

        if method_actual == "exact":
            distmat = get_distmat(coords, latlong)
            distv = lvech(distmat)
            floor_corr = float(np.mean(distv == 0.0))
            minavc_eff = floor_corr + (1.0 - floor_corr) * MINAVC
            c0 = get_c0_from_avc(distv, avc0)
            cmax = get_c0_from_avc(distv, minavc_eff)
            w, tied = get_w_with_ties(distmat, c0, qmax)
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
            tied = False

        wfin, cvfin, q = set_final_w(oms, w, qmax)
        if q < qmax or qmax >= q_limit:
            break
        qmax = qmax + (qmax + 1) // 2

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
        tied=tied,
    )
