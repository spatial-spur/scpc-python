from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import pyfixest as pf

from scpc import scpc
from scpc.utils.matrix import (
    make_iv_residualizer,
    orthogonalize_w_iv,
    orthogonalize_w_cluster_iv,
)


def test_oblique_weights_match_direct_matrix():
    rng = np.random.default_rng(39)
    x = np.column_stack((np.ones(12), rng.normal(size=12)))
    z = np.column_stack((np.ones(12), rng.normal(size=(12, 2))))
    w = rng.normal(size=(6, 4))
    clusters = np.repeat(np.arange(6), 2)
    weights = rng.normal(size=12)
    sign = weights / np.sqrt(np.bincount(clusters, weights=weights**2)[clusters])
    pz = z @ np.linalg.solve(z.T @ z, z.T)
    m = np.eye(12) - x @ np.linalg.solve(x.T @ pz @ x, x.T @ pz)
    transposed = make_iv_residualizer(x, z, transpose=True)
    expanded = w[clusters]
    expected = np.column_stack(
        [
            np.bincount(
                clusters, weights=sign * (m.T @ (weights * expanded[:, j])), minlength=6
            )
            for j in range(1, 4)
        ]
    )
    actual = orthogonalize_w_cluster_iv(w, clusters, weights, transposed)
    np.testing.assert_allclose(actual[:, 1:], expected, atol=1e-12)
    sign = np.sign(weights)
    full = np.repeat(w, 2, axis=0)
    actual = orthogonalize_w_iv(full, weights, sign, transposed)
    np.testing.assert_allclose(
        actual[:, 1:],
        sign[:, None] * (m.T @ (weights[:, None] * full[:, 1:])),
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_identity_iv_matches_ols(backend):
    s = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
    x = np.array([-1.0, -4.0, 2.0, 2.0, 1.0])
    d = pd.DataFrame({"s": s, "x": x, "z": x, "y": x + (s == 0)})
    kw: dict[str, Any] = dict(
        coords_euclidean=["s"], avc=0.5, cvs=True, nthreads=1, backend=backend
    )
    ols = scpc(pf.feols("y ~ x", d, vcov="hetero"), d, **kw)
    iv = scpc(pf.feols("y ~ 1 | x ~ z", d, vcov="hetero"), d, **kw)
    np.testing.assert_allclose(iv.scpcstats, ols.scpcstats, atol=2e-8)
    assert iv.scpccvs is not None and ols.scpccvs is not None
    np.testing.assert_allclose(iv.scpccvs, ols.scpccvs, atol=2e-8)
    assert iv.scpccvs[1, 2] == pytest.approx(17.62678224, abs=1e-6)


@pytest.mark.parametrize("backend", ["fallback", "rust"])
@pytest.mark.parametrize("cluster", [None, "cl"])
def test_absorbed_fe_matches_dummies(backend, cluster):
    rng = np.random.default_rng(167)
    g = np.repeat(np.arange(10), 10)
    x = rng.normal(size=100) + np.repeat(np.linspace(-2, 2, 10), 10)
    y = 0.03 * x + np.sin(g) + rng.normal(size=100)
    cl = np.repeat(np.arange(50), 2)
    sx = np.repeat(rng.uniform(size=50), 2)
    sy = np.repeat(rng.uniform(size=50), 2)
    d = pd.DataFrame(dict(y=y, x=x, g=g, sx=sx, sy=sy, cl=cl))
    kw: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"],
        cvs=True,
        nthreads=1,
        backend=backend,
        cluster=cluster,
    )
    absorbed = scpc(pf.feols("y ~ x | g", d), d, **kw)
    dummy = scpc(smf.ols("y ~ x + C(g)", d).fit(), d, **kw)
    np.testing.assert_allclose(absorbed.scpcstats[0], dummy.scpcstats[-1], atol=2e-7)
    assert absorbed.scpccvs is not None and dummy.scpccvs is not None
    np.testing.assert_allclose(absorbed.scpccvs[0], dummy.scpccvs[-1], atol=2e-7)


def test_multiway_absorbed_fe_routes_to_fallback():
    rng = np.random.default_rng(109)
    d = pd.DataFrame({"g": np.repeat(np.arange(8), 8), "h": np.tile(np.arange(8), 8)})
    d["x"] = rng.normal(size=len(d))
    d["y"] = 0.3 * d.x + np.sin(d.g) + np.cos(d.h) + rng.normal(size=len(d))
    d["sx"] = rng.uniform(size=len(d))
    d["sy"] = rng.uniform(size=len(d))
    fit = pf.feols("y ~ x | g + h", d)
    kw: dict[str, Any] = dict(coords_euclidean=["sx", "sy"], cvs=True, nthreads=1)
    auto = scpc(fit, d, backend="auto", **kw)
    assert auto.backend == "fallback"
    dummy = scpc(smf.ols("y ~ x + C(g) + C(h)", d).fit(), d, backend="fallback", **kw)
    np.testing.assert_allclose(auto.scpcstats[0], dummy.scpcstats[-1], atol=2e-6)
    assert auto.scpccvs is not None and dummy.scpccvs is not None
    np.testing.assert_allclose(auto.scpccvs[0], dummy.scpccvs[-1], atol=2e-6)
    with pytest.raises(NotImplementedError, match="multi-way"):
        scpc(fit, d, backend="rust", **kw)


@pytest.mark.parametrize("backend", ["fallback", "rust"])
@pytest.mark.parametrize("cluster", [None, "cl"])
def test_identity_iv_with_fixed_effects_matches_ols(backend, cluster):
    rng = np.random.default_rng(120)
    g = np.repeat(np.arange(8), 8)
    x = rng.normal(size=64)
    d = pd.DataFrame(
        {
            "x": x,
            "z": x,
            "y": 0.4 * x + np.sin(g) + rng.normal(size=64),
            "g": g,
            "cl": np.repeat(np.arange(32), 2),
            "sx": np.repeat(rng.uniform(size=32), 2),
            "sy": np.repeat(rng.uniform(size=32), 2),
        }
    )
    opts: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"],
        cluster=cluster,
        backend=backend,
        cvs=True,
        nthreads=1,
    )
    ols = scpc(pf.feols("y ~ x | g", d), d, **opts)
    iv = scpc(pf.feols("y ~ 1 | g | x ~ z", d), d, **opts)
    np.testing.assert_allclose(ols.scpcstats, iv.scpcstats, atol=1e-7)
    assert ols.scpccvs is not None and iv.scpccvs is not None
    np.testing.assert_allclose(ols.scpccvs, iv.scpccvs, atol=1e-7)


def test_identity_iv_null_rejection_by_quasi_monte_carlo():
    from scipy.special import ndtri
    from scipy.stats import qmc
    from scpc.utils.spatial import set_oms_wfin

    sites = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
    x = np.array([-1.0, -4.0, 2.0, 2.0, 1.0])
    design = np.column_stack((np.ones(5), x))
    slope_weight = 5 * (design @ np.linalg.inv(design.T @ design)[:, 1])
    geometry = set_oms_wfin(
        np.column_stack((sites, np.zeros(5))),
        0.5,
        latlong=False,
        method="exact",
        large_n_seed=1,
    )
    weights = geometry.wfin.copy()
    weights[:, 0] *= abs(slope_weight)
    residual_maker = np.eye(5) - design @ np.linalg.inv(design.T @ design) @ design.T
    weights[:, 1:] = np.sign(slope_weight)[:, None] * (
        residual_maker @ (slope_weight[:, None] * weights[:, 1:])
    )
    sigma = np.exp(-geometry.c0 * np.abs(sites[:, None] - sites[None, :]))
    covariance = weights.T @ sigma @ weights
    draws = (
        ndtri(qmc.Sobol(d=4, scramble=True, seed=489).random_base2(19))
        @ np.linalg.cholesky(covariance).T
    )
    t = np.sqrt(3) * abs(draws[:, 0]) / np.linalg.norm(draws[:, 1:], axis=1)
    assert np.mean(t > 17.62678224) == pytest.approx(0.05, abs=0.0004)
    assert np.mean(t > 14.50861818) == pytest.approx(0.07143375445, abs=0.0004)
