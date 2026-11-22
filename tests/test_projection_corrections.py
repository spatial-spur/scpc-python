from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import pyfixest as pf

from scpc import scpc
from scpc.core import load_rust_backend
from scpc.utils.matrix import (
    make_iv_residualizer,
    project_conditional_weights,
    project_clustered_weights,
)


RUST_AVAILABLE = load_rust_backend() is not None
RUST_PARAMETER = pytest.param(
    "rust",
    marks=pytest.mark.skipif(
        not RUST_AVAILABLE, reason="compatible native extension unavailable"
    ),
)


def test_cluster_direction_rescaling_preserves_a_tiny_clusters_signs():
    w = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    ids = np.repeat([0, 1], 2)
    xj = np.array([1e308, -1e308, 1e-300, -1e-300])
    v = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2)
    projector = np.eye(4) - np.outer(v, v)
    result = project_clustered_weights(
        w, ids, xj, lambda a: projector @ a, rescale=True
    )
    # The nuisance projection transfers a nonzero component to the tiny group.
    # Its within-cluster direction still has magnitude 1/sqrt(2) there.
    np.testing.assert_allclose(result, [[1.0, 0.75], [0.0, -0.25]], atol=1e-15)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_singleton_labels_preserve_a_zero_coefficient_direction(backend):
    x = np.arange(-2.0, 3.0)
    d = pd.DataFrame(
        {
            "x": x,
            "y": x + np.array([1.0, -2.0, 0.0, 2.0, -1.0]),
            "s": np.random.default_rng(42).uniform(size=5),
            "group": np.arange(5),
        }
    )
    fit = pf.feols("y ~ x", d)
    opts: dict[str, Any] = dict(
        coords_euclidean=["s"], avc=0.3, cvs=True, backend=backend, nthreads=1
    )
    flat = scpc(fit, d, **opts)
    singleton = scpc(fit, d, cluster="group", **opts)
    assert singleton.q == flat.q
    np.testing.assert_allclose(singleton.scpcstats, flat.scpcstats, atol=1e-10)
    assert singleton.scpccvs is not None and flat.scpccvs is not None
    np.testing.assert_allclose(singleton.scpccvs, flat.scpccvs, atol=1e-10)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_large_intercept_shift_preserves_the_standard_error(backend):
    n = 30
    d = pd.DataFrame({"y": (np.arange(n) % 7 - 3) / 4, "s": np.linspace(0.0, 1.0, n)})
    shifted = d.assign(y=d["y"] + 1e12)
    opts: dict[str, Any] = dict(coords_euclidean=["s"], backend=backend, nthreads=1)
    base = scpc(smf.ols("y ~ 1", d).fit(), d, **opts)
    result = scpc(smf.ols("y ~ 1", shifted).fit(), shifted, **opts)
    beta, se, t = result.scpcstats[0, :3]
    assert se == pytest.approx(base.scpcstats[0, 1], rel=1e-12)
    assert t == pytest.approx(beta / se, rel=1e-12)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_unconditional_scpc_skips_conditional_design(monkeypatch, backend):
    rng = np.random.default_rng(182)
    df = pd.DataFrame(rng.normal(size=(30, 4)), columns=["y", "x", "s1", "s2"])
    model = smf.ols("y ~ x", data=df).fit()

    def unexpected_design(*args, **kwargs):
        raise AssertionError(
            "Unconditional SCPC tried to construct a conditional design"
        )

    monkeypatch.setattr("scpc.core.get_conditional_projection_setup", unexpected_design)
    out = scpc(model, df, coords_euclidean=["s1", "s2"], uncond=True, backend=backend)
    assert np.isfinite(out.scpcstats).all()


def test_pyfixest_model_without_ols_metadata_is_rejected():
    rng = np.random.default_rng(188)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["y", "x", "s"])
    model = pf.feols("y ~ x", df)
    del model._method
    with pytest.raises(NotImplementedError, match="OLS"):
        scpc(model, df, coords_euclidean=["s"], backend="fallback")


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
    actual = project_clustered_weights(w, clusters, weights, transposed)
    np.testing.assert_allclose(actual[:, 1:], expected, atol=1e-12)
    sign = np.sign(weights)
    full = np.repeat(w, 2, axis=0)
    actual = project_conditional_weights(full, weights, sign, transposed)
    np.testing.assert_allclose(
        actual[:, 1:],
        sign[:, None] * (m.T @ (weights[:, None] * full[:, 1:])),
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
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


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
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


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_absorbed_ols_rebuilds_the_same_formula_basis_for_both_backends(backend):
    rng = np.random.default_rng(799)
    n = 40
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x", "y", "s"])
    df["fe"] = np.repeat(np.arange(n // 5), 5)

    def square(x):
        return x**2

    fit = pf.feols("y ~ square(x) | fe", df, context={"square": square})
    kwargs: dict[str, Any] = dict(coords_euclidean=["s"], cvs=True, nthreads=1)
    expected = scpc(fit, df, backend="fallback", **kwargs)
    # Simulate nuisance leakage in the stored, already-demeaned fit matrix.
    fit._X = fit._X + 1e4 * np.asarray(df["fe"])[:, None]
    actual = scpc(fit, df, backend=backend, **kwargs)
    np.testing.assert_allclose(actual.scpcstats, expected.scpcstats, atol=1e-7)
    assert actual.scpccvs is not None and expected.scpccvs is not None
    np.testing.assert_allclose(actual.scpccvs, expected.scpccvs, atol=1e-7)


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
    if RUST_AVAILABLE:
        with pytest.raises(NotImplementedError, match="multi-way"):
            scpc(fit, d, backend="rust", **kw)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
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


def test_pyfixest_transforms_and_categorical_fe_parity() -> None:
    rng = np.random.default_rng(20261121)
    n = 60
    df = pd.DataFrame(
        {
            "fe": np.repeat(np.arange(10), 6),
            "cat": np.tile([0, 1, 2], 20),
            "x": np.exp(rng.normal(size=n)),
            "sx": rng.uniform(size=n),
            "sy": rng.uniform(size=n),
        }
    )
    df["y"] = (
        0.5 * (df["cat"] == 1)
        - 0.3 * (df["cat"] == 2)
        + 0.4 * np.log(df["x"])
        + np.sin(df["fe"])
        + rng.normal(size=n)
    )
    kw: dict[str, Any] = dict(coords_euclidean=["sx", "sy"], cvs=True, nthreads=1)
    for fml in (
        "y ~ i(cat) | fe",
        "y ~ i(cat, x) | fe",
        "y ~ log(x) | fe",
        "y ~ C(cat) | fe",
    ):
        fit = pf.feols(fml, df)
        fb = scpc(fit, df, backend="fallback", **kw)
        assert np.isfinite(fb.scpcstats).all()
        if RUST_AVAILABLE:
            rs = scpc(fit, df, backend="rust", **kw)
            np.testing.assert_allclose(rs.scpcstats, fb.scpcstats, rtol=1e-7, atol=1e-7)
            assert rs.scpccvs is not None and fb.scpccvs is not None
            np.testing.assert_allclose(rs.scpccvs, fb.scpccvs, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
@pytest.mark.parametrize("iv", [False, True], ids=["ols", "iv"])
def test_pyfixest_conditional_scpc_preserves_fitted_scale(
    backend: str, iv: bool
) -> None:
    rng = np.random.default_rng(20261122)
    n = 60
    data = pd.DataFrame(
        {
            "fe": np.repeat(np.arange(10), 6),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "w": rng.normal(size=n),
            "sx": rng.uniform(size=n),
            "sy": rng.uniform(size=n),
        }
    )
    data["x"] += 0.5 * data["w"]
    data["y"] = (
        0.75 * data["x"] + 0.3 * data["z"] + np.sin(data["fe"]) + rng.normal(size=n)
    )
    # Formulaic learns the scales before discarding rows with missing outcomes.
    data.loc[::6, ["x", "z", "w"]] *= 100
    data.loc[::6, "y"] = np.nan
    for name in ("x", "z", "w"):
        data[f"{name}s"] = (data[name] - data[name].mean()) / data[name].std()

    formula, explicit = (
        ("y ~ scale(z) | fe | x ~ scale(w)", "y ~ zs | fe | x ~ ws")
        if iv
        else ("y ~ scale(x) + z | fe", "y ~ xs + z | fe")
    )
    fit = pf.feols(formula, data=data)
    reference = pf.feols(explicit, data=data)
    np.testing.assert_allclose(fit.coef(), reference.coef(), atol=1e-12)
    np.testing.assert_allclose(fit.resid(), reference.resid(), atol=1e-12)

    result = scpc(fit, data, coords_euclidean=["sx", "sy"], backend=backend, nthreads=1)
    expected = scpc(
        reference, data, coords_euclidean=["sx", "sy"], backend=backend, nthreads=1
    )
    np.testing.assert_allclose(
        result.scpcstats, expected.scpcstats, atol=1e-8, rtol=1e-8
    )
