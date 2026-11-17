from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyfixest as pf
import pytest
import spatial_spur_rust
import statsmodels.formula.api as smf

from scpc import scpc
from tests.utils import make_basic_iv_data

RUST_MOD = cast(Any, spatial_spur_rust)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "verification" / "data"
FIXTURE_PATH = REPO_ROOT / "spatial-rust" / "fixtures" / "m1_exact_scpc.json"

M1_CASES = [
    ("bench_200_euclid_cond", "bench_200.csv", False, False, True),
    ("bench_200_euclid_uncond", "bench_200.csv", False, True, False),
    ("bench_200_latlong_cond", "bench_200.csv", True, False, True),
    ("bench_1000_euclid_cond", "bench_1000.csv", False, False, True),
    ("bench_1000_euclid_uncond", "bench_1000.csv", False, True, False),
    ("bench_1000_latlong_cond", "bench_1000.csv", True, False, True),
]


HAS_MONOREPO_FIXTURES = (
    FIXTURE_PATH.exists()
    and (DATA_DIR / "bench_200.csv").exists()
    and (DATA_DIR / "bench_1000.csv").exists()
)


def load_fixtures() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


def load_bench_df(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / csv_name)


@pytest.mark.parametrize("engine", ["statsmodels", "pyfixest"])
@pytest.mark.parametrize(
    ("formula", "cutoff"), [("y ~ x", 17.625169), ("y ~ x - 1", 9.973326)]
)
def test_conditional_ols_cutoff_and_fitted_intercept(
    engine: str, formula: str, cutoff: float
) -> None:
    data = pd.DataFrame({"s": [0, 1, 2, 4, 6], "x": [-1, -4, 2, 2, 1]})
    data["y"] = data["x"] + (data["s"] == 0)
    fit = (
        smf.ols(formula, data=data).fit()
        if engine == "statsmodels"
        else pf.feols(formula, data)
    )
    for backend in ("fallback", "rust"):
        result = scpc(
            fit, data, coords_euclidean=["s"], avc=0.5, cvs=True, backend=backend
        )
        assert result.scpccvs is not None
        assert result.scpccvs[result.coef_names.index("x"), 2] == pytest.approx(
            cutoff, abs=0.002
        )


def make_small_ols_setup() -> tuple[Any, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "y": [1.0, 1.8, 2.9, 3.7, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "coord_x": [0.0, 1.0, 0.5, 1.5, 2.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 1.5],
        }
    )
    model = smf.ols("y ~ x", data=data).fit()
    return model, data


def test_spatial_spur_rust_api_version_fortran_layout_and_validation() -> None:
    assert RUST_MOD.API_VERSION == 2

    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 1.0],
            [2.0, 1.5],
        ],
        dtype=np.float64,
    )
    coef = np.array([1.0, 0.9], dtype=np.float64)
    wj = np.array(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.2, 0.1],
            [0.5, -0.3],
            [-0.4, 0.2],
        ],
        dtype=np.float64,
    )
    xj = np.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.1],
            [1.0, 0.6],
            [1.0, 1.2],
        ],
        dtype=np.float64,
    )
    xjs = np.sign(xj)
    u, _, _ = np.linalg.svd(xj, full_matrices=False)

    res_c = RUST_MOD.run_exact_scpc(
        coords,
        coef,
        wj,
        0.1,
        latlong=False,
        uncond=False,
        cvs=True,
        xj=xj,
        xjs=xjs,
        proj_u=u,
        proj_v=u,
        nthreads=1,
    )
    res_f = RUST_MOD.run_exact_scpc(
        np.asfortranarray(coords),
        coef,
        np.asfortranarray(wj),
        0.1,
        latlong=False,
        uncond=False,
        cvs=True,
        xj=np.asfortranarray(xj),
        xjs=np.asfortranarray(xjs),
        proj_u=np.asfortranarray(u),
        proj_v=np.asfortranarray(u),
        nthreads=1,
    )

    assert res_c["scpccvs"] is not None
    assert res_f["scpccvs"] is not None
    npt.assert_allclose(res_c["scpcstats"], res_f["scpcstats"], atol=1e-14, rtol=0.0)
    npt.assert_allclose(res_c["scpccvs"], res_f["scpccvs"], atol=1e-14, rtol=0.0)
    npt.assert_allclose(res_c["wfin"], res_f["wfin"], atol=1e-14, rtol=0.0)
    assert res_c["q"] == res_f["q"]
    assert res_c["c0"] == pytest.approx(res_f["c0"], abs=1e-14)
    assert res_c["cv"] == pytest.approx(res_f["cv"], abs=1e-14)

    bad_coords = coords.copy()
    bad_coords[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RUST_MOD.run_exact_scpc(bad_coords, coef, wj, 0.1, uncond=True)

    with pytest.raises(ValueError, match="avc"):
        RUST_MOD.run_exact_scpc(coords, coef, wj, 0.001, uncond=True)

    with pytest.raises(ValueError, match="wj"):
        RUST_MOD.run_exact_scpc(coords, coef, wj[:3, :], 0.1, uncond=True)

    with pytest.raises(ValueError, match="proj_u and proj_v"):
        RUST_MOD.run_exact_scpc(
            coords, coef, wj, 0.1, uncond=False, xj=xj, xjs=xjs, proj_u=u
        )

    with pytest.raises(ValueError, match="2 columns"):
        RUST_MOD.run_exact_scpc(coords[:, :1], coef, wj, 0.1, latlong=True, uncond=True)

    with pytest.raises(ValueError, match="nthreads"):
        RUST_MOD.run_exact_scpc(coords, coef, wj, 0.1, uncond=True, nthreads=0)

    with pytest.raises(TypeError, match="nthreads"):
        RUST_MOD.run_exact_scpc(coords, coef, wj, 0.1, uncond=True, nthreads=True)


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
@pytest.mark.parametrize(
    ("case_name", "csv_name", "latlong", "uncond", "cvs"), M1_CASES
)
@pytest.mark.parametrize("engine", ["statsmodels", "pyfixest"])
def test_m1_fixtures_parity_rust_vs_fallback_and_baseline(
    case_name: str,
    csv_name: str,
    latlong: bool,
    uncond: bool,
    cvs: bool,
    engine: str,
) -> None:
    fixtures = load_fixtures()
    expected = fixtures["python"][case_name][engine]
    df = load_bench_df(csv_name)

    coord_kwargs: dict[str, Any] = (
        {"lon": "s_1", "lat": "s_2"}
        if latlong
        else {"coords_euclidean": ["s_1", "s_2"]}
    )

    if engine == "statsmodels":
        fit = smf.ols("y ~ x1 + x2", data=df).fit()
    else:
        fit = pf.feols("y ~ x1 + x2", data=df, vcov="hetero")

    res_rust = scpc(
        fit, df, avc=0.03, uncond=uncond, cvs=cvs, backend="rust", **coord_kwargs
    )
    res_fb = scpc(
        fit, df, avc=0.03, uncond=uncond, cvs=cvs, backend="fallback", **coord_kwargs
    )

    assert res_rust.backend == "rust"
    assert res_fb.backend == "fallback"
    assert res_rust.method == "exact"
    assert res_fb.method == "exact"
    assert res_rust.q == expected["q"] == res_fb.q
    assert res_rust.c0 == pytest.approx(expected["c0"], abs=1e-10, rel=0.0)
    assert res_rust.cv == pytest.approx(expected["cv"], abs=1e-10, rel=0.0)
    assert res_rust.c0 == pytest.approx(res_fb.c0, abs=1e-10, rel=0.0)
    assert res_rust.cv == pytest.approx(res_fb.cv, abs=1e-10, rel=0.0)
    assert res_rust.coef_names == expected["coef_names"] == res_fb.coef_names

    # Frozen fixtures contain the historical conditional calibration.
    stable_cols = slice(None) if uncond else slice(3)
    npt.assert_allclose(
        res_rust.scpcstats[:, stable_cols],
        np.array(expected["scpcstats"])[:, stable_cols],
        atol=1e-10,
        rtol=0.0,
    )
    npt.assert_allclose(res_rust.scpcstats, res_fb.scpcstats, atol=1e-10, rtol=0.0)

    if cvs:
        assert res_rust.scpccvs is not None
        assert res_fb.scpccvs is not None
        if uncond:
            npt.assert_allclose(
                res_rust.scpccvs, np.array(expected["scpccvs"]), atol=1e-10, rtol=0.0
            )
        npt.assert_allclose(res_rust.scpccvs, res_fb.scpccvs, atol=1e-10, rtol=0.0)
    else:
        assert res_rust.scpccvs is None
        assert res_fb.scpccvs is None


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
@pytest.mark.parametrize("latlong", [False, True])
@pytest.mark.parametrize(("uncond", "cvs"), [(True, True), (False, False)])
def test_m1_additional_uncond_and_cvs_combinations(
    latlong: bool, uncond: bool, cvs: bool
) -> None:
    df = load_bench_df("bench_200.csv")
    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    coord_kwargs: dict[str, Any] = (
        {"lon": "s_1", "lat": "s_2"}
        if latlong
        else {"coords_euclidean": ["s_1", "s_2"]}
    )

    res_rust = scpc(
        fit, df, avc=0.03, uncond=uncond, cvs=cvs, backend="rust", **coord_kwargs
    )
    res_fb = scpc(
        fit, df, avc=0.03, uncond=uncond, cvs=cvs, backend="fallback", **coord_kwargs
    )

    assert res_rust.backend == "rust"
    assert res_fb.backend == "fallback"
    assert res_rust.q == res_fb.q
    assert res_rust.c0 == pytest.approx(res_fb.c0, abs=1e-10, rel=0.0)
    assert res_rust.cv == pytest.approx(res_fb.cv, abs=1e-10, rel=0.0)
    npt.assert_allclose(res_rust.scpcstats, res_fb.scpcstats, atol=1e-10, rtol=0.0)
    if cvs:
        assert res_rust.scpccvs is not None
        assert res_fb.scpccvs is not None
        npt.assert_allclose(res_rust.scpccvs, res_fb.scpccvs, atol=1e-10, rtol=0.0)
    else:
        assert res_rust.scpccvs is None


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
def test_rust_nthreads_1_runs_rust_and_matches_nthreads_4_and_fallback() -> None:
    df = load_bench_df("bench_200.csv")
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    res_1 = scpc(
        fit,
        df,
        coords_euclidean=["s_1", "s_2"],
        avc=0.03,
        uncond=False,
        cvs=True,
        backend="rust",
        nthreads=1,
    )
    res_4 = scpc(
        fit,
        df,
        coords_euclidean=["s_1", "s_2"],
        avc=0.03,
        uncond=False,
        cvs=True,
        backend="rust",
        nthreads=4,
    )
    res_fb = scpc(
        fit,
        df,
        coords_euclidean=["s_1", "s_2"],
        avc=0.03,
        uncond=False,
        cvs=True,
        backend="fallback",
    )

    assert res_1.backend == "rust"
    assert res_4.backend == "rust"
    assert res_fb.backend == "fallback"
    assert res_1.q == res_4.q == res_fb.q
    npt.assert_allclose(res_1.scpcstats, res_4.scpcstats, atol=1e-12, rtol=0.0)
    npt.assert_allclose(res_1.scpcstats, res_fb.scpcstats, atol=1e-10, rtol=0.0)
    assert res_1.scpccvs is not None
    assert res_4.scpccvs is not None
    assert res_fb.scpccvs is not None
    npt.assert_allclose(res_1.scpccvs, res_4.scpccvs, atol=1e-12, rtol=0.0)
    npt.assert_allclose(res_1.scpccvs, res_fb.scpccvs, atol=1e-10, rtol=0.0)


def test_spatial_backend_env_var_and_argument_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = make_small_ols_setup()

    monkeypatch.setenv("SPATIAL_BACKEND", "fallback")
    res_env_fb = scpc(model, data, coords_euclidean=("coord_x", "coord_y"), uncond=True)
    assert res_env_fb.backend == "fallback"

    res_override_rust = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        uncond=True,
        backend="rust",
    )
    assert res_override_rust.backend == "rust"

    monkeypatch.setenv("SPATIAL_BACKEND", "rust")
    res_env_rust = scpc(
        model, data, coords_euclidean=("coord_x", "coord_y"), uncond=True
    )
    assert res_env_rust.backend == "rust"

    res_override_fb = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        uncond=True,
        backend="fallback",
    )
    assert res_override_fb.backend == "fallback"

    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)
    res_auto = scpc(model, data, coords_euclidean=("coord_x", "coord_y"), uncond=True)
    assert res_auto.backend == "rust"

    with pytest.raises(ValueError, match="backend"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"), backend="bad")

    monkeypatch.setenv("SPATIAL_BACKEND", "bad")
    with pytest.raises(ValueError, match="SPATIAL_BACKEND"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"))
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)

    for bad_nthreads in (0, -2, True):
        with pytest.raises(ValueError, match="nthreads"):
            scpc(
                model,
                data,
                coords_euclidean=("coord_x", "coord_y"),
                nthreads=bad_nthreads,
            )


def test_unsupported_modes_raise_on_rust_and_fallback_on_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)

    # 1. Clustered inference
    cl_data = pd.DataFrame(
        {
            "y": [2.1, 2.3, 0.9, 1.2, 1.6, 1.8],
            "x": [1.0, 1.2, 0.0, 0.2, 0.5, 0.7],
            "coord_x": [2.0, 2.0, 0.0, 0.0, 1.0, 1.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "cluster": ["a", "a", "b", "b", "c", "c"],
        }
    )
    cl_model = smf.ols("y ~ x", data=cl_data).fit()
    with pytest.raises(NotImplementedError, match="clustered"):
        scpc(
            cl_model,
            cl_data,
            coords_euclidean=("coord_x", "coord_y"),
            cluster="cluster",
            backend="rust",
        )
    res_cl_auto = scpc(
        cl_model,
        cl_data,
        coords_euclidean=("coord_x", "coord_y"),
        cluster="cluster",
        backend="auto",
    )
    assert res_cl_auto.backend == "fallback"

    # 2. Approximate large-n branch
    model, data = make_small_ols_setup()
    with pytest.raises(NotImplementedError, match="approximate"):
        scpc(
            model,
            data,
            coords_euclidean=("coord_x", "coord_y"),
            method="approx",
            backend="rust",
        )
    res_approx_auto = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        method="approx",
        backend="auto",
    )
    assert res_approx_auto.backend == "fallback"

    # 3. Conditional IV
    iv_data = make_basic_iv_data(2001)
    iv_fit = pf.feols("y ~ w | x ~ z", data=iv_data)
    with pytest.raises(NotImplementedError, match="instrumental"):
        scpc(
            iv_fit,
            iv_data,
            coords_euclidean=("coord_x", "coord_y"),
            uncond=False,
            backend="rust",
        )
    res_iv_auto = scpc(
        iv_fit,
        iv_data,
        coords_euclidean=("coord_x", "coord_y"),
        uncond=False,
        backend="auto",
    )
    assert res_iv_auto.backend == "fallback"

    # 4. Missing extension module
    monkeypatch.setattr("scpc.core.load_rust_backend", lambda: None)
    with pytest.raises(RuntimeError, match="spatial_spur_rust"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"), backend="rust")
    res_missing_auto = scpc(
        model, data, coords_euclidean=("coord_x", "coord_y"), backend="auto"
    )
    assert res_missing_auto.backend == "fallback"


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
def test_1d_coords_ncoef_and_tied_eigenvalues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)

    df = load_bench_df("bench_200.csv")
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    res_1d_rust = scpc(
        fit,
        df,
        coords_euclidean=("s_1",),
        avc=0.03,
        uncond=False,
        cvs=True,
        ncoef=2,
        backend="rust",
    )
    res_1d_fb = scpc(
        fit,
        df,
        coords_euclidean=("s_1",),
        avc=0.03,
        uncond=False,
        cvs=True,
        ncoef=2,
        backend="fallback",
    )
    assert res_1d_rust.backend == "rust"
    assert res_1d_rust.coef_names == ["Intercept", "x1"]
    npt.assert_allclose(
        res_1d_rust.scpcstats, res_1d_fb.scpcstats, atol=1e-10, rtol=0.0
    )
    assert res_1d_rust.scpccvs is not None
    assert res_1d_fb.scpccvs is not None
    npt.assert_allclose(res_1d_rust.scpccvs, res_1d_fb.scpccvs, atol=1e-10, rtol=0.0)

    # Symmetric square coordinates trigger tied eigenvalues
    tied_data = pd.DataFrame(
        {
            "y": [1.0, 2.2, 2.9, 4.4],
            "x": [0.0, 1.0, 2.0, 3.0],
            "coord_x": [0.0, 1.0, 0.0, 1.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    tied_fit = smf.ols("y ~ x", data=tied_data).fit()
    res_tied_auto = scpc(
        tied_fit,
        tied_data,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        uncond=True,
        backend="auto",
    )
    assert res_tied_auto.backend == "fallback"

    res_tied_rust = scpc(
        tied_fit,
        tied_data,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        uncond=True,
        backend="rust",
    )
    assert res_tied_rust.backend == "rust"
