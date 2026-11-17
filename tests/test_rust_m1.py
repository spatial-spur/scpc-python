from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyfixest as pf
import pytest
import statsmodels.formula.api as smf

from scpc import scpc
from scpc.core import load_rust_backend
from scpc.utils.matrix import lvech
from scpc.utils.spatial import get_avc, get_distmat, get_rp
from tests.utils import make_basic_iv_data

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


def rust_available() -> bool:
    return load_rust_backend() is not None


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
    backends = ("fallback", "rust") if rust_available() else ("fallback",)
    for backend in backends:
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


def make_direct_ffi_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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
    return coords, coef, wj, xj, xjs, u


def test_spatial_spur_rust_api_version_fortran_layout_and_validation() -> None:
    rust_mod: Any = pytest.importorskip("spatial_spur_rust")
    assert rust_mod.API_VERSION == 2

    coords, coef, wj, xj, xjs, u = make_direct_ffi_inputs()

    res_c = rust_mod.run_exact_scpc(
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
    res_f = rust_mod.run_exact_scpc(
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
        rust_mod.run_exact_scpc(bad_coords, coef, wj, 0.1, uncond=True)

    with pytest.raises(ValueError, match="avc"):
        rust_mod.run_exact_scpc(coords, coef, wj, 0.001, uncond=True)

    with pytest.raises(ValueError, match="wj"):
        rust_mod.run_exact_scpc(coords, coef, wj[:3, :], 0.1, uncond=True)

    with pytest.raises(ValueError, match="proj_u and proj_v"):
        rust_mod.run_exact_scpc(
            coords, coef, wj, 0.1, uncond=False, xj=xj, xjs=xjs, proj_u=u
        )

    with pytest.raises(ValueError, match="2 columns"):
        rust_mod.run_exact_scpc(coords[:, :1], coef, wj, 0.1, latlong=True, uncond=True)

    with pytest.raises(ValueError, match="nthreads"):
        rust_mod.run_exact_scpc(coords, coef, wj, 0.1, uncond=True, nthreads=0)

    with pytest.raises(TypeError, match="nthreads"):
        rust_mod.run_exact_scpc(coords, coef, wj, 0.1, uncond=True, nthreads=True)


def test_rust_ffi_arbitrary_strides_unaligned_and_sliced_views() -> None:
    rust_mod: Any = pytest.importorskip("spatial_spur_rust")
    coords, coef, wj, xj, xjs, u = make_direct_ffi_inputs()

    ref = rust_mod.run_exact_scpc(
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

    def make_byte_strided_2d(src: np.ndarray, strides: tuple[int, int]) -> np.ndarray:
        buf = np.zeros(1024, dtype=np.uint8)
        view = np.ndarray(src.shape, dtype=np.float64, buffer=buf, strides=strides)
        view[:] = src
        return view

    def make_unaligned_c(src: np.ndarray) -> np.ndarray:
        buf = np.zeros(src.nbytes + 64, dtype=np.uint8)
        view = np.ndarray(src.shape, dtype=np.float64, buffer=buf[1:])
        view[:] = src
        assert not view.flags["ALIGNED"]
        return view

    # (24, 9) non-multiple-of-8 byte stride
    res_strided = rust_mod.run_exact_scpc(
        make_byte_strided_2d(coords, (24, 9)),
        coef,
        make_byte_strided_2d(wj, (24, 9)),
        0.1,
        uncond=False,
        cvs=True,
        xj=make_byte_strided_2d(xj, (24, 9)),
        xjs=make_byte_strided_2d(xjs, (24, 9)),
        proj_u=make_byte_strided_2d(u, (24, 9)),
        proj_v=make_byte_strided_2d(u, (24, 9)),
        nthreads=1,
    )
    npt.assert_allclose(
        res_strided["scpcstats"], ref["scpcstats"], atol=1e-14, rtol=0.0
    )
    assert res_strided["c0"] == pytest.approx(ref["c0"], rel=1e-14, abs=1e-14)

    # Unaligned C-contiguous views (offset by 1 byte)
    res_unaligned = rust_mod.run_exact_scpc(
        make_unaligned_c(coords),
        make_unaligned_c(coef),
        make_unaligned_c(wj),
        0.1,
        uncond=False,
        cvs=True,
        xj=make_unaligned_c(xj),
        xjs=make_unaligned_c(xjs),
        proj_u=make_unaligned_c(u),
        proj_v=make_unaligned_c(u),
        nthreads=1,
    )
    npt.assert_allclose(
        res_unaligned["scpcstats"], ref["scpcstats"], atol=1e-14, rtol=0.0
    )
    assert res_unaligned["c0"] == pytest.approx(ref["c0"], rel=1e-14, abs=1e-14)

    # Negative-stride 2D slices and non-contiguous / negative 1D slices
    coef_step2 = np.array([coef[0], 999.0, coef[1], -999.0], dtype=np.float64)[::2]
    coef_rev = coef[::-1].copy()[::-1]
    assert coef_step2.strides[0] == 16
    assert coef_rev.strides[0] == -8

    coords_neg = coords[::-1, ::-1].copy()[::-1, ::-1]
    wj_neg = wj[::-1, ::-1].copy()[::-1, ::-1]
    xj_neg = xj[::-1, ::-1].copy()[::-1, ::-1]
    xjs_neg = xjs[::-1, ::-1].copy()[::-1, ::-1]
    u_neg = u[::-1, ::-1].copy()[::-1, ::-1]
    assert coords_neg.strides[0] < 0 and coords_neg.strides[1] < 0

    for coef_view in (coef_step2, coef_rev):
        res_neg = rust_mod.run_exact_scpc(
            coords_neg,
            coef_view,
            wj_neg,
            0.1,
            uncond=False,
            cvs=True,
            xj=xj_neg,
            xjs=xjs_neg,
            proj_u=u_neg,
            proj_v=u_neg,
            nthreads=1,
        )
        npt.assert_allclose(
            res_neg["scpcstats"], ref["scpcstats"], atol=1e-14, rtol=0.0
        )
        assert res_neg["c0"] == pytest.approx(ref["c0"], rel=1e-14, abs=1e-14)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork not available")
def test_rust_fork_safety_after_warming_threads() -> None:
    rust_mod: Any = pytest.importorskip("spatial_spur_rust")
    coords, coef, wj, xj, xjs, u = make_direct_ffi_inputs()

    # Warm both serial and multi-threaded pool paths in the parent process.
    ref1 = rust_mod.run_exact_scpc(
        coords,
        coef,
        wj,
        0.1,
        uncond=False,
        cvs=True,
        xj=xj,
        xjs=xjs,
        proj_u=u,
        proj_v=u,
        nthreads=1,
    )
    rust_mod.run_exact_scpc(
        coords,
        coef,
        wj,
        0.1,
        uncond=False,
        cvs=True,
        xj=xj,
        xjs=xjs,
        proj_u=u,
        proj_v=u,
        nthreads=2,
    )

    pid = os.fork()
    if pid == 0:
        try:
            child_res = rust_mod.run_exact_scpc(
                coords,
                coef,
                wj,
                0.1,
                uncond=False,
                cvs=True,
                xj=xj,
                xjs=xjs,
                proj_u=u,
                proj_v=u,
                nthreads=1,
            )
            if child_res["q"] != ref1["q"]:
                os._exit(2)
            if not np.allclose(
                child_res["scpcstats"], ref1["scpcstats"], atol=1e-14, rtol=0.0
            ):
                os._exit(3)
            os._exit(0)
        except BaseException:
            os._exit(1)

    deadline = time.monotonic() + 10.0
    status: int | None = None
    while time.monotonic() < deadline:
        waited_pid, waited_status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            status = waited_status
            break
        time.sleep(0.01)

    if status is None:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise AssertionError(
            "forked child deadlocked or timed out in run_exact_scpc(nthreads=1)"
        )

    assert os.WIFEXITED(status)
    assert os.waitstatus_to_exitcode(status) == 0


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

    res_fb = scpc(
        fit, df, avc=0.03, uncond=uncond, cvs=cvs, backend="fallback", **coord_kwargs
    )
    assert res_fb.backend == "fallback"
    assert res_fb.method == "exact"
    assert res_fb.tied is False
    assert res_fb.q == expected["q"]
    assert res_fb.coef_names == expected["coef_names"]

    dvec = lvech(get_distmat(df[["s_1", "s_2"]].to_numpy(dtype=np.float64), latlong))
    assert abs(get_avc(res_fb.c0, dvec) - 0.03) < 1e-12

    # Frozen fixtures reflect the historical 0.001-bracket c0 solver (~2e-3 relative).
    assert res_fb.c0 == pytest.approx(expected["c0"], rel=2e-3, abs=1e-12)
    assert res_fb.cv == pytest.approx(expected["cv"], rel=2e-3, abs=1e-12)
    expected_stats = np.array(expected["scpcstats"])
    if uncond:
        npt.assert_allclose(
            res_fb.scpcstats[:, [0, 1, 2, 4, 5]],
            expected_stats[:, [0, 1, 2, 4, 5]],
            rtol=2e-3,
            atol=1e-10,
        )
        npt.assert_allclose(
            res_fb.scpcstats[:, 3],
            expected_stats[:, 3],
            rtol=1e-2,
            atol=1e-7,
        )
    else:
        npt.assert_allclose(
            res_fb.scpcstats[:, :3],
            expected_stats[:, :3],
            rtol=2e-3,
            atol=1e-10,
        )
    if cvs:
        assert res_fb.scpccvs is not None
        if uncond:
            npt.assert_allclose(
                res_fb.scpccvs, np.array(expected["scpccvs"]), rtol=2e-3, atol=1e-10
            )
    else:
        assert res_fb.scpccvs is None

    if rust_available():
        res_rust = scpc(
            fit, df, avc=0.03, uncond=uncond, cvs=cvs, backend="rust", **coord_kwargs
        )
        assert res_rust.backend == "rust"
        assert res_rust.method == "exact"
        assert res_rust.tied is False
        assert res_rust.q == res_fb.q
        assert res_rust.coef_names == res_fb.coef_names
        assert abs(get_avc(res_rust.c0, dvec) - 0.03) < 1e-12
        assert res_rust.c0 == pytest.approx(res_fb.c0, rel=1e-10, abs=1e-10)
        assert res_rust.cv == pytest.approx(res_fb.cv, rel=1e-10, abs=1e-10)
        npt.assert_allclose(
            res_rust.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-10
        )
        if cvs:
            assert res_rust.scpccvs is not None
            assert res_fb.scpccvs is not None
            npt.assert_allclose(
                res_rust.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-10
            )
        else:
            assert res_rust.scpccvs is None


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
@pytest.mark.parametrize("latlong", [False, True])
@pytest.mark.parametrize(("uncond", "cvs"), [(True, True), (False, False)])
def test_m1_additional_uncond_and_cvs_combinations(
    latlong: bool, uncond: bool, cvs: bool
) -> None:
    pytest.importorskip("spatial_spur_rust")
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
    assert res_rust.c0 == pytest.approx(res_fb.c0, rel=1e-10, abs=1e-10)
    assert res_rust.cv == pytest.approx(res_fb.cv, rel=1e-10, abs=1e-10)
    npt.assert_allclose(res_rust.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-10)
    if cvs:
        assert res_rust.scpccvs is not None
        assert res_fb.scpccvs is not None
        npt.assert_allclose(res_rust.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-10)
    else:
        assert res_rust.scpccvs is None


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
def test_rust_nthreads_1_runs_rust_and_matches_nthreads_4_and_fallback() -> None:
    pytest.importorskip("spatial_spur_rust")
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


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
def test_scale_invariant_calibration_and_colocation_floor() -> None:
    df_full = load_bench_df("bench_200.csv")
    df_30 = df_full.iloc[:30].copy()
    fit_30 = smf.ols("y ~ x1 + x2", data=df_30).fit()
    fit_full = smf.ols("y ~ x1 + x2", data=df_full).fit()
    backends = ("fallback", "rust") if rust_available() else ("fallback",)

    # 30-row handoff reproducer: c0 ~ 6099987.97 when coords scaled by 1e-6, oracle = 0.03
    for backend in backends:
        res_30_base = scpc(
            fit_30,
            df_30,
            coords_euclidean=["s_1", "s_2"],
            avc=0.03,
            uncond=False,
            cvs=True,
            backend=backend,
        )
        d_30_base = lvech(
            get_distmat(df_30[["s_1", "s_2"]].to_numpy(dtype=np.float64), False)
        )
        assert abs(get_avc(res_30_base.c0, d_30_base) - 0.03) < 1e-12

        for scale in (1e-6, 1e6):
            df_30_scaled = df_30.copy()
            df_30_scaled["s_1"] = df_30_scaled["s_1"] * scale
            df_30_scaled["s_2"] = df_30_scaled["s_2"] * scale
            res_30_scaled = scpc(
                fit_30,
                df_30_scaled,
                coords_euclidean=["s_1", "s_2"],
                avc=0.03,
                uncond=False,
                cvs=True,
                backend=backend,
            )
            d_30_scaled = lvech(
                get_distmat(
                    df_30_scaled[["s_1", "s_2"]].to_numpy(dtype=np.float64), False
                )
            )
            assert abs(get_avc(res_30_scaled.c0, d_30_scaled) - 0.03) < 1e-12
            assert res_30_scaled.c0 * scale == pytest.approx(
                res_30_base.c0, rel=1e-10, abs=1e-12
            )
            assert res_30_scaled.q == res_30_base.q
            if scale == 1e-6:
                assert res_30_scaled.c0 == pytest.approx(6099987.9657, rel=1e-6)

        # Full untied bench_200 checks exact scpcstats & scpccvs invariance under scaling
        res_full_base = scpc(
            fit_full,
            df_full,
            coords_euclidean=["s_1", "s_2"],
            avc=0.03,
            uncond=False,
            cvs=True,
            backend=backend,
        )
        assert res_full_base.tied is False
        for scale in (1e-6, 1e6):
            df_full_scaled = df_full.copy()
            df_full_scaled["s_1"] = df_full_scaled["s_1"] * scale
            df_full_scaled["s_2"] = df_full_scaled["s_2"] * scale
            res_full_scaled = scpc(
                fit_full,
                df_full_scaled,
                coords_euclidean=["s_1", "s_2"],
                avc=0.03,
                uncond=False,
                cvs=True,
                backend=backend,
            )
            assert res_full_scaled.tied is False
            assert res_full_scaled.q == res_full_base.q
            assert res_full_scaled.c0 * scale == pytest.approx(
                res_full_base.c0, rel=1e-10, abs=1e-12
            )
            npt.assert_allclose(
                res_full_scaled.scpcstats,
                res_full_base.scpcstats,
                rtol=1e-10,
                atol=1e-10,
            )
            assert res_full_scaled.scpccvs is not None
            assert res_full_base.scpccvs is not None
            npt.assert_allclose(
                res_full_scaled.scpccvs,
                res_full_base.scpccvs,
                rtol=1e-10,
                atol=1e-10,
            )

    # Co-location floor: 6 rows with 2 duplicated pairs out of 15 -> floor = 2/15
    coloc_df = pd.DataFrame(
        {
            "y": [1.0, 2.5, 2.7, 4.9, 5.2, 7.8],
            "x": [0.0, 0.8, 1.3, 2.1, 3.0, 4.2],
            "s_1": [0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
            "s_2": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        }
    )
    coloc_fit = smf.ols("y ~ x", data=coloc_df).fit()
    coloc_distv = lvech(
        get_distmat(coloc_df[["s_1", "s_2"]].to_numpy(dtype=np.float64), False)
    )
    floor_corr = float(np.mean(coloc_distv == 0.0))
    assert floor_corr == pytest.approx(2.0 / 15.0, abs=1e-15)

    for backend in backends:
        res_ok = scpc(
            coloc_fit,
            coloc_df,
            coords_euclidean=["s_1", "s_2"],
            avc=0.5,
            uncond=True,
            backend=backend,
        )
        assert abs(get_avc(res_ok.c0, coloc_distv) - 0.5) < 1e-12

        for bad_avc in (floor_corr, floor_corr - 0.05):
            with pytest.raises(ValueError):
                scpc(
                    coloc_fit,
                    coloc_df,
                    coords_euclidean=["s_1", "s_2"],
                    avc=bad_avc,
                    uncond=True,
                    backend=backend,
                )

    identical_df = coloc_df.copy()
    identical_df["s_1"] = 0.0
    identical_df["s_2"] = 0.0
    for backend in backends:
        with pytest.raises(ValueError):
            scpc(
                coloc_fit,
                identical_df,
                coords_euclidean=["s_1", "s_2"],
                avc=0.5,
                uncond=True,
                backend=backend,
            )


def test_near_zero_t_statistic_probability() -> None:
    rho = 0.03
    expected_p = 1.0 - (2.0 / math.pi) * math.atan(
        1e-6 * math.sqrt((1.0 - rho) / (1.0 + rho))
    )
    assert expected_p == pytest.approx(0.99999938, abs=1e-8)

    om = np.diag([1.0 + rho, 1.0 - rho])
    assert get_rp(om, 1e-6) == pytest.approx(expected_p, abs=1e-12, rel=0.0)
    assert get_rp(np.array([[1.0, 0.0], [0.0, 0.0]]), 1.5) == pytest.approx(
        1.0, abs=1e-12
    )

    df = pd.DataFrame({"y": [1.0 + 1e-6, -1.0 + 1e-6], "s": [0.0, 1.0]})
    fit = smf.ols("y ~ 1", data=df).fit()
    res_fb = scpc(
        fit, df, coords_euclidean=["s"], avc=0.03, uncond=True, backend="fallback"
    )
    assert res_fb.q == 1
    assert res_fb.scpcstats[0, 2] == pytest.approx(1e-6, rel=1e-10, abs=1e-16)
    assert res_fb.scpcstats[0, 3] == pytest.approx(expected_p, abs=1e-12, rel=0.0)

    if rust_available():
        res_rust = scpc(
            fit, df, coords_euclidean=["s"], avc=0.03, uncond=True, backend="rust"
        )
        assert res_rust.q == 1
        assert res_rust.scpcstats[0, 3] == pytest.approx(expected_p, abs=1e-12, rel=0.0)
        npt.assert_allclose(res_rust.scpcstats, res_fb.scpcstats, atol=1e-12, rtol=0.0)

        rust_mod: Any = pytest.importorskip("spatial_spur_rust")
        direct = rust_mod.run_exact_scpc(
            np.array([[0.0], [1.0]]),
            np.array([1e-6]),
            np.array([[1.0 + 1e-6], [-1.0 + 1e-6]]),
            0.03,
            uncond=True,
            nthreads=1,
        )
        assert direct["q"] == 1
        assert direct["scpcstats"][0, 3] == pytest.approx(
            expected_p, abs=1e-12, rel=0.0
        )


def test_conditional_rank_crossing_matches_fallback() -> None:
    rng = np.random.default_rng(2026)
    p = np.sort(rng.uniform(size=50))
    s = rng.uniform(size=50)
    _ = rng.normal(0.0, 0.5, size=50)
    x = 1e-11 * (p - 0.5)
    y = 0.3 + p + rng.normal(0.0, 0.5, size=50)
    df = pd.DataFrame({"y": y, "x": x, "p": p, "s": s})
    fit = smf.ols("y ~ x", data=df).fit()

    res_fb = scpc(
        fit, df, coords_euclidean=["p", "s"], cvs=True, nthreads=1, backend="fallback"
    )
    assert res_fb.scpcstats[1, 3] == pytest.approx(0.06515033504085099, abs=1e-10)
    assert res_fb.scpcstats[1, 3] > 0.05

    if rust_available():
        res_rust = scpc(
            fit, df, coords_euclidean=["p", "s"], cvs=True, nthreads=1, backend="rust"
        )
        assert res_rust.q == res_fb.q
        npt.assert_allclose(
            res_rust.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-12
        )
        assert res_rust.scpccvs is not None
        assert res_fb.scpccvs is not None
        npt.assert_allclose(res_rust.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
def test_positive_outcome_rescaling_invariance() -> None:
    df = load_bench_df("bench_200.csv")
    fit_base = smf.ols("y ~ x1 + x2", data=df).fit()
    backends = ("fallback", "rust") if rust_available() else ("fallback",)

    for backend in backends:
        res_base = scpc(
            fit_base,
            df,
            coords_euclidean=["s_1", "s_2"],
            avc=0.03,
            uncond=False,
            cvs=True,
            backend=backend,
        )
        for y_scale in (1e-12, 1e6):
            df_scaled = df.copy()
            df_scaled["y"] = df_scaled["y"] * y_scale
            fit_scaled = smf.ols("y ~ x1 + x2", data=df_scaled).fit()
            res_scaled = scpc(
                fit_scaled,
                df_scaled,
                coords_euclidean=["s_1", "s_2"],
                avc=0.03,
                uncond=False,
                cvs=True,
                backend=backend,
            )
            assert res_scaled.q == res_base.q
            npt.assert_allclose(
                res_scaled.scpcstats[:, 0],
                res_base.scpcstats[:, 0] * y_scale,
                rtol=1e-10,
                atol=0.0,
            )
            npt.assert_allclose(
                res_scaled.scpcstats[:, 1],
                res_base.scpcstats[:, 1] * y_scale,
                rtol=1e-10,
                atol=0.0,
            )
            npt.assert_allclose(
                res_scaled.scpcstats[:, 2:4],
                res_base.scpcstats[:, 2:4],
                rtol=1e-10,
                atol=1e-10,
            )
            npt.assert_allclose(
                res_scaled.scpcstats[:, 4:6],
                res_base.scpcstats[:, 4:6] * y_scale,
                rtol=1e-10,
                atol=0.0,
            )
            assert res_scaled.scpccvs is not None
            assert res_base.scpccvs is not None
            npt.assert_allclose(
                res_scaled.scpccvs, res_base.scpccvs, rtol=1e-10, atol=1e-10
            )


def test_spatial_backend_env_var_and_argument_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = make_small_ols_setup()
    expected_default = "rust" if rust_available() else "fallback"

    monkeypatch.setenv("SPATIAL_BACKEND", "fallback")
    res_env_fb = scpc(model, data, coords_euclidean=("coord_x", "coord_y"), uncond=True)
    assert res_env_fb.backend == "fallback"

    # Explicit backend="auto" overrides SPATIAL_BACKEND="fallback"
    res_override_auto = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        uncond=True,
        backend="auto",
    )
    assert res_override_auto.backend == expected_default

    if rust_available():
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
    assert res_auto.backend == expected_default

    with pytest.raises(ValueError, match="backend"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"), backend="bad")

    monkeypatch.setenv("SPATIAL_BACKEND", "bad")
    with pytest.raises(ValueError, match="SPATIAL_BACKEND"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"))
    # Explicit backend="auto" also bypasses invalid SPATIAL_BACKEND
    res_explicit_auto = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        uncond=True,
        backend="auto",
    )
    assert res_explicit_auto.backend == expected_default
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)

    for bad_nthreads in (0, -2, True):
        with pytest.raises(ValueError, match="nthreads"):
            scpc(
                model,
                data,
                coords_euclidean=("coord_x", "coord_y"),
                nthreads=bad_nthreads,
            )


def test_absent_and_incompatible_rust_wheel_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = make_small_ols_setup()

    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)
    monkeypatch.setattr("scpc.core.load_rust_backend", lambda: None)

    with pytest.raises(RuntimeError, match="spatial_spur_rust"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"), backend="rust")

    res_auto = scpc(
        model, data, coords_euclidean=("coord_x", "coord_y"), backend="auto"
    )
    assert res_auto.backend == "fallback"

    # Incompatible API_VERSION must also be rejected cleanly
    monkeypatch.undo()
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)
    monkeypatch.setattr(
        "scpc.core.importlib.import_module",
        lambda name: SimpleNamespace(API_VERSION=999),
    )
    assert load_rust_backend() is None
    with pytest.raises(RuntimeError, match="spatial_spur_rust"):
        scpc(model, data, coords_euclidean=("coord_x", "coord_y"), backend="rust")
    res_bad_abi_auto = scpc(
        model, data, coords_euclidean=("coord_x", "coord_y"), backend="auto"
    )
    assert res_bad_abi_auto.backend == "fallback"


def test_unsupported_modes_raise_on_rust_and_fallback_on_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)
    has_rust = rust_available()

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
    if has_rust:
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
    if has_rust:
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

    # 3. Conditional AND unconditional IV
    iv_data = make_basic_iv_data(2001)
    iv_fit = pf.feols("y ~ w | x ~ z", data=iv_data)
    for uncond_flag in (False, True):
        if has_rust:
            with pytest.raises(NotImplementedError, match="instrumental"):
                scpc(
                    iv_fit,
                    iv_data,
                    coords_euclidean=("coord_x", "coord_y"),
                    uncond=uncond_flag,
                    backend="rust",
                )
        res_iv_auto = scpc(
            iv_fit,
            iv_data,
            coords_euclidean=("coord_x", "coord_y"),
            uncond=uncond_flag,
            backend="auto",
        )
        assert res_iv_auto.backend == "fallback"

    # 4. Conditional AND unconditional absorbed FE
    fe_data = pd.DataFrame(
        {
            "y": [1.0, 2.2, 2.9, 4.3, 4.9, 6.2],
            "x": [0.0, 1.0, 0.5, 1.5, 1.0, 2.0],
            "fe": [1, 1, 2, 2, 3, 3],
            "coord_x": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "coord_y": [0.0, 1.0, 0.5, 1.5, 1.0, 2.0],
        }
    )
    fe_fit = pf.feols("y ~ x | fe", data=fe_data)
    for uncond_flag in (False, True):
        if has_rust:
            with pytest.raises(NotImplementedError, match="absorbed fixed-effect"):
                scpc(
                    fe_fit,
                    fe_data,
                    coords_euclidean=("coord_x", "coord_y"),
                    uncond=uncond_flag,
                    backend="rust",
                )
        res_fe_auto = scpc(
            fe_fit,
            fe_data,
            coords_euclidean=("coord_x", "coord_y"),
            uncond=uncond_flag,
            backend="auto",
        )
        assert res_fe_auto.backend == "fallback"

    # 5. Weighted OLS (statsmodels WLS/GLS/GLM and pyfixest weighted OLS)
    w_data = pd.DataFrame(
        {
            "y": [1.0, 2.1, 2.9, 4.2, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "w": [1.0, 2.0, 1.5, 2.5, 1.0],
            "coord_x": [0.0, 1.0, 0.5, 1.5, 2.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 1.5],
        }
    )
    wls_fit = smf.wls("y ~ x", data=w_data, weights=w_data["w"]).fit()
    gls_fit = smf.gls("y ~ x", data=w_data).fit()
    glm_fit = smf.glm("y ~ x", data=w_data).fit()
    pf_w_fit = pf.feols("y ~ x", data=w_data, weights="w")
    for weighted_fit in (wls_fit, gls_fit, pf_w_fit):
        if has_rust:
            with pytest.raises(NotImplementedError, match="weighted"):
                scpc(
                    weighted_fit,
                    w_data,
                    coords_euclidean=("coord_x", "coord_y"),
                    uncond=True,
                    backend="rust",
                )
        res_w_auto = scpc(
            weighted_fit,
            w_data,
            coords_euclidean=("coord_x", "coord_y"),
            uncond=True,
            backend="auto",
        )
        assert res_w_auto.backend == "fallback"

    if has_rust:
        with pytest.raises(NotImplementedError, match="weighted"):
            scpc(
                glm_fit,
                w_data,
                coords_euclidean=("coord_x", "coord_y"),
                uncond=True,
                backend="rust",
            )


@pytest.mark.skipif(not HAS_MONOREPO_FIXTURES, reason="monorepo fixtures not available")
def test_1d_coords_ncoef_and_tied_eigenvalues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)

    df = load_bench_df("bench_200.csv")
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

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
    assert res_1d_fb.coef_names == ["Intercept", "x1"]

    if rust_available():
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
        assert res_1d_rust.backend == "rust"
        assert res_1d_rust.coef_names == ["Intercept", "x1"]
        npt.assert_allclose(
            res_1d_rust.scpcstats, res_1d_fb.scpcstats, atol=1e-10, rtol=0.0
        )
        assert res_1d_rust.scpccvs is not None
        assert res_1d_fb.scpccvs is not None
        npt.assert_allclose(
            res_1d_rust.scpccvs, res_1d_fb.scpccvs, atol=1e-10, rtol=0.0
        )

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
    assert res_tied_auto.tied is True

    if rust_available():
        res_tied_rust = scpc(
            tied_fit,
            tied_data,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=True,
            backend="rust",
        )
        assert res_tied_rust.backend == "rust"
        assert res_tied_rust.tied is True

    # First 20 rows of bench_200 also trigger tied leading eigenvalues
    df20 = df.iloc[:20].copy()
    fit20 = smf.ols("y ~ x1 + x2", data=df20).fit()
    res20_auto = scpc(
        fit20,
        df20,
        coords_euclidean=("s_1", "s_2"),
        avc=0.03,
        backend="auto",
    )
    assert res20_auto.backend == "fallback"
    assert res20_auto.tied is True
    if rust_available():
        res20_rust = scpc(
            fit20,
            df20,
            coords_euclidean=("s_1", "s_2"),
            avc=0.03,
            backend="rust",
        )
        assert res20_rust.backend == "rust"
        assert res20_rust.tied is True
