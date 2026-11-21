from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyfixest as pf
import pytest
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

from scpc import InvalidSampledCovarianceError, scpc
from scpc.core import load_rust_backend
from scpc.utils.matrix import lvech
from scpc.utils.spatial import (
    get_avc,
    get_distmat,
    get_rp,
)
from tests.utils import (
    make_basic_iv_data,
    make_clustered_fe_iv_data,
    make_clustered_iv_data,
    make_one_way_fe_iv_data,
    make_two_way_fe_iv_data,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "rust_core"


M1_CASES = [
    ("bench_200_euclid_cond", "bench_200.csv", False, False, True),
    ("bench_200_euclid_uncond", "bench_200.csv", False, True, False),
    ("bench_200_latlong_cond", "bench_200.csv", True, False, True),
    ("bench_1000_euclid_cond", "bench_1000.csv", False, False, True),
    ("bench_1000_euclid_uncond", "bench_1000.csv", False, True, False),
    ("bench_1000_latlong_cond", "bench_1000.csv", True, False, True),
]


def rust_available() -> bool:
    return load_rust_backend() is not None


def require_rust() -> Any:
    rust_mod = load_rust_backend()
    if rust_mod is None:
        pytest.skip()
    return rust_mod


def load_fixtures() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "m1_exact_scpc.json").read_text())


def load_bench_df(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / csv_name)


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


@pytest.mark.parametrize("method", ["exact", "approx"])
def test_native_model_preparation_uses_raw_fit(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    require_rust()
    fit, data = make_small_ols_setup()
    options: dict[str, Any] = {
        "coords_euclidean": ("coord_x", "coord_y"),
        "method": method,
        "cvs": True,
    }
    expected = scpc(fit, data, backend="fallback", **options)

    def unexpected_host_preparation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "The native model operation must prepare the fitted design itself."
        )

    monkeypatch.setattr(
        "scpc.core.get_conditional_projection_setup", unexpected_host_preparation
    )
    actual = scpc(fit, data, backend="rust", **options)
    assert actual.backend == "rust"
    npt.assert_allclose(actual.scpcstats, expected.scpcstats, rtol=1e-10, atol=1e-12)
    assert actual.scpccvs is not None and expected.scpccvs is not None
    npt.assert_allclose(actual.scpccvs, expected.scpccvs, rtol=1e-10, atol=1e-12)


def test_conditional_fit_checks_the_entire_selected_nuisance_design() -> None:
    data = load_bench_df("bench_200.csv")
    data["aliased"] = data["x1"] + 1e-12 * data["x2"]
    fit = smf.ols("y ~ x1 + aliased", data=data).fit()
    backends = ("fallback", "rust") if rust_available() else ("fallback",)
    for backend in backends:
        with pytest.raises(ValueError, match="rank deficient"):
            scpc(fit, data, coords_euclidean=("s_1", "s_2"), ncoef=1, backend=backend)


@pytest.mark.parametrize(
    "options, error", [({"ncoef": 0}, "ncoef"), ({"avc": math.nan}, "avc")]
)
def test_invalid_input_is_rejected_before_backend_selection(
    options: dict[str, Any], error: str
) -> None:
    fit, data = make_small_ols_setup()
    for backend in ("rust", "auto", "fallback"):
        with pytest.raises(ValueError, match=error):
            scpc(
                fit,
                data,
                coords_euclidean=("coord_x", "coord_y"),
                backend=backend,
                **options,
            )


def test_native_model_rejects_aggregated_scores_and_invalid_fe_labels() -> None:
    native = require_rust()
    coords = np.arange(3.0)[:, None]
    ids = np.repeat(np.arange(3), 2)
    coef = np.array([1.0])
    bread = np.ones((1, 1))
    with pytest.raises(ValueError, match="one row per model observation"):
        native.run_model_scpc(coords, coef, bread, coords, 0.1, cluster_id=ids)
    design = np.ones((6, 1))
    with pytest.raises(TypeError, match="fe_id must have an integer dtype"):
        native.run_model_scpc(
            coords,
            coef,
            bread,
            design,
            0.1,
            design=design,
            cluster_id=ids,
            fe_id=ids.astype(float),
        )


def test_cluster_coordinates_must_match_each_observation_on_the_fitted_sample() -> None:
    fit, data = make_small_ols_setup()
    data["site"] = [0, 0, 0, 1, 1]
    data["coord_x"] = [0.0, 1.0, -1.0, 2.0, 2.0]
    for backend in ("fallback", "rust") if rust_available() else ("fallback",):
        with pytest.raises(ValueError, match="Coordinates vary within clusters"):
            scpc(
                fit, data, coords_euclidean=["coord_x"], cluster="site", backend=backend
            )


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_active_cluster_labels_must_be_present(missing: object) -> None:
    fit, data = make_small_ols_setup()
    data["site"] = ["a", "a", missing, "b", "b"]
    with pytest.raises(ValueError, match="Cluster labels must be present"):
        scpc(
            fit, data, coords_euclidean=["coord_x"], cluster="site", backend="fallback"
        )


def test_cluster_validation_uses_only_rows_kept_by_the_fit() -> None:
    data = load_bench_df("bench_200.csv")
    dropped = data.iloc[[0]].assign(y=np.nan, cluster_id=np.nan, s_1_clustered=np.nan)
    data = pd.concat([data, dropped], ignore_index=True)
    fit = smf.ols("y ~ x1 + x2", data=data).fit()
    result = scpc(
        fit,
        data,
        cluster="cluster_id",
        coords_euclidean=["s_1_clustered", "s_2_clustered"],
        backend="fallback",
    )
    assert np.isfinite(result.scpcstats).all()


def make_direct_ffi_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
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
    scores = np.array(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.2, 0.1],
            [0.5, -0.3],
            [-0.4, 0.2],
        ],
        dtype=np.float64,
    )
    design = np.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.1],
            [1.0, 0.6],
            [1.0, 1.2],
        ],
        dtype=np.float64,
    )
    bread = np.array([[0.4, 0.06], [0.06, 0.25]])
    return coords, coef, bread, scores, design


def test_spatial_spur_rust_api_version_fortran_layout_and_validation() -> None:
    rust_mod: Any = require_rust()
    assert rust_mod.API_VERSION == 7

    coords, coef, bread, scores, design = make_direct_ffi_inputs()

    res_c = rust_mod.run_model_scpc(
        coords,
        coef,
        bread,
        scores,
        0.1,
        latlong=False,
        cvs=True,
        design=design,
        nthreads=1,
    )
    res_f = rust_mod.run_model_scpc(
        np.asfortranarray(coords),
        coef,
        np.asfortranarray(bread),
        np.asfortranarray(scores),
        0.1,
        latlong=False,
        cvs=True,
        design=np.asfortranarray(design),
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
        rust_mod.run_model_scpc(bad_coords, coef, bread, scores, 0.1)

    with pytest.raises(ValueError, match="avc"):
        rust_mod.run_model_scpc(coords, coef, bread, scores, 0.001)

    with pytest.raises(ValueError, match="scores"):
        rust_mod.run_model_scpc(coords, coef, bread, scores[:3, :], 0.1)

    with pytest.raises(ValueError, match="conditional model matrices"):
        rust_mod.run_model_scpc(coords, coef, bread, scores, 0.1, instruments=design)

    with pytest.raises(ValueError, match="design"):
        rust_mod.run_model_scpc(coords, coef, bread, scores, 0.1, design=design[:3])

    with pytest.raises(ValueError, match="2 columns"):
        rust_mod.run_model_scpc(coords[:, :1], coef, bread, scores, 0.1, latlong=True)

    with pytest.raises(ValueError, match="nthreads"):
        rust_mod.run_model_scpc(coords, coef, bread, scores, 0.1, nthreads=0)

    with pytest.raises(TypeError, match="nthreads"):
        rust_mod.run_model_scpc(coords, coef, bread, scores, 0.1, nthreads=True)


def test_rust_ffi_arbitrary_strides_unaligned_and_sliced_views() -> None:
    rust_mod: Any = require_rust()
    coords, coef, bread, scores, design = make_direct_ffi_inputs()

    ref = rust_mod.run_model_scpc(
        coords,
        coef,
        bread,
        scores,
        0.1,
        latlong=False,
        cvs=True,
        design=design,
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
    res_strided = rust_mod.run_model_scpc(
        make_byte_strided_2d(coords, (24, 9)),
        coef,
        make_byte_strided_2d(bread, (24, 9)),
        make_byte_strided_2d(scores, (24, 9)),
        0.1,
        cvs=True,
        design=make_byte_strided_2d(design, (24, 9)),
        nthreads=1,
    )
    npt.assert_allclose(
        res_strided["scpcstats"], ref["scpcstats"], atol=1e-14, rtol=0.0
    )
    assert res_strided["c0"] == pytest.approx(ref["c0"], rel=1e-14, abs=1e-14)

    # Unaligned C-contiguous views (offset by 1 byte)
    res_unaligned = rust_mod.run_model_scpc(
        make_unaligned_c(coords),
        make_unaligned_c(coef),
        make_unaligned_c(bread),
        make_unaligned_c(scores),
        0.1,
        cvs=True,
        design=make_unaligned_c(design),
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
    bread_neg = bread[::-1, ::-1].copy()[::-1, ::-1]
    scores_neg = scores[::-1, ::-1].copy()[::-1, ::-1]
    design_neg = design[::-1, ::-1].copy()[::-1, ::-1]
    assert coords_neg.strides[0] < 0 and coords_neg.strides[1] < 0

    for coef_view in (coef_step2, coef_rev):
        res_neg = rust_mod.run_model_scpc(
            coords_neg,
            coef_view,
            bread_neg,
            scores_neg,
            0.1,
            cvs=True,
            design=design_neg,
            nthreads=1,
        )
        npt.assert_allclose(
            res_neg["scpcstats"], ref["scpcstats"], atol=1e-14, rtol=0.0
        )
        assert res_neg["c0"] == pytest.approx(ref["c0"], rel=1e-14, abs=1e-14)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork not available")
def test_rust_fork_safety_after_warming_threads() -> None:
    rust_mod: Any = require_rust()
    coords, coef, bread, scores, design = make_direct_ffi_inputs()

    # Initialize the native pool in the parent before forking.
    rust_mod.run_model_scpc(
        coords,
        coef,
        bread,
        scores,
        0.1,
        cvs=True,
        design=design,
        nthreads=2,
    )

    pid = os.fork()
    if pid == 0:
        try:
            rust_mod.run_model_scpc(
                coords,
                coef,
                bread,
                scores,
                0.1,
                cvs=True,
                design=design,
                nthreads=1,
            )
        except RuntimeError as err:
            if "spawn" in str(err) and "forkserver" in str(err):
                os._exit(0)
            os._exit(1)
        except BaseException:
            os._exit(2)
        os._exit(3)

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
            "forked child deadlocked or timed out in run_model_scpc(nthreads=1)"
        )

    assert os.WIFEXITED(status)
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork not available")
def test_structural_invalid_native_call_does_not_poison_fork() -> None:
    require_rust()
    # Each subprocess starts with fresh native state, independent of test order.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from multiprocessing import get_context
from tests.test_rust_m1 import make_direct_ffi_inputs, require_rust

native = require_rust()
coords, coef, bread, scores, _ = make_direct_ffi_inputs()
try:
    native.run_model_scpc(coords, coef, bread, scores[:-1], 0.1)
except ValueError:
    pass
else:
    raise AssertionError("Invalid score shape was accepted")

def run_valid():
    native.run_model_scpc(coords, coef, bread, scores, 0.1, nthreads=1)

child = get_context("fork").Process(target=run_valid)
child.start()
child.join(8)
if child.is_alive():
    child.kill()
    child.join()
    raise AssertionError("Child timed out after an invalid parent call")
assert child.exitcode == 0, child.exitcode
""",
        ],
        check=True,
        timeout=30,
    )


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


@pytest.mark.parametrize("latlong", [False, True])
@pytest.mark.parametrize(("uncond", "cvs"), [(True, True), (False, False)])
def test_m1_additional_uncond_and_cvs_combinations(
    latlong: bool, uncond: bool, cvs: bool
) -> None:
    require_rust()
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


def test_rust_nthreads_1_runs_rust_and_matches_nthreads_4_and_fallback() -> None:
    require_rust()
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

        rust_mod: Any = require_rust()
        direct = rust_mod.run_model_scpc(
            np.array([[0.0], [1.0]]),
            np.array([1e-6]),
            np.array([[0.5]]),
            np.array([[1.0], [-1.0]]),
            0.03,
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
    assert (
        res_bad_abi_auto.fallback_reason == "missing_or_incompatible_native_extension"
    )


def test_unsupported_modes_raise_on_rust_and_fallback_on_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPATIAL_BACKEND", raising=False)
    has_rust = rust_available()

    # 1. Clustered inference is now supported on Rust exact OLS
    cl_data = pd.DataFrame(
        {
            "y": [2.1, 2.3, 0.9, 1.2, 1.6, 1.8, 3.0, 3.2, 0.4, 0.6],
            "x": [1.0, 1.2, 0.0, 0.2, 0.5, 0.7, 1.5, 1.7, -0.3, -0.1],
            "coord_x": [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 3.5, 3.5, 5.0, 5.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 2.5, 2.5],
            "cluster": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        }
    )
    cl_model = smf.ols("y ~ x", data=cl_data).fit()
    res_cl_fb = scpc(
        cl_model,
        cl_data,
        coords_euclidean=("coord_x", "coord_y"),
        cluster="cluster",
        cvs=True,
        backend="fallback",
    )
    res_cl_auto = scpc(
        cl_model,
        cl_data,
        coords_euclidean=("coord_x", "coord_y"),
        cluster="cluster",
        cvs=True,
        backend="auto",
    )
    if has_rust:
        res_cl_rust = scpc(
            cl_model,
            cl_data,
            coords_euclidean=("coord_x", "coord_y"),
            cluster="cluster",
            cvs=True,
            backend="rust",
        )
        assert res_cl_auto.backend == "rust"
        assert res_cl_rust.backend == "rust"
        assert res_cl_rust.q == res_cl_fb.q
        npt.assert_allclose(
            res_cl_rust.scpcstats, res_cl_fb.scpcstats, rtol=1e-10, atol=1e-10
        )
        assert res_cl_rust.scpccvs is not None and res_cl_fb.scpccvs is not None
        npt.assert_allclose(
            res_cl_rust.scpccvs, res_cl_fb.scpccvs, rtol=1e-10, atol=1e-10
        )
    else:
        assert res_cl_auto.backend == "fallback"

    # 2. Approximate large-n branch now runs natively on Rust and matches fallback
    model, data = make_small_ols_setup()
    res_approx_fb = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        method="approx",
        cvs=True,
        backend="fallback",
    )
    res_approx_auto = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        method="approx",
        cvs=True,
        backend="auto",
    )
    assert res_approx_fb.method == "approx"
    assert res_approx_auto.method == "approx"
    if has_rust:
        res_approx_ru = scpc(
            model,
            data,
            coords_euclidean=("coord_x", "coord_y"),
            method="approx",
            cvs=True,
            backend="rust",
        )
        assert res_approx_auto.backend == "rust"
        assert res_approx_ru.backend == "rust"
        assert res_approx_ru.method == "approx"
        assert res_approx_ru.q == res_approx_fb.q
        assert res_approx_ru.c0 == pytest.approx(res_approx_fb.c0, rel=1e-10, abs=1e-12)
        assert res_approx_ru.cv == pytest.approx(res_approx_fb.cv, rel=1e-10, abs=1e-12)
        npt.assert_allclose(
            res_approx_ru.scpcstats, res_approx_fb.scpcstats, rtol=1e-10, atol=1e-10
        )
        assert res_approx_ru.scpccvs is not None and res_approx_fb.scpccvs is not None
        npt.assert_allclose(
            res_approx_ru.scpccvs, res_approx_fb.scpccvs, rtol=1e-10, atol=1e-10
        )
    else:
        assert res_approx_auto.backend == "fallback"

    # 3. Conditional AND unconditional non-FE IV now run on Rust
    iv_data = make_basic_iv_data(2001)
    iv_fit = pf.feols("y ~ w | x ~ z", data=iv_data)
    for uncond_flag in (False, True):
        res_iv_fb = scpc(
            iv_fit,
            iv_data,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=uncond_flag,
            cvs=True,
            backend="fallback",
        )
        res_iv_auto = scpc(
            iv_fit,
            iv_data,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=uncond_flag,
            cvs=True,
            backend="auto",
        )
        if has_rust:
            assert res_iv_auto.backend == "rust"
            assert res_iv_auto.q == res_iv_fb.q
            npt.assert_allclose(
                res_iv_auto.scpcstats, res_iv_fb.scpcstats, rtol=1e-10, atol=1e-10
            )
        else:
            assert res_iv_auto.backend == "fallback"

    # 4. 1-way FE OLS runs on Rust; 2-way FE IV falls back conditionally and runs unconditionally
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
        res_fe_fb = scpc(
            fe_fit,
            fe_data,
            coords_euclidean=("coord_x", "coord_y"),
            uncond=uncond_flag,
            cvs=True,
            backend="fallback",
        )
        res_fe_auto = scpc(
            fe_fit,
            fe_data,
            coords_euclidean=("coord_x", "coord_y"),
            uncond=uncond_flag,
            cvs=True,
            backend="auto",
        )
        if has_rust:
            assert res_fe_auto.backend == "rust"
            npt.assert_allclose(
                res_fe_auto.scpcstats, res_fe_fb.scpcstats, rtol=1e-10, atol=1e-10
            )
        else:
            assert res_fe_auto.backend == "fallback"

    iv_2way_data = make_two_way_fe_iv_data(2005)
    iv_2way_fit = pf.feols("y ~ w | fe1 + fe2 | x ~ z", data=iv_2way_data)
    if has_rust:
        with pytest.raises(
            NotImplementedError, match="multi-way absorbed fixed-effect"
        ):
            scpc(
                iv_2way_fit,
                iv_2way_data,
                coords_euclidean=("coord_x", "coord_y"),
                avc=0.1,
                uncond=False,
                backend="rust",
            )
    res_2way_auto = scpc(
        iv_2way_fit,
        iv_2way_data,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        uncond=False,
        backend="auto",
    )
    assert res_2way_auto.backend == "fallback"
    if has_rust:
        res_2way_uncond = scpc(
            iv_2way_fit,
            iv_2way_data,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=True,
            backend="rust",
        )
        res_2way_uncond_fb = scpc(
            iv_2way_fit,
            iv_2way_data,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=True,
            backend="fallback",
        )
        assert res_2way_uncond.backend == "rust"
        npt.assert_allclose(
            res_2way_uncond.scpcstats,
            res_2way_uncond_fb.scpcstats,
            rtol=1e-10,
            atol=1e-10,
        )


def test_unsupported_models_fail_for_all_backends() -> None:
    # Weighted and nonlinear models (statsmodels WLS/GLS/GLM and pyfixest weighted OLS)
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
    for fit in (wls_fit, gls_fit, glm_fit, pf_w_fit):
        for backend in ("auto", "fallback", "rust"):
            with pytest.raises(NotImplementedError, match="weighted"):
                scpc(
                    fit,
                    w_data,
                    coords_euclidean=("coord_x", "coord_y"),
                    uncond=True,
                    backend=backend,
                )


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


@pytest.mark.parametrize("engine", ["statsmodels", "pyfixest"])
@pytest.mark.parametrize(
    ("formula", "cutoff"), [("y ~ x", 17.380977), ("y ~ x - 1", 9.902023)]
)
def test_clustered_conditional_ols_cutoffs_and_bench_200_parity(
    engine: str, formula: str, cutoff: float
) -> None:
    sites = pd.DataFrame(
        {"s": [0.0, 1.0, 2.0, 4.0, 6.0], "x0": [-1.0, -4.0, 2.0, 2.0, 1.0]}
    )
    rows = []
    for _, r in sites.iterrows():
        for delta in (-0.25, 0.25):
            x_val = r["x0"] + delta
            rows.append({"s": r["s"], "x": x_val, "y": x_val + float(r["s"] == 0.0)})
    cl_df = pd.DataFrame(rows)
    fit = (
        smf.ols(formula, data=cl_df).fit()
        if engine == "statsmodels"
        else pf.feols(formula, data=cl_df)
    )

    backends = ("fallback", "rust") if rust_available() else ("fallback",)
    results = {}
    for bk in backends:
        res = scpc(
            fit,
            cl_df,
            coords_euclidean=("s",),
            cluster="s",
            avc=0.5,
            uncond=False,
            cvs=True,
            backend=bk,
        )
        assert res.backend == bk
        assert res.scpccvs is not None
        x_idx = res.coef_names.index("x")
        assert abs(float(res.scpccvs[x_idx, 2]) - cutoff) < 0.002
        results[bk] = res

    if "rust" in results:
        npt.assert_allclose(
            results["rust"].scpcstats,
            results["fallback"].scpcstats,
            rtol=1e-10,
            atol=1e-10,
        )
        assert (
            results["rust"].scpccvs is not None
            and results["fallback"].scpccvs is not None
        )
        npt.assert_allclose(
            results["rust"].scpccvs,
            results["fallback"].scpccvs,
            rtol=1e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize("uncond", [False, True])
@pytest.mark.parametrize("latlong", [False, True])
def test_clustered_ols_bench_200_rust_vs_fallback_parity(
    uncond: bool, latlong: bool
) -> None:
    df = load_bench_df("bench_200.csv")
    coord_kwargs: dict[str, Any] = (
        {"lon": "s_1_clustered", "lat": "s_2_clustered"}
        if latlong
        else {"coords_euclidean": ("s_1_clustered", "s_2_clustered")}
    )
    for fit in (
        smf.ols("y ~ x1 + x2", data=df).fit(),
        pf.feols("y ~ x1 + x2", data=df),
    ):
        res_fb = scpc(
            fit,
            df,
            cluster="cluster_id",
            avc=0.03,
            uncond=uncond,
            cvs=True,
            backend="fallback",
            **coord_kwargs,
        )
        assert res_fb.backend == "fallback"
        if rust_available():
            res_ru = scpc(
                fit,
                df,
                cluster="cluster_id",
                avc=0.03,
                uncond=uncond,
                cvs=True,
                backend="rust",
                **coord_kwargs,
            )
            assert res_ru.backend == "rust"
            assert res_ru.q == res_fb.q
            assert res_ru.c0 == pytest.approx(res_fb.c0, rel=1e-10, abs=1e-12)
            assert res_ru.cv == pytest.approx(res_fb.cv, rel=1e-10, abs=1e-12)
            npt.assert_allclose(
                res_ru.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-10
            )
            assert res_ru.scpccvs is not None and res_fb.scpccvs is not None
            npt.assert_allclose(res_ru.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("case_name", "formula", "cluster"),
    [
        ("iv_basic", "y ~ w | x ~ z", None),
        ("iv_1way_fe", "y ~ w | fe | x ~ z", None),
        ("iv_clustered", "y ~ w | x ~ z", "cl"),
        ("iv_1way_fe_clustered", "y ~ w | fe | x ~ z", "cl"),
    ],
)
@pytest.mark.parametrize("uncond", [False, True])
def test_iv_and_one_way_fe_rust_vs_fallback_parity(
    case_name: str, formula: str, cluster: str | None, uncond: bool
) -> None:
    if case_name == "iv_basic":
        df = make_basic_iv_data(2001)
    elif case_name == "iv_1way_fe":
        df = make_one_way_fe_iv_data(2002)
    elif case_name == "iv_clustered":
        df = make_clustered_iv_data(2003)
    else:
        df = make_clustered_fe_iv_data(2004)

    fit = pf.feols(formula, data=df)
    res_fb = scpc(
        fit,
        df,
        coords_euclidean=("coord_x", "coord_y"),
        cluster=cluster,
        avc=0.1,
        uncond=uncond,
        cvs=True,
        backend="fallback",
    )
    assert res_fb.backend == "fallback"
    if rust_available():
        res_ru = scpc(
            fit,
            df,
            coords_euclidean=("coord_x", "coord_y"),
            cluster=cluster,
            avc=0.1,
            uncond=uncond,
            cvs=True,
            backend="rust",
        )
        assert res_ru.backend == "rust"
        assert res_ru.q == res_fb.q
        assert res_ru.c0 == pytest.approx(res_fb.c0, rel=1e-10, abs=1e-12)
        assert res_ru.cv == pytest.approx(res_fb.cv, rel=1e-10, abs=1e-12)
        npt.assert_allclose(res_ru.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-10)
        assert res_ru.scpccvs is not None and res_fb.scpccvs is not None
        npt.assert_allclose(res_ru.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("latlong", [False, True])
@pytest.mark.parametrize("uncond", [False, True])
def test_approx_scpc_bench_200_ols_parity_and_threads(
    latlong: bool, uncond: bool
) -> None:
    df = load_bench_df("bench_200.csv")
    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    coord_kwargs: dict[str, Any] = (
        {"lon": "s_1", "lat": "s_2"}
        if latlong
        else {"coords_euclidean": ("s_1", "s_2")}
    )

    res_fb = scpc(
        fit,
        df,
        **coord_kwargs,
        avc=0.05,
        method="approx",
        large_n_seed=7,
        uncond=uncond,
        cvs=True,
        backend="fallback",
    )
    assert res_fb.backend == "fallback"
    assert res_fb.method == "approx"
    assert res_fb.large_n_seed == 7
    if rust_available():
        res_1 = scpc(
            fit,
            df,
            **coord_kwargs,
            avc=0.05,
            method="approx",
            large_n_seed=7,
            uncond=uncond,
            cvs=True,
            backend="rust",
            nthreads=1,
        )
        res_2 = scpc(
            fit,
            df,
            **coord_kwargs,
            avc=0.05,
            method="approx",
            large_n_seed=7,
            uncond=uncond,
            cvs=True,
            backend="rust",
            nthreads=2,
        )
        assert res_1.backend == "rust"
        assert res_1.method == "approx"
        assert res_1.q == res_fb.q
        assert res_1.c0 == pytest.approx(res_fb.c0, rel=1e-10, abs=1e-12)
        assert res_1.cv == pytest.approx(res_fb.cv, rel=1e-10, abs=1e-12)
        npt.assert_allclose(res_1.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-10)
        assert (
            res_1.scpccvs is not None
            and res_2.scpccvs is not None
            and res_fb.scpccvs is not None
        )
        npt.assert_allclose(res_1.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-10)
        npt.assert_allclose(res_1.scpcstats, res_2.scpcstats, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(res_1.scpccvs, res_2.scpccvs, rtol=1e-13, atol=1e-13)


def test_nystrom_conditional_inference_is_invariant_to_coefficient_order_and_subset() -> (
    None
):
    data = load_bench_df("bench_200.csv").assign(constant=1.0)
    full = OLS(data.y, data[["constant", "x1", "x2"]]).fit()
    reordered = OLS(data.y, data[["x2", "constant", "x1"]]).fit()
    options: dict[str, Any] = dict(
        coords_euclidean=("s_1", "s_2"),
        method="approx",
        cvs=True,
        nthreads=1,
        large_n_seed=1,
    )
    for backend in ("fallback", "rust") if rust_available() else ("fallback",):
        original = scpc(full, data, backend=backend, **options)
        subset = scpc(reordered, data, ncoef=2, backend=backend, **options)
        assert subset.coef_names == ["x2", "constant"]
        order = [original.coef_names.index(name) for name in subset.coef_names]
        assert original.scpccvs is not None and subset.scpccvs is not None
        npt.assert_allclose(
            subset.scpccvs, original.scpccvs[order], rtol=1e-10, atol=1e-10
        )
        npt.assert_allclose(
            subset.scpcstats, original.scpcstats[order], rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize(
    ("formula", "cluster"),
    [
        ("y ~ w | x ~ z", None),
        ("y ~ w | fe | x ~ z", None),
        ("y ~ w | x ~ z", "cl"),
        ("y ~ w | fe | x ~ z", "cl"),
    ],
)
@pytest.mark.parametrize("uncond", [False, True])
def test_approx_scpc_pyfixest_iv_and_cluster_parity(
    formula: str, cluster: str | None, uncond: bool
) -> None:
    pf = pytest.importorskip("pyfixest")
    df = make_clustered_fe_iv_data(2004)
    fit = pf.feols(formula, data=df)

    res_fb = scpc(
        fit,
        df,
        coords_euclidean=("coord_x", "coord_y"),
        cluster=cluster,
        avc=0.1,
        method="approx",
        large_n_seed=3,
        uncond=uncond,
        cvs=True,
        backend="fallback",
    )
    assert res_fb.backend == "fallback"
    assert res_fb.method == "approx"
    if rust_available():
        res_ru = scpc(
            fit,
            df,
            coords_euclidean=("coord_x", "coord_y"),
            cluster=cluster,
            avc=0.1,
            method="approx",
            large_n_seed=3,
            uncond=uncond,
            cvs=True,
            backend="rust",
        )
        assert res_ru.backend == "rust"
        assert res_ru.method == "approx"
        assert res_ru.q == res_fb.q
        assert res_ru.c0 == pytest.approx(res_fb.c0, rel=1e-10, abs=1e-12)
        assert res_ru.cv == pytest.approx(res_fb.cv, rel=1e-10, abs=1e-12)
        npt.assert_allclose(res_ru.scpcstats, res_fb.scpcstats, rtol=1e-10, atol=1e-10)
        assert res_ru.scpccvs is not None and res_fb.scpccvs is not None
        npt.assert_allclose(res_ru.scpccvs, res_fb.scpccvs, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not rust_available(), reason="spatial_spur_rust not installed")
def test_approx_threshold_override_and_tied_dispatch() -> None:
    # n = 4500 triggers approx under auto, while explicit exact overrides the threshold.
    rng = np.random.default_rng(2026)
    n_big = 4500
    s1 = rng.uniform(0.0, 1.0, size=n_big)
    s2 = rng.uniform(0.0, 1.0, size=n_big)
    x = rng.normal(size=n_big)
    y = 0.5 + 1.2 * x + rng.normal(scale=0.5, size=n_big)
    df_big = pd.DataFrame({"y": y, "x": x, "s_1": s1, "s_2": s2})
    fit_big = smf.ols("y ~ x", data=df_big).fit()

    res_auto = scpc(
        fit_big,
        df_big,
        coords_euclidean=("s_1", "s_2"),
        method="auto",
        backend="rust",
    )
    assert res_auto.backend == "rust"
    assert res_auto.method == "approx"

    res_exact = scpc(
        fit_big,
        df_big,
        coords_euclidean=("s_1", "s_2"),
        method="exact",
        backend="rust",
    )
    assert res_exact.backend == "rust"
    assert res_exact.method == "exact"

    # A tied native basis may use the fallback, whose sampled grid still has
    # to satisfy the same covariance validity checks.
    n_tie = 20
    s1_tie = np.concatenate([np.zeros(10), np.ones(10)])
    s2_tie = np.zeros(n_tie)
    x_tie = rng.normal(size=n_tie)
    y_tie = 1.0 + x_tie + rng.normal(size=n_tie)
    df_tie = pd.DataFrame({"y": y_tie, "x": x_tie, "s1": s1_tie, "s2": s2_tie})
    fit_tie = smf.ols("y ~ x", data=df_tie).fit()

    with pytest.raises(require_rust().TiedEigenspaceError, match="tied eigenspace"):
        scpc(
            fit_tie,
            df_tie,
            coords_euclidean=("s1", "s2"),
            avc=0.5,
            method="approx",
            backend="rust",
        )
    with pytest.raises(ValueError, match="degenerate principal direction"):
        scpc(
            fit_tie,
            df_tie,
            coords_euclidean=("s1", "s2"),
            avc=0.5,
            method="approx",
            backend="auto",
        )


def test_invalid_sampled_covariance_stops_public_approximation() -> None:
    i = np.arange(8, dtype=float)
    data = pd.DataFrame({"x": i, "y": np.cos(i), "sx": i, "sy": np.sin(i)})
    fit = smf.ols("y ~ x", data=data).fit()
    backends = (
        ("fallback", "rust", "auto") if rust_available() else ("fallback", "auto")
    )
    for backend in backends:
        common: dict[str, Any] = dict(
            coords_euclidean=("sx", "sy"),
            avc=0.95,
            uncond=True,
            nthreads=1,
            backend=backend,
        )
        with pytest.raises(InvalidSampledCovarianceError, match="method='exact'"):
            scpc(fit, data, method="approx", **common)
        exact = scpc(fit, data, method="exact", **common)
        assert np.isfinite(exact.scpcstats).all()


def test_native_tied_approximation_can_use_valid_fallback() -> None:
    rust_mod = require_rust()
    data = pd.DataFrame({"sx": [0, 0, 1, 1], "sy": [0, 1, 1, 0], "y": [0, 1, 3, 5]})
    fit = smf.ols("y ~ 1", data=data).fit()
    common: dict[str, Any] = dict(
        coords_euclidean=("sx", "sy"),
        avc=0.2,
        uncond=True,
        nthreads=1,
        method="approx",
    )
    with pytest.raises(rust_mod.TiedEigenspaceError):
        scpc(fit, data, backend="rust", **common)
    result = scpc(fit, data, backend="auto", **common)
    assert result.backend == "fallback" and result.tied
    assert result.fallback_reason == "tied_eigenspace"
