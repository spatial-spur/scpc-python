from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyfixest as pf
import pytest

from scpc import scpc
from scpc.types import SCPCResult
from tests.config import ATOL, RTOL
from tests.utils import (
    R,
    execute_r_code,
    make_basic_iv_data,
    make_clustered_fe_iv_data,
    make_clustered_iv_data,
    make_one_way_fe_iv_data,
    make_two_way_fe_iv_data,
    reorder_r_rows_to_py,
)


def _run_r_scpc_iv(
    formula: str,
    data,
    *,
    cluster: str | None = None,
):
    fixture_by_case: dict[tuple[str, str | None], str] = {
        ("y ~ w | x ~ z", None): "test_scpc_matches_r_for_non_fe_iv_exact",
        ("y ~ w | fe | x ~ z", None): "test_scpc_matches_r_for_one_way_fe_iv_exact",
        (
            "y ~ w | fe1 + fe2 | x ~ z",
            None,
        ): "test_scpc_matches_r_for_two_way_fe_iv_exact",
        ("y ~ w | x ~ z", "cl"): "test_scpc_matches_r_for_clustered_iv_exact",
        ("y ~ w | fe | x ~ z", "cl"): "test_scpc_matches_r_for_clustered_fe_iv_exact",
    }
    return execute_r_code(
        """
        suppressPackageStartupMessages(library(fixest))
        old_threads <- fixest::getFixest_nthreads()
        fixest::setFixest_nthreads(1L)
        on.exit(fixest::setFixest_nthreads(old_threads), add = TRUE)

        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x),
          w = unlist(payload$data$w),
          z = unlist(payload$data$z),
          coord_x = unlist(payload$data$coord_x),
          coord_y = unlist(payload$data$coord_y),
          check.names = FALSE
        )
        if (!is.null(payload$data$fe)) {
          d$fe <- unlist(payload$data$fe)
        }
        if (!is.null(payload$data$fe1)) {
          d$fe1 <- unlist(payload$data$fe1)
        }
        if (!is.null(payload$data$fe2)) {
          d$fe2 <- unlist(payload$data$fe2)
        }
        if (!is.null(payload$data$cl)) {
          d$cl <- unlist(payload$data$cl)
        }
        fit <- fixest::feols(stats::as.formula(payload$formula), data = d)
        out <- scpc(
          fit,
          data = d,
          coords_euclidean = c("coord_x", "coord_y"),
          cluster = payload$cluster,
          avc = 0.1,
          uncond = FALSE,
          cvs = TRUE
        )
        result <- list(
          term_names = rownames(out$scpcstats),
          scpcstats = unname(split(out$scpcstats, row(out$scpcstats))),
          scpccvs = if (is.null(out$scpccvs)) NULL else unname(split(out$scpccvs, row(out$scpccvs))),
          q = out$q,
          cv = out$cv,
          c0 = out$c0,
          avc = out$avc
        )
        """,
        payload={
            "formula": formula,
            "data": data.to_dict(orient="list"),
            "cluster": cluster,
        },
        fixture=fixture_by_case[(formula, cluster)],
    )


def _assert_pyfixest_scpc_parity(
    formula: str,
    data,
    *,
    cluster: str | None = None,
    method: str = "exact",
    large_n_seed: int = 1,
) -> None:
    fit = pf.feols(formula, data=data)
    py_result = scpc(
        fit,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        cluster=cluster,
        avc=0.1,
        method=method,
        large_n_seed=large_n_seed,
        uncond=False,
        cvs=True,
    )

    assert isinstance(py_result, SCPCResult)
    assert np.isfinite(py_result.scpcstats).all()
    assert py_result.scpccvs is not None
    assert np.isfinite(py_result.scpccvs).all()
    assert py_result.method == method
    assert py_result.large_n_seed == large_n_seed

    r_value = _run_r_scpc_iv(
        formula,
        data,
        cluster=cluster,
    )
    py_names = [str(name) for name in fit._coefnames]
    r_stats = reorder_r_rows_to_py(
        py_names,
        [str(name) for name in r_value["term_names"]],
        np.array(r_value["scpcstats"]),
    )
    r_cvs = reorder_r_rows_to_py(
        py_names,
        [str(name) for name in r_value["term_names"]],
        np.array(r_value["scpccvs"]),
    )

    npt.assert_allclose(py_result.scpcstats, r_stats, atol=ATOL, rtol=RTOL)
    assert py_result.scpccvs is not None
    npt.assert_allclose(py_result.scpccvs, r_cvs, atol=ATOL, rtol=RTOL)
    assert py_result.q == r_value["q"]
    assert py_result.cv == pytest.approx(r_value["cv"], abs=ATOL, rel=RTOL)
    assert py_result.c0 == pytest.approx(r_value["c0"], abs=ATOL, rel=RTOL)
    assert py_result.avc == pytest.approx(r_value["avc"], abs=ATOL, rel=RTOL)


def test_scpc_matches_r_for_non_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity("y ~ w | x ~ z", make_basic_iv_data(2001))


def test_scpc_matches_r_for_one_way_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | fe | x ~ z",
        make_one_way_fe_iv_data(2002),
    )


def test_scpc_matches_r_for_two_way_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | fe1 + fe2 | x ~ z",
        make_two_way_fe_iv_data(2005),
    )


def test_scpc_matches_r_for_clustered_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | x ~ z",
        make_clustered_iv_data(2003),
        cluster="cl",
    )


def test_scpc_matches_r_for_clustered_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | fe | x ~ z",
        make_clustered_fe_iv_data(2004),
        cluster="cl",
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_scpc_matches_r_for_non_fe_iv_approx() -> None:
    fit = pf.feols("y ~ w | x ~ z", data=make_basic_iv_data(2001))
    result = scpc(
        fit,
        make_basic_iv_data(2001),
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        method="approx",
        large_n_seed=17,
        uncond=False,
        cvs=True,
    )

    assert isinstance(result, SCPCResult)
    assert np.isfinite(result.scpcstats).all()
    assert result.scpccvs is not None
    assert np.isfinite(result.scpccvs).all()
    assert result.method == "approx"
    assert result.large_n_seed == 17


def test_scpc_matches_r_for_one_way_fe_iv_approx() -> None:
    data = make_one_way_fe_iv_data(2002)
    fit = pf.feols("y ~ w | fe | x ~ z", data=data)
    result = scpc(
        fit,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        method="approx",
        large_n_seed=17,
        uncond=False,
        cvs=True,
    )

    assert isinstance(result, SCPCResult)
    assert np.isfinite(result.scpcstats).all()
    assert result.scpccvs is not None
    assert np.isfinite(result.scpccvs).all()
    assert result.method == "approx"
    assert result.large_n_seed == 17


def test_scpc_matches_r_for_clustered_iv_approx() -> None:
    data = make_clustered_iv_data(2003)
    fit = pf.feols("y ~ w | x ~ z", data=data)
    result = scpc(
        fit,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        cluster="cl",
        avc=0.1,
        method="approx",
        large_n_seed=17,
        uncond=False,
        cvs=True,
    )

    assert isinstance(result, SCPCResult)
    assert np.isfinite(result.scpcstats).all()
    assert result.scpccvs is not None
    assert np.isfinite(result.scpccvs).all()
    assert result.method == "approx"
    assert result.large_n_seed == 17


def test_scpc_matches_r_for_clustered_fe_iv_approx() -> None:
    data = make_clustered_fe_iv_data(2004)
    fit = pf.feols("y ~ w | fe | x ~ z", data=data)
    result = scpc(
        fit,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        cluster="cl",
        avc=0.1,
        method="approx",
        large_n_seed=17,
        uncond=False,
        cvs=True,
    )

    assert isinstance(result, SCPCResult)
    assert np.isfinite(result.scpcstats).all()
    assert result.scpccvs is not None
    assert np.isfinite(result.scpccvs).all()
    assert result.method == "approx"
    assert result.large_n_seed == 17


def test_scpc_rejects_fixest_multi_objects() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | x ~ z", data=data, split="fe")

    with pytest.raises(ValueError, match="FixestMulti"):
        scpc(
            model,
            data,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=True,
            cvs=False,
        )


def test_scpc_pyfixest_iv_supports_i_instruments_and_literal_intercept_column() -> None:
    rng = np.random.default_rng(404)
    n = 48
    inst_grp = rng.integers(0, 3, size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    z1 = (inst_grp == 1).astype(float)
    z2 = (inst_grp == 2).astype(float)
    x = 0.9 * z1 - 0.8 * z2 + 0.35 * w + 0.45 * u
    literal_col = rng.normal(size=n)
    y = 1.2 * x + 0.4 * w + 0.6 * literal_col + u
    fe = np.repeat(np.arange(4), n // 4)
    df = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": w,
            "inst_grp": inst_grp,
            "z1": z1,
            "z2": z2,
            "Intercept": literal_col,
            "literal_col": literal_col,
            "fe": fe,
            "coord_x": rng.uniform(0.0, 1.0, size=n),
            "coord_y": rng.uniform(0.0, 1.0, size=n),
        }
    )

    # 1) PyFixest i() instrument matches explicit indicator columns (both OLS-IV and FE-IV)
    for formula_i, formula_exp in [
        ("y ~ w | x ~ i(inst_grp)", "y ~ w | x ~ z1 + z2"),
        ("y ~ w | fe | x ~ i(inst_grp)", "y ~ w | fe | x ~ z1 + z2"),
    ]:
        res_i = scpc(
            pf.feols(formula_i, data=df),
            df,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=False,
            cvs=True,
        )
        res_exp = scpc(
            pf.feols(formula_exp, data=df),
            df,
            coords_euclidean=("coord_x", "coord_y"),
            avc=0.1,
            uncond=False,
            cvs=True,
        )
        npt.assert_allclose(res_i.c0, res_exp.c0, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(res_i.cv, res_exp.cv, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(res_i.scpcstats, res_exp.scpcstats, rtol=1e-10, atol=1e-10)

    # 2) Literal column named "Intercept" in a 0 + Intercept specification
    res_lit = scpc(
        pf.feols("y ~ 0 + Intercept + w | x ~ z1 + z2", data=df),
        df,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        uncond=False,
        cvs=True,
    )
    res_ref = scpc(
        pf.feols("y ~ 0 + literal_col + w | x ~ z1 + z2", data=df),
        df,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        uncond=False,
        cvs=True,
    )
    npt.assert_allclose(res_lit.scpcstats, res_ref.scpcstats, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(res_lit.w, res_ref.w, rtol=1e-10, atol=1e-10)


def test_scpc_pyfixest_lean_error_and_extreme_coordinate_scaling() -> None:
    rng = np.random.default_rng(505)
    n = 36
    x = rng.normal(size=n)
    s1 = rng.uniform(0.0, 1.0, size=n)
    s2 = rng.uniform(0.0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "y": 0.5 + 1.1 * x + 0.3 * rng.normal(size=n),
            "x": x,
            "s1": s1,
            "s2": s2,
            "s1_tiny": s1 * 1e-200,
            "s2_tiny": s2 * 1e-200,
            "s1_huge": s1 * 1e200,
            "s2_huge": s2 * 1e200,
        }
    )

    lean_fit = pf.feols("y ~ x", data=df, lean=True)
    with pytest.raises(ValueError, match="lean=False"):
        scpc(lean_fit, df, coords_euclidean=("s1", "s2"))

    fit = pf.feols("y ~ x", data=df)
    for method in ("exact", "approx"):
        ref = scpc(
            fit,
            df,
            coords_euclidean=("s1", "s2"),
            method=method,
            large_n_seed=11,
            backend="fallback",
            avc=0.05,
        )
        for pair in (("s1_tiny", "s2_tiny"), ("s1_huge", "s2_huge")):
            scaled = scpc(
                fit,
                df,
                coords_euclidean=pair,
                method=method,
                large_n_seed=11,
                backend="fallback",
                avc=0.05,
            )
            assert np.isfinite(scaled.scpcstats).all()
            npt.assert_allclose(scaled.cv, ref.cv, rtol=1e-10, atol=1e-10)
            npt.assert_allclose(scaled.scpcstats, ref.scpcstats, rtol=1e-10, atol=1e-10)
