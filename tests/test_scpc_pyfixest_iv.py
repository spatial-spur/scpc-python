from __future__ import annotations

import numpy as np
import numpy.testing as npt
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


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_scpc_matches_r_for_non_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity("y ~ w | x ~ z", make_basic_iv_data(2001))


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_scpc_matches_r_for_one_way_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | fe | x ~ z",
        make_one_way_fe_iv_data(2002),
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_scpc_matches_r_for_two_way_fe_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | fe1 + fe2 | x ~ z",
        make_two_way_fe_iv_data(2005),
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_scpc_matches_r_for_clustered_iv_exact() -> None:
    _assert_pyfixest_scpc_parity(
        "y ~ w | x ~ z",
        make_clustered_iv_data(2003),
        cluster="cl",
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
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
