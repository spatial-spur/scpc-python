from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pyfixest as pf
import pytest

from scpc.utils.data import get_fixest_score_matrix
from tests.config import ATOL, RTOL
from tests.utils import (
    R,
    execute_r_code,
    make_basic_iv_data,
    make_one_way_fe_iv_data,
    normalize_r_score_names,
    reorder_r_columns_to_py,
)


def test_get_fixest_score_matrix_maps_iv_scores_into_coefficient_space() -> None:
    data = make_basic_iv_data(2001)
    model = pf.feols("y ~ w | x ~ z", data=data)

    npt.assert_allclose(
        np.asarray(model._scores, dtype=float),
        np.asarray(model._Z, dtype=float) * np.asarray(model._u_hat, dtype=float)[:, None],
        atol=1e-12,
        rtol=0.0,
    )
    npt.assert_allclose(
        get_fixest_score_matrix(model),
        np.asarray(model._scores, dtype=float)
        @ np.asarray(model._tZZinv, dtype=float)
        @ np.asarray(model._tZX, dtype=float),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_r_reference_fe_iv_score_names_match_coef_order() -> None:
    data = make_one_way_fe_iv_data(2002)
    r_value = execute_r_code(
        """
        suppressPackageStartupMessages({
          library(fixest)
          library(sandwich)
        })
        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x),
          w = unlist(payload$data$w),
          z = unlist(payload$data$z),
          fe = unlist(payload$data$fe)
        )
        fit <- fixest::feols(y ~ w | fe | x ~ z, data = d)
        result <- list(
          coef_names = unname(names(stats::coef(fit))),
          score_names = unname(colnames(sandwich::estfun(fit)))
        )
        """,
        payload={"data": data.to_dict(orient="list")},
    )

    assert r_value["score_names"][0] == "fit_x"
    assert r_value["score_names"][1] == ""
    assert normalize_r_score_names(
        [str(name) for name in r_value["score_names"]],
        [str(name) for name in r_value["coef_names"]],
    ) == [str(name) for name in r_value["coef_names"]]


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_fixest_score_matrix() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | fe | x ~ z", data=data)

    r_value = execute_r_code(
        """
        suppressPackageStartupMessages({
          library(fixest)
          library(sandwich)
        })
        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x),
          w = unlist(payload$data$w),
          z = unlist(payload$data$z),
          fe = unlist(payload$data$fe)
        )
        fit <- fixest::feols(y ~ w | fe | x ~ z, data = d)
        result <- list(
          coef_names = unname(names(stats::coef(fit))),
          score_names = unname(colnames(sandwich::estfun(fit))),
          score_mat = unname(split(sandwich::estfun(fit), row(sandwich::estfun(fit))))
        )
        """,
        payload={"data": data.to_dict(orient="list")},
    )
    py_value = get_fixest_score_matrix(model)
    r_value_matrix = reorder_r_columns_to_py(
        [str(name) for name in model._coefnames],
        normalize_r_score_names(
            [str(name) for name in r_value["score_names"]],
            [str(name) for name in r_value["coef_names"]],
        ),
        np.array(r_value["score_mat"]),
    )

    npt.assert_allclose(py_value, r_value_matrix, atol=ATOL, rtol=RTOL)
