from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pyfixest as pf
import pytest

from scpc.utils.data import get_fixest_bread_inv
from tests.config import ATOL, RTOL
from tests.utils import (
    R,
    execute_r_code,
    make_one_way_fe_iv_data,
    reorder_r_square_matrix_to_py,
)


def test_get_fixest_bread_inv_returns_the_stored_pyfixest_bread_matrix() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | fe | x ~ z", data=data)

    npt.assert_allclose(
        get_fixest_bread_inv(model),
        np.asarray(model._bread, dtype=float),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_fixest_bread_inv() -> None:
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
          bread_names = unname(colnames(sandwich::bread(fit))),
          bread = unname(split(sandwich::bread(fit) / nobs(fit), row(sandwich::bread(fit))))
        )
        """,
        payload={"data": data.to_dict(orient="list")},
    )
    py_value = get_fixest_bread_inv(model)
    r_value_matrix = reorder_r_square_matrix_to_py(
        [str(name) for name in model._coefnames],
        [str(name) for name in r_value["bread_names"]],
        np.array(r_value["bread"]),
    )

    npt.assert_allclose(py_value, r_value_matrix, atol=ATOL, rtol=RTOL)
