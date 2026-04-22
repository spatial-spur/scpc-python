from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyfixest as pf
import pytest
import statsmodels.formula.api as smf

from scpc.utils.data import get_scpc_model_matrix
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code, reorder_r_columns_to_py


def test_get_scpc_model_matrix_returns_intercept_and_regressor_columns() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, np.nan, 3.0],
        },
        index=[1, 2, 3, 4],
    )
    model = smf.ols("y ~ x", data=data).fit()
    expected = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]])

    npt.assert_allclose(get_scpc_model_matrix(model), expected, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_scpc_model_matrix() -> None:
    payload = {
        "data": {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, None, 3.0]},
    }
    data = pd.DataFrame(payload["data"], index=[1, 2, 3, 4])
    model = smf.ols("y ~ x", data=data).fit()

    r_value = execute_r_code(
        """
        d <- data.frame(
          y = vapply(payload$data$y, as.numeric, numeric(1)),
          x = vapply(
            payload$data$x,
            function(value) if (is.null(value)) NA_real_ else as.numeric(value),
            numeric(1)
          )
        )
        fit <- lm(y ~ x, data = d)
        result <- getFromNamespace(".get_scpc_model_matrix", "scpcR")(fit)
        """,
        payload=payload,
    )
    py_value = get_scpc_model_matrix(model)

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)


def test_get_scpc_model_matrix_returns_the_pyfixest_second_stage_matrix() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "w": [0.5, -0.2, 1.1, 0.7],
            "x": [0.2, 0.8, 1.2, 1.7],
            "z": [0.1, 0.3, 0.5, 0.8],
        }
    )
    model = pf.feols("y ~ w | x ~ z", data=data)
    coef_names = [str(name) for name in model._coefnames]
    z_names = [str(name) for name in model._coefnames_z]
    endo_idx = [i for i, name in enumerate(coef_names) if name not in z_names]
    exo_idx = [i for i, name in enumerate(coef_names) if name in z_names]
    model_mat = get_scpc_model_matrix(model)

    npt.assert_allclose(
        model_mat[:, exo_idx],
        np.asarray(model._X, dtype=float)[:, exo_idx],
        atol=1e-12,
        rtol=0.0,
    )
    npt.assert_allclose(
        model_mat[:, endo_idx],
        np.asarray(model._X_hat, dtype=float)[:, None],
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_scpc_model_matrix_for_pyfixest_iv() -> None:
    payload = {
        "data": {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "w": [0.4, -0.1, 1.0, 0.6, -0.3],
            "x": [0.2, 0.8, 1.1, 1.7, 2.0],
            "z": [0.1, 0.3, 0.5, 0.8, 1.1],
        }
    }
    data = pd.DataFrame(payload["data"])
    model = pf.feols("y ~ w | x ~ z", data=data)

    r_value = execute_r_code(
        """
        suppressPackageStartupMessages(library(fixest))
        d <- data.frame(
          y = vapply(payload$data$y, as.numeric, numeric(1)),
          w = vapply(payload$data$w, as.numeric, numeric(1)),
          x = vapply(payload$data$x, as.numeric, numeric(1)),
          z = vapply(payload$data$z, as.numeric, numeric(1))
        )
        fit <- fixest::feols(y ~ w | x ~ z, data = d)
        design <- getFromNamespace(".get_fixest_iv_design", "scpcR")(fit)
        result <- list(
          coef_names = unname(design$coef_names),
          model_mat = unname(split(design$model_mat, row(design$model_mat)))
        )
        """,
        payload=payload,
    )
    py_value = get_scpc_model_matrix(model)
    r_value_matrix = reorder_r_columns_to_py(
        [str(name) for name in model._coefnames],
        [str(name) for name in r_value["coef_names"]],
        np.array(r_value["model_mat"]),
    )

    npt.assert_allclose(py_value, r_value_matrix, atol=ATOL, rtol=RTOL)
