from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyfixest as pf
import pytest
import statsmodels.formula.api as smf

from scpc.utils.data import get_conditional_projection_setup
from scpc.utils.data import get_fixest_iv_design
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code, reorder_r_columns_to_py
from tests.utils import make_one_way_fe_iv_data


def test_get_conditional_projection_setup_keeps_the_baseline_matrix_for_statsmodels() -> (
    None
):
    data = pd.DataFrame({"y": [1.0, 2.0, 4.0], "x": [0.0, 1.0, 3.0]}, index=[1, 2, 4])
    model = smf.ols("y ~ x", data=data).fit()
    model_mat = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]])

    setup = get_conditional_projection_setup(model, model_mat, n=3, uncond=False)

    npt.assert_allclose(setup.model_mat, model_mat, atol=1e-12, rtol=0.0)
    assert setup.include_intercept is True
    assert setup.fixef_id is None
    assert setup.residualize is None
    assert setup.is_iv is False


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_conditional_projection_setup() -> None:
    payload = {
        "data": {"y": [1.0, 2.0, 4.0], "x": [0.0, 1.0, 3.0]},
        "model_mat": [[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]],
        "n": 3,
        "uncond": False,
    }
    data = pd.DataFrame(payload["data"], index=[1, 2, 4])
    model = smf.ols("y ~ x", data=data).fit()

    r_value = execute_r_code(
        """
        d <- data.frame(
          y = vapply(payload$data$y, as.numeric, numeric(1)),
          x = vapply(payload$data$x, as.numeric, numeric(1))
        )
        fit <- lm(y ~ x, data = d)
        out <- getFromNamespace(".get_conditional_projection_setup", "scpcR")(
          fit,
          matrix(unlist(payload$model_mat), nrow = length(payload$model_mat), byrow = TRUE),
          payload$n,
          payload$uncond
        )
        result <- list(
          model_mat = unname(split(out$model_mat, row(out$model_mat))),
          include_intercept = out$include_intercept,
          fixef_id = out$fixef_id
        )
        """,
        payload=payload,
    )
    py_value = get_conditional_projection_setup(
        model,
        np.array(payload["model_mat"]),
        n=payload["n"],
        uncond=payload["uncond"],
    )

    npt.assert_allclose(
        py_value.model_mat, np.array(r_value["model_mat"]), atol=ATOL, rtol=RTOL
    )
    assert py_value.include_intercept is r_value["include_intercept"]
    assert py_value.fixef_id == r_value["fixef_id"]
    assert py_value.residualize is None
    assert py_value.is_iv is False


def test_get_conditional_projection_setup_marks_pyfixest_iv_models() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "w": [0.4, -0.1, 1.0, 0.6, -0.3],
            "x": [0.2, 0.8, 1.1, 1.7, 2.0],
            "z": [0.1, 0.3, 0.5, 0.8, 1.1],
        }
    )
    model = pf.feols("y ~ w | x ~ z", data=data)
    model_mat = np.asarray(model._X, dtype=float)
    design = get_fixest_iv_design(model)

    setup = get_conditional_projection_setup(
        model, model_mat, n=model_mat.shape[0], uncond=False
    )

    npt.assert_allclose(
        setup.model_mat,
        np.asarray(design["model_mat"], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    assert setup.include_intercept is True
    assert setup.fixef_id is None
    assert callable(setup.residualize)
    assert setup.is_iv is True


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_conditional_projection_setup_for_pyfixest_fe_iv() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | fe | x ~ z", data=data)
    design = get_fixest_iv_design(model)
    model_mat = np.asarray(design["model_mat"], dtype=float)
    py_value = get_conditional_projection_setup(
        model,
        model_mat,
        n=model_mat.shape[0],
        uncond=False,
    )
    y = np.column_stack(
        (
            np.linspace(-1.0, 1.0, len(data), dtype=float),
            np.linspace(1.0, -1.0, len(data), dtype=float),
        )
    )

    r_value = execute_r_code(
        """
        suppressPackageStartupMessages(library(fixest))
        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x),
          w = unlist(payload$data$w),
          z = unlist(payload$data$z),
          fe = unlist(payload$data$fe)
        )
        fit <- fixest::feols(y ~ w | fe | x ~ z, data = d)
        design <- getFromNamespace(".get_fixest_iv_design", "scpcR")(fit)
        cond_fixef_id <- design$fixef_id
        model_mat_cond <- getFromNamespace(".demean_for_scpc", "scpcR")(
          design$model_mat,
          cond_fixef_id,
          "fixest IV second-stage model matrix"
        )
        residualize <- getFromNamespace(".make_iv_residualizer", "scpcR")(
          design$X,
          design$Z,
          fixef_id = cond_fixef_id
        )
        y_mat <- matrix(
          unlist(payload$y),
          nrow = length(payload$y),
          byrow = TRUE
        )
        resid <- residualize(y_mat)
        result <- list(
          coef_names = unname(names(stats::coef(fit))),
          model_mat = unname(split(model_mat_cond, row(model_mat_cond))),
          resid = unname(split(resid, row(resid)))
        )
        """,
        payload={"data": data.to_dict(orient="list"), "y": y.tolist()},
    )
    r_model_mat = reorder_r_columns_to_py(
        [str(name) for name in model._coefnames],
        [str(name) for name in r_value["coef_names"]],
        np.array(r_value["model_mat"]),
    )

    npt.assert_allclose(py_value.model_mat, r_model_mat, atol=ATOL, rtol=RTOL)
    assert py_value.include_intercept is False
    assert py_value.fixef_id is None
    assert py_value.residualize is not None
    npt.assert_allclose(
        np.asarray(py_value.residualize(y), dtype=float),
        np.array(r_value["resid"]),
        atol=ATOL,
        rtol=RTOL,
    )
    assert py_value.is_iv is True


def test_get_conditional_projection_setup_uses_demeaned_pyfixest_fe_matrix() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "fe": [1, 1, 2, 2],
        }
    )
    model = pf.feols("y ~ x | fe", data=data)
    model_mat = np.asarray(model._X, dtype=float)

    setup = get_conditional_projection_setup(
        model, model_mat, n=model_mat.shape[0], uncond=False
    )

    npt.assert_allclose(setup.model_mat, model_mat, atol=1e-12, rtol=0.0)
    assert setup.include_intercept is False
    assert setup.fixef_id is None
    assert setup.residualize is None
    assert setup.is_iv is False


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_conditional_projection_setup_for_pyfixest_fe() -> None:
    payload = {
        "data": {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "fe": [1, 1, 2, 2],
        }
    }
    data = pd.DataFrame(payload["data"])
    model = pf.feols("y ~ x | fe", data=data)
    model_mat = np.asarray(model._X, dtype=float)

    r_value = execute_r_code(
        """
        suppressPackageStartupMessages(library(fixest))
        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x),
          fe = unlist(payload$data$fe)
        )
        fit <- fixest::feols(y ~ x | fe, data = d)
        model_mat <- stats::model.matrix(fit)
        out <- getFromNamespace(".get_conditional_projection_setup", "scpcR")(
          fit,
          model_mat,
          nrow(model_mat),
          FALSE
        )
        result <- list(
          coef_names = unname(names(stats::coef(fit))),
          model_mat = unname(split(out$model_mat, row(out$model_mat))),
          include_intercept = out$include_intercept
        )
        """,
        payload=payload,
    )
    py_value = get_conditional_projection_setup(
        model,
        model_mat,
        n=model_mat.shape[0],
        uncond=False,
    )
    r_value_matrix = reorder_r_columns_to_py(
        [str(name) for name in model._coefnames],
        [str(name) for name in r_value["coef_names"]],
        np.array(r_value["model_mat"]),
    )

    npt.assert_allclose(py_value.model_mat, r_value_matrix, atol=ATOL, rtol=RTOL)
    assert py_value.include_intercept is r_value["include_intercept"]
