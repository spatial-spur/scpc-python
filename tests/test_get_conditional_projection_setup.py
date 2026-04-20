from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from scpc.utils.data import get_conditional_projection_setup
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


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
