from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from scpc.utils.data import get_scpc_model_matrix
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


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
