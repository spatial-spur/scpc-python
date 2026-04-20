from __future__ import annotations

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from scpc.utils.data import has_fixest_fe
from tests.utils import R, execute_r_code


def test_has_fixest_fe_is_false_for_statsmodels_ols() -> None:
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}, index=[1, 2, 3])
    model = smf.ols("y ~ x", data=data).fit()

    assert has_fixest_fe(model) is False


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_has_fixest_fe() -> None:
    payload = {"data": {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}}
    data = pd.DataFrame(payload["data"], index=[1, 2, 3])
    model = smf.ols("y ~ x", data=data).fit()

    r_value = execute_r_code(
        """
        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x)
        )
        fit <- lm(y ~ x, data = d)
        result <- getFromNamespace(".has_fixest_fe", "scpcR")(fit)
        """,
        payload=payload,
    )
    py_value = has_fixest_fe(model)

    assert py_value is r_value
