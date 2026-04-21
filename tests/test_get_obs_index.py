from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from scpc.utils.data import get_obs_index
from tests.utils import R, execute_r_code


def test_get_obs_index_returns_the_surviving_row_positions() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, np.nan, 3.0],
        },
        index=[1, 2, 3, 4],
    )
    model = smf.ols("y ~ x", data=data).fit()

    assert list(get_obs_index(model, data)) == [1, 2, 4]


def test_get_obs_index_maps_gappy_integer_index_to_row_positions() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
        },
        index=[1, 4, 7],
    )
    model = smf.ols("y ~ x", data=data).fit()

    assert list(get_obs_index(model, data)) == [1, 2, 3]


def test_get_obs_index_preserves_model_order_for_permuted_integer_index() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
        },
        index=[2, 1, 3],
    )
    model = smf.ols("y ~ x", data=data).fit()

    assert list(get_obs_index(model, data)) == [1, 2, 3]


def test_get_obs_index_rejects_duplicate_data_index() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
        },
        index=[1, 1, 2],
    )
    model = smf.ols("y ~ x", data=data).fit()

    with pytest.raises(ValueError, match=r"`data\.index` must be unique"):
        get_obs_index(model, data)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_obs_index() -> None:
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
        result <- getFromNamespace(".get_obs_index", "scpcR")(fit, d)
        """,
        payload=payload,
    )
    py_value = get_obs_index(model, data)

    assert list(py_value) == r_value
