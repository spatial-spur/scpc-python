from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from scpc import scpc
from scpc.types import SCPCResult
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_scpc_returns_a_result_with_the_expected_structure() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 1.8, 2.9, 3.7, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "coord_x": [0.0, 1.0, 0.5, 1.5, 2.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 1.5],
        },
        index=[1, 2, 3, 4, 5],
    )
    model = smf.ols("y ~ x", data=data).fit()

    result = scpc(
        model,
        data,
        coords_euclidean=("coord_x", "coord_y"),
        avc=0.1,
        uncond=True,
        cvs=False,
    )

    assert isinstance(result, SCPCResult)
    assert result.scpcstats.shape == (2, 6)
    assert result.scpccvs is None
    assert result.avc == pytest.approx(0.1, abs=1e-12)
    assert result.q >= 1
    assert result.w.shape[0] == 5


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_scpc() -> None:
    payload = {
        "data": {
            "y": [1.0, 1.8, 2.9, 3.7, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "coord_x": [0.0, 1.0, 0.5, 1.5, 2.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 1.5],
        },
        "coords_euclidean": ["coord_x", "coord_y"],
        "avc": 0.1,
    }
    data = pd.DataFrame(payload["data"], index=[1, 2, 3, 4, 5])
    model = smf.ols("y ~ x", data=data).fit()

    r_value = execute_r_code(
        """
        d <- data.frame(
          y = vapply(payload$data$y, as.numeric, numeric(1)),
          x = vapply(payload$data$x, as.numeric, numeric(1)),
          coord_x = vapply(payload$data$coord_x, as.numeric, numeric(1)),
          coord_y = vapply(payload$data$coord_y, as.numeric, numeric(1))
        )
        fit <- lm(y ~ x, data = d)
        out <- scpc(
          fit,
          data = d,
          # TODO: rename this to `coords_euclidean` once scpcR adopts the planned API.
          coord_euclidean = unlist(payload$coords_euclidean),
          avc = payload$avc,
          uncond = TRUE,
          cvs = FALSE
        )
        result <- list(
          scpcstats = unname(split(out$scpcstats, row(out$scpcstats))),
          avc = out$avc,
          c0 = out$c0,
          cv = out$cv,
          q = out$q
        )
        """,
        payload=payload,
    )
    py_value = scpc(
        model,
        data,
        coords_euclidean=tuple(payload["coords_euclidean"]),
        avc=payload["avc"],
        uncond=True,
        cvs=False,
    )

    npt.assert_allclose(
        py_value.scpcstats, np.array(r_value["scpcstats"]), atol=ATOL, rtol=RTOL
    )
    assert py_value.avc == pytest.approx(r_value["avc"], abs=ATOL, rel=RTOL)
    assert py_value.c0 == pytest.approx(r_value["c0"], abs=ATOL, rel=RTOL)
    assert py_value.cv == pytest.approx(r_value["cv"], abs=ATOL, rel=RTOL)
    assert py_value.q == r_value["q"]


def test_scpc_requires_keyword_only_options() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 1.8, 2.9, 3.7, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "lon": [0.0, 1.0, 0.5, 1.5, 2.0],
            "lat": [0.0, 0.0, 1.0, 1.0, 1.5],
        },
        index=[1, 2, 3, 4, 5],
    )
    model = smf.ols("y ~ x", data=data).fit()

    with pytest.raises(TypeError):
        scpc(model, data, "lon", "lat")


def test_scpc_no_longer_accepts_k() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 1.8, 2.9, 3.7, 5.1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "coord_x": [0.0, 1.0, 0.5, 1.5, 2.0],
            "coord_y": [0.0, 0.0, 1.0, 1.0, 1.5],
        },
        index=[1, 2, 3, 4, 5],
    )
    model = smf.ols("y ~ x", data=data).fit()

    with pytest.raises(TypeError, match="k"):
        scpc(model, data, k=1)
