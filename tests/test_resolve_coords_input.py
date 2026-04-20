from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from scpc.utils.data import resolve_coords_input
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_resolve_coords_input_selects_the_euclidean_coordinate_columns() -> None:
    data = pd.DataFrame(
        {
            "coord_x": [0.0, 1.0, 2.0, 3.0],
            "coord_y": [0.0, 1.0, 1.5, 2.0],
        },
        index=[1, 2, 3, 4],
    )
    obs_index = np.array([1, 2, 4])

    result = resolve_coords_input(data, obs_index, None, None, ("coord_x", "coord_y"))

    npt.assert_allclose(
        result.coords,
        np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]]),
        atol=1e-12,
        rtol=0.0,
    )
    assert result.latlong is False


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_resolve_coords_input() -> None:
    payload = {
        "data": {
            "coord_x": [0.0, 1.0, 2.0, 3.0],
            "coord_y": [0.0, 1.0, 1.5, 2.0],
        },
        "obs_index": [1, 2, 4],
        "coord_euclidean": ["coord_x", "coord_y"],
    }
    data = pd.DataFrame(payload["data"], index=[1, 2, 3, 4])

    r_value = execute_r_code(
        """
        d <- data.frame(
          coord_x = vapply(payload$data$coord_x, as.numeric, numeric(1)),
          coord_y = vapply(payload$data$coord_y, as.numeric, numeric(1))
        )
        out <- getFromNamespace(".resolve_coords_input", "scpcR")(
          d,
          unlist(payload$obs_index),
          NULL,
          NULL,
          unlist(payload$coord_euclidean)
        )
        result <- list(
          coords = unname(split(out$coords, row(out$coords))),
          latlong = out$latlong
        )
        """,
        payload=payload,
    )
    py_value = resolve_coords_input(
        data,
        np.array(payload["obs_index"]),
        None,
        None,
        tuple(payload["coord_euclidean"]),
    )

    npt.assert_allclose(
        py_value.coords, np.array(r_value["coords"]), atol=ATOL, rtol=RTOL
    )
    assert py_value.latlong is r_value["latlong"]
