from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import jumble_s
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_jumble_s_reorders_the_first_m_rows_from_the_random_state() -> None:
    s = np.array([[1.0], [2.0], [3.0], [4.0]])

    coords, random_t = jumble_s(s, 3, 1)

    npt.assert_allclose(
        coords,
        np.array([[1.0], [4.0], [2.0], [3.0]]),
        atol=1e-12,
        rtol=0.0,
    )
    assert random_t == 3598220700


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_jumble_s() -> None:
    s = [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]]

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".jumble_s", "scpcR")(
          matrix(unlist(payload$s), nrow = length(payload$s), byrow = TRUE),
          payload$m,
          payload$random_t
        )
        result <- list(
          coords = unname(split(out$coords, row(out$coords))),
          state = out$state
        )
        """,
        payload={"s": s, "m": 3, "random_t": 1},
    )
    py_coords, py_state = jumble_s(np.array(s), 3, 1)

    npt.assert_allclose(py_coords, np.array(r_value["coords"]), atol=ATOL, rtol=RTOL)
    assert py_state == r_value["state"]
