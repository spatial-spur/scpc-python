from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import raninds
from tests.utils import R, execute_r_code


def test_raninds_generates_linked_zero_based_indices() -> None:
    indices, random_t = raninds(5, 6, 1)

    npt.assert_array_equal(indices, np.array([0, 4, 3, 1, 2, 4, 0]))
    assert random_t == 3259150681


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_raninds() -> None:
    r_value = execute_r_code(
        """
        out <- getFromNamespace(".raninds", "scpcR")(
          payload$n,
          payload$capM,
          payload$random_t
        )
        result <- list(indices = out$indices, state = out$state)
        """,
        payload={"n": 5, "capM": 6, "random_t": 1},
    )
    py_indices, py_state = raninds(5, 6, 1)

    npt.assert_array_equal(py_indices, np.array(r_value["indices"]) - 1)
    assert py_state == r_value["state"]
