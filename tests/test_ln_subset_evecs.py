from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import ln_subset_evecs
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def assert_columns_allclose_up_to_sign(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    for column in range(left.shape[1]):
        if np.allclose(left[:, column], right[:, column], atol=ATOL, rtol=RTOL):
            continue
        npt.assert_allclose(left[:, column], -right[:, column], atol=ATOL, rtol=RTOL)


def test_ln_subset_evecs_returns_the_expected_subset_eigenvectors() -> None:
    distmat = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    result = ln_subset_evecs(distmat, 0.5, 2)
    expected = np.array(
        [
            [0.7071067811865476, 0.4082482904638631],
            [0.0, -0.8164965809277261],
            [-0.7071067811865476, 0.4082482904638631],
        ]
    )

    assert_columns_allclose_up_to_sign(result, expected)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_ln_subset_evecs() -> None:
    distmat = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ]

    r_value = execute_r_code(
        """
        result <- getFromNamespace(".ln_subset_evecs", "scpcR")(
          matrix(unlist(payload$distmat), nrow = length(payload$distmat), byrow = TRUE),
          payload$c0,
          payload$qmax
        )
        """,
        payload={"distmat": distmat, "c0": 0.5, "qmax": 2},
    )
    py_value = ln_subset_evecs(np.array(distmat), 0.5, 2)

    assert_columns_allclose_up_to_sign(py_value, np.array(r_value))
