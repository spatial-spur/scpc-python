from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import get_w
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def assert_columns_allclose_up_to_sign(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    npt.assert_allclose(left[:, 0], right[:, 0], atol=ATOL, rtol=RTOL)
    for column in range(1, left.shape[1]):
        npt.assert_allclose(
            np.abs(left[:, column]), np.abs(right[:, column]), atol=ATOL, rtol=RTOL
        )


def test_get_w_builds_the_expected_two_point_basis() -> None:
    distmat = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = np.array(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ]
    )

    assert_columns_allclose_up_to_sign(get_w(distmat, math.log(2.0), 1), expected)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_w() -> None:
    distmat = [[0.0, 1.0], [1.0, 0.0]]
    c0 = 0.5
    qmax = 1

    r_value = execute_r_code(
        """
        result <- getFromNamespace(".getW", "scpcR")(
          matrix(unlist(payload$distmat), nrow = length(payload$distmat), byrow = TRUE),
          payload$c0,
          payload$qmax
        )
        """,
        payload={"distmat": distmat, "c0": c0, "qmax": qmax},
    )
    py_value = get_w(np.array(distmat), c0, qmax)

    assert_columns_allclose_up_to_sign(py_value, np.array(r_value))
