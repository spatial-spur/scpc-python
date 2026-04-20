from __future__ import annotations

import math

import numpy as np
import pytest

from scpc.utils.matrix import get_tau
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_tau_returns_the_expected_statistic() -> None:
    y = np.array([2.0, 1.0])
    w = np.array(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ]
    )

    assert get_tau(y, w) == pytest.approx(3.0, abs=1e-12)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_tau() -> None:
    y = [2.0, 1.0]
    w = [
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
    ]

    r_value = execute_r_code(
        """
        result <- as.numeric(getFromNamespace(".gettau", "scpcR")(
          unlist(payload$y),
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE)
        ))
        """,
        payload={"y": y, "w": w},
    )
    py_value = get_tau(np.array(y), np.array(w))

    assert py_value == pytest.approx(r_value, abs=ATOL, rel=RTOL)
