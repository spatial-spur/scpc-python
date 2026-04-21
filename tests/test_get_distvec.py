from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import get_distvec
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_distvec_computes_paired_euclidean_distances() -> None:
    s1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    s2 = np.array([[3.0, 4.0], [1.0, 2.0], [2.0, 3.0]])

    result = get_distvec(s1, s2, latlong=False)

    npt.assert_allclose(result, np.array([5.0, 1.0, 3.0]), atol=1e-12, rtol=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
@pytest.mark.parametrize("latlong", [False, True])
def test_python_r_parity_get_distvec(latlong: bool) -> None:
    s1 = [
        [10.0, 50.0],
        [10.5, 50.25],
        [11.0, 50.5],
    ]
    s2 = [
        [10.2, 50.1],
        [10.8, 50.0],
        [11.4, 50.75],
    ]

    r_value = execute_r_code(
        """
        result <- getFromNamespace(".getdistvec", "scpcR")(
          matrix(unlist(payload$s1), nrow = length(payload$s1), byrow = TRUE),
          matrix(unlist(payload$s2), nrow = length(payload$s2), byrow = TRUE),
          payload$latlong
        )
        """,
        payload={"s1": s1, "s2": s2, "latlong": latlong},
    )
    py_value = get_distvec(np.array(s1), np.array(s2), latlong=latlong)

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
