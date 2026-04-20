from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import get_distmat
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_distmat_returns_euclidean_distances() -> None:
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    expected = np.array(
        [
            [0.0, 5.0, 10.0],
            [5.0, 0.0, 5.0],
            [10.0, 5.0, 0.0],
        ]
    )

    npt.assert_allclose(get_distmat(coords, False), expected, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_distmat() -> None:
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])

    r_value = execute_r_code(
        'result <- getFromNamespace(".getdistmat", "scpcR")(as.matrix(do.call(rbind, payload$coords)), FALSE)',
        payload={"coords": coords.tolist()},
    )
    py_value = get_distmat(coords, False)

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
