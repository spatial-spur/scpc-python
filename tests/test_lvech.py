from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import lvech
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_lvech_returns_the_strict_lower_triangle() -> None:
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    npt.assert_allclose(lvech(mat), np.array([4.0, 7.0, 8.0]), atol=1e-12, rtol=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_lvech() -> None:
    mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    r_value = execute_r_code(
        'result <- getFromNamespace(".lvech", "scpcR")(as.matrix(do.call(rbind, payload$mat)))',
        payload={"mat": mat},
    )
    py_value = lvech(np.array(mat))

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
