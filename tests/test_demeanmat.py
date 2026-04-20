from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import demeanmat
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_demeanmat_double_demeans_the_matrix() -> None:
    mat = np.array([[1.0, 2.0], [4.0, 8.0]])
    expected = np.array([[0.75, -0.75], [-0.75, 0.75]])

    npt.assert_allclose(demeanmat(mat), expected, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_demeanmat() -> None:
    mat = [[1.0, 2.0], [4.0, 8.0]]

    r_value = execute_r_code(
        'result <- getFromNamespace(".demeanmat", "scpcR")(matrix(unlist(payload$mat), nrow = length(payload$mat), byrow = TRUE))',
        payload={"mat": mat},
    )
    py_value = demeanmat(np.array(mat))

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
