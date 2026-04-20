from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import orthogonalize_w
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_orthogonalize_w_leaves_an_already_orthogonal_basis_unchanged() -> None:
    w = np.array(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ]
    )
    xj = np.array([1.0, 1.0])
    xjs = np.array([1.0, 1.0])
    model_mat = np.array([[1.0], [1.0]])

    npt.assert_allclose(
        orthogonalize_w(w, xj, xjs, model_mat, include_intercept=False),
        w,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_orthogonalize_w() -> None:
    w = [
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
    ]
    xj = [1.0, 1.0]
    xjs = [1.0, 1.0]
    model_mat = [[1.0], [1.0]]

    r_value = execute_r_code(
        """
        result <- getFromNamespace(".orthogonalize_W", "scpcR")(
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE),
          unlist(payload$xj),
          unlist(payload$xjs),
          matrix(unlist(payload$model_mat), nrow = length(payload$model_mat), byrow = TRUE),
          FALSE,
          NULL
        )
        """,
        payload={"w": w, "xj": xj, "xjs": xjs, "model_mat": model_mat},
    )
    py_value = orthogonalize_w(
        np.array(w),
        np.array(xj),
        np.array(xjs),
        np.array(model_mat),
        include_intercept=False,
    )

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
