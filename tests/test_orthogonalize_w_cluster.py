from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import orthogonalize_w_cluster
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_orthogonalize_w_cluster_leaves_the_simple_cluster_basis_unchanged() -> None:
    w = np.array(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ]
    )
    cl_vec = np.array(["a", "b"])
    xj_indiv = np.array([1.0, 1.0])
    model_mat_indiv = np.array([[1.0], [1.0]])

    npt.assert_allclose(
        orthogonalize_w_cluster(
            w,
            cl_vec,
            xj_indiv,
            model_mat_indiv,
            include_intercept=False,
        ),
        w,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_orthogonalize_w_cluster() -> None:
    w = [
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
    ]
    cl_vec = ["a", "b"]
    xj_indiv = [1.0, 1.0]
    model_mat_indiv = [[1.0], [1.0]]

    r_value = execute_r_code(
        """
        result <- getFromNamespace(".orthogonalize_W_cluster", "scpcR")(
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE),
          factor(unlist(payload$cl_vec)),
          unlist(payload$xj_indiv),
          matrix(
            unlist(payload$model_mat_indiv),
            nrow = length(payload$model_mat_indiv),
            byrow = TRUE
          ),
          FALSE
        )
        """,
        payload={
            "w": w,
            "cl_vec": cl_vec,
            "xj_indiv": xj_indiv,
            "model_mat_indiv": model_mat_indiv,
        },
    )
    py_value = orthogonalize_w_cluster(
        np.array(w),
        np.array(cl_vec),
        np.array(xj_indiv),
        np.array(model_mat_indiv),
        include_intercept=False,
    )

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
