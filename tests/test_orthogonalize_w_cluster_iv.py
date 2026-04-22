from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import make_iv_residualizer, orthogonalize_w_cluster_iv
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_orthogonalize_w_cluster_iv_accepts_preencoded_cluster_indices() -> None:
    wfin = np.array(
        [
            [1.0 / math.sqrt(3.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(3.0), 0.0],
            [1.0 / math.sqrt(3.0), -1.0 / math.sqrt(2.0)],
        ]
    )
    cl_labels = np.array(["c", "a", "c", "b"])
    cl_idx = np.array([2, 0, 2, 1])
    xj_indiv = np.array([1.0, 2.0, 1.5, 0.5])
    residualize = make_iv_residualizer(np.ones((4, 1)), np.ones((4, 1)))

    from_labels = orthogonalize_w_cluster_iv(wfin, cl_labels, xj_indiv, residualize)
    from_idx = orthogonalize_w_cluster_iv(wfin, cl_idx, xj_indiv, residualize)

    npt.assert_allclose(from_idx, from_labels, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_orthogonalize_w_cluster_iv() -> None:
    payload = {
        "wfin": [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ],
        "cl_vec": ["a", "b"],
        "xj_indiv": [1.0, 1.0],
        "X": [[1.0], [1.0]],
        "Z": [[1.0], [1.0]],
    }

    r_value = execute_r_code(
        """
        Wfin <- matrix(unlist(payload$wfin), nrow = length(payload$wfin), byrow = TRUE)
        X <- matrix(unlist(payload$X), nrow = length(payload$X), byrow = TRUE)
        Z <- matrix(unlist(payload$Z), nrow = length(payload$Z), byrow = TRUE)
        residualize <- getFromNamespace(".make_iv_residualizer", "scpcR")(X, Z)
        result <- getFromNamespace(".orthogonalize_W_cluster_iv", "scpcR")(
          Wfin,
          factor(unlist(payload$cl_vec)),
          unlist(payload$xj_indiv),
          residualize
        )
        """,
        payload=payload,
    )
    py_value = orthogonalize_w_cluster_iv(
        np.array(payload["wfin"]),
        np.array(payload["cl_vec"]),
        np.array(payload["xj_indiv"]),
        make_iv_residualizer(np.array(payload["X"]), np.array(payload["Z"])),
    )

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
