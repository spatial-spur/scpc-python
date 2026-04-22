from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import make_iv_residualizer, orthogonalize_w_iv
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_orthogonalize_w_iv_leaves_a_simple_basis_unchanged() -> None:
    wfin = np.array(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ]
    )
    xj = np.array([1.0, 1.0])
    xjs = np.array([1.0, 1.0])
    residualize = make_iv_residualizer(np.ones((2, 1)), np.ones((2, 1)))

    npt.assert_allclose(
        orthogonalize_w_iv(wfin, xj, xjs, residualize),
        wfin,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_orthogonalize_w_iv() -> None:
    payload = {
        "wfin": [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
        ],
        "xj": [1.0, 1.0],
        "xjs": [1.0, 1.0],
        "X": [[1.0], [1.0]],
        "Z": [[1.0], [1.0]],
    }

    r_value = execute_r_code(
        """
        Wfin <- matrix(unlist(payload$wfin), nrow = length(payload$wfin), byrow = TRUE)
        X <- matrix(unlist(payload$X), nrow = length(payload$X), byrow = TRUE)
        Z <- matrix(unlist(payload$Z), nrow = length(payload$Z), byrow = TRUE)
        residualize <- getFromNamespace(".make_iv_residualizer", "scpcR")(X, Z)
        result <- getFromNamespace(".orthogonalize_W_iv", "scpcR")(
          Wfin,
          unlist(payload$xj),
          unlist(payload$xjs),
          residualize
        )
        """,
        payload=payload,
    )
    py_value = orthogonalize_w_iv(
        np.array(payload["wfin"]),
        np.array(payload["xj"]),
        np.array(payload["xjs"]),
        make_iv_residualizer(np.array(payload["X"]), np.array(payload["Z"])),
    )

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
