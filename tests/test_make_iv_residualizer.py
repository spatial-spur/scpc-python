from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import make_iv_residualizer
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_make_iv_residualizer_matches_the_direct_qr_based_formula() -> None:
    x = np.array(
        [
            [1.0, -1.0, 0.5],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.4],
            [1.0, 2.0, 2.1],
            [1.0, 0.5, -0.1],
            [1.0, -0.5, 0.3],
        ]
    )
    z = np.array(
        [
            [1.0, -1.0, 1.5],
            [1.0, 0.0, 0.2],
            [1.0, 1.0, -0.4],
            [1.0, 2.0, 1.1],
            [1.0, 0.5, 0.3],
            [1.0, -0.5, -0.8],
        ]
    )
    y = np.array([2.0, 1.3, 3.1, 4.2, 0.4, 1.0])

    qz, _ = np.linalg.qr(z, mode="reduced")
    pzx = qz @ (qz.T @ x)
    pzy = qz @ (qz.T @ y[:, None])
    expected = y[:, None] - x @ np.linalg.solve(x.T @ pzx, x.T @ pzy)

    residualize = make_iv_residualizer(x, z)

    npt.assert_allclose(
        residualize(y),
        expected[:, 0],
        atol=1e-12,
        rtol=0.0,
    )


def test_make_iv_residualizer_errors_on_rank_deficient_designs() -> None:
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 3.0, 3.0],
        ]
    )
    z = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="rank deficient"):
        make_iv_residualizer(x, z)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_make_iv_residualizer() -> None:
    payload = {
        "X": [
            [1.0, -1.0, 0.5],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.4],
            [1.0, 2.0, 2.1],
            [1.0, 0.5, -0.1],
            [1.0, -0.5, 0.3],
        ],
        "Z": [
            [1.0, -1.0, 1.5],
            [1.0, 0.0, 0.2],
            [1.0, 1.0, -0.4],
            [1.0, 2.0, 1.1],
            [1.0, 0.5, 0.3],
            [1.0, -0.5, -0.8],
        ],
        "Y": [
            [2.0, -0.5],
            [1.3, 0.1],
            [3.1, 0.7],
            [4.2, 1.4],
            [0.4, 0.3],
            [1.0, -1.2],
        ],
    }

    r_value = execute_r_code(
        """
        X <- matrix(unlist(payload$X), nrow = length(payload$X), byrow = TRUE)
        Z <- matrix(unlist(payload$Z), nrow = length(payload$Z), byrow = TRUE)
        Y <- matrix(unlist(payload$Y), nrow = length(payload$Y), byrow = TRUE)
        residualize <- getFromNamespace(".make_iv_residualizer", "scpcR")(X, Z)
        result <- residualize(Y)
        """,
        payload=payload,
    )
    py_value = make_iv_residualizer(np.array(payload["X"]), np.array(payload["Z"]))(
        np.array(payload["Y"])
    )

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
