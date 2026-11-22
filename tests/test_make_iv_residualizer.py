from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.matrix import make_fixef_demeaner, make_iv_residualizer
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_iv_drops_fully_absorbed_extra_instrument() -> None:
    groups = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]])
    demean = make_fixef_demeaner(groups, len(groups))
    x = np.array([0, 2, 1, 4, 3, 8, 1, 7], dtype=float)
    z = np.array([0, 1, 3, 2, 1, 5, 4, 7], dtype=float)
    absorbed = groups[:, 0] + np.pi * groups[:, 1]
    values = x**2
    expected = make_iv_residualizer(x, z, demean=demean, transpose=True)(values)
    actual = make_iv_residualizer(
        x, np.column_stack([z, absorbed]), demean=demean, transpose=True
    )(values)
    npt.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="rank deficient"):
        make_iv_residualizer(x, absorbed, demean=demean)


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


@pytest.mark.parametrize("p", [1, 2])
def test_iv_rejects_near_zero_projected_design(p) -> None:
    z = np.array([[1, 1], [1, -1], [1, 1], [1, -1]], dtype=float)
    x = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float)
    x += 1e-13 * z
    with pytest.raises(ValueError, match="rank deficient"):
        make_iv_residualizer(x[:, :p], z[:, :p])


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


def test_iv_residualizer_is_invariant_to_column_units() -> None:
    rng = np.random.default_rng(718)
    z = rng.normal(size=(50, 4))
    x = z @ rng.normal(size=(4, 2)) + rng.normal(size=(50, 2))
    values = rng.normal(size=(50, 3))
    kwargs = dict(transpose=True)
    base = make_iv_residualizer(x, z, **kwargs)(values)
    rescaled = make_iv_residualizer(
        x * [1e-100, 1e100], z * [1e80, 1e-80, 1.0, 2.0], **kwargs
    )(values)
    np.testing.assert_allclose(rescaled, base, rtol=1e-12, atol=1e-12)
