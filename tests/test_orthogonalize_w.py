from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from scpc.utils.matrix import orthogonalize_w
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def _dummy_matrix(groups: np.ndarray, *, drop_first: bool = False) -> np.ndarray:
    levels = np.unique(groups)
    dummies = np.equal.outer(groups, levels).astype(float)
    return dummies[:, 1:] if drop_first else dummies


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


def test_orthogonalize_w_one_way_fixef_matches_explicit_dummies() -> None:
    rng = np.random.default_rng(20261118)
    n = 24
    fe = np.repeat(np.arange(1, 7), 4)
    fe_labels = np.array([f"g{val}" for val in fe])
    w_raw = rng.normal(size=(n, 4))
    w, _ = np.linalg.qr(w_raw)
    xj = rng.normal(size=n)
    xjs = np.sign(xj)
    model_mat = rng.normal(size=(n, 2)) + 0.5 * fe[:, None]

    expected = orthogonalize_w(
        w,
        xj,
        xjs,
        np.column_stack((model_mat, _dummy_matrix(fe))),
        include_intercept=False,
    )

    for include_intercept in (False, True):
        for fe_arg in (fe, fe_labels, pd.DataFrame({"fe": fe})):
            actual = orthogonalize_w(
                w,
                xj,
                xjs,
                model_mat,
                include_intercept=include_intercept,
                fixef_id=fe_arg,
            )
            npt.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_orthogonalize_w_two_way_fixef_matches_explicit_dummies() -> None:
    rng = np.random.default_rng(9102)
    n = 30
    fe1 = np.array([i % 5 for i in range(n)])
    fe2 = np.array([(i * 2 + (i // 5)) % 4 for i in range(n)])
    w, _ = np.linalg.qr(rng.normal(size=(n, 4)))
    xj = rng.normal(size=n)
    xjs = np.sign(xj)
    model_mat = rng.normal(size=(n, 2)) + 0.3 * fe1[:, None] - 0.2 * fe2[:, None]

    dummies = np.column_stack(
        (_dummy_matrix(fe1), _dummy_matrix(fe2, drop_first=True))
    )
    expected = orthogonalize_w(
        w,
        xj,
        xjs,
        np.column_stack((model_mat, dummies)),
        include_intercept=False,
    )

    for fe_arg in (
        np.column_stack((fe1, fe2)),
        pd.DataFrame({"fe1": fe1, "fe2": fe2}),
        {"fe1": fe1, "fe2": fe2},
    ):
        actual = orthogonalize_w(
            w,
            xj,
            xjs,
            model_mat,
            include_intercept=True,
            fixef_id=fe_arg,
        )
        npt.assert_allclose(actual, expected, atol=1e-10, rtol=0.0)


def test_orthogonalize_w_fixef_empty_model_mat_matches_explicit_dummies() -> None:
    rng = np.random.default_rng(77)
    n = 18
    fe = np.repeat(np.arange(6), 3)
    w, _ = np.linalg.qr(rng.normal(size=(n, 3)))
    xj = rng.normal(size=n)
    xjs = np.sign(xj)

    expected = orthogonalize_w(
        w,
        xj,
        xjs,
        _dummy_matrix(fe),
        include_intercept=False,
    )
    actual = orthogonalize_w(
        w,
        xj,
        xjs,
        np.empty((n, 0)),
        include_intercept=True,
        fixef_id=fe,
    )
    npt.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_orthogonalize_w_fixef_validation_and_rank_deficiency() -> None:
    w = np.eye(6, 2)
    xj = np.ones(6)
    xjs = np.ones(6)
    x = np.arange(6, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="row count"):
        orthogonalize_w(w, xj, xjs, x, fixef_id=[1, 1, 2])
    with pytest.raises(ValueError, match="1D or 2D"):
        orthogonalize_w(w, xj, xjs, x, fixef_id="bad")
    with pytest.raises(ValueError, match="missing or NaN"):
        orthogonalize_w(w, xj, xjs, x, fixef_id=[1.0, 1.0, np.nan, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="non-finite"):
        orthogonalize_w(w, xj, xjs, x, fixef_id=[1.0, 1.0, np.inf, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="rank deficient"):
        orthogonalize_w(
            w,
            xj,
            xjs,
            np.column_stack((np.ones(6), x)),
            include_intercept=False,
            fixef_id=[1, 1, 1, 2, 2, 2],
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


@pytest.mark.skipif(R is None, reason="Rscript not installed")
@pytest.mark.parametrize("include_intercept", [False, True])
def test_python_r_parity_orthogonalize_w_one_way_fixef(
    include_intercept: bool,
) -> None:
    rng = np.random.default_rng(142857)
    n = 20
    fe = np.repeat(np.arange(1, 6), 4)
    w, _ = np.linalg.qr(rng.normal(size=(n, 4)))
    xj = rng.normal(size=n)
    xjs = np.sign(xj)
    model_mat = rng.normal(size=(n, 2)) + 0.4 * fe[:, None]

    # Pinned scpcR v0.1.3 swapped xj/xjs inside .orthogonalize_W; invoke its
    # live .make_ols_residualizer with the corrected S_x M_V D_x order, plus
    # verify .orthogonalize_W directly when xj == xjs.
    r_value = execute_r_code(
        """
        W <- matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE)
        xj <- unlist(payload$xj)
        xjs <- unlist(payload$xjs)
        X <- matrix(unlist(payload$model_mat), nrow = length(payload$model_mat), byrow = TRUE)
        fe <- unlist(payload$fixef_id)
        residualize <- getFromNamespace(".make_ols_residualizer", "scpcR")(
          X,
          isTRUE(payload$include_intercept),
          fe
        )
        Wx <- W
        Wx[, 1] <- Wx[, 1] * xj * xjs
        Rx <- sweep(Wx[, -1, drop = FALSE], 1, xj, `*`)
        Rr <- residualize(Rx)
        Wx[, -1] <- sweep(Rr, 1, xjs, `*`)
        result <- Wx
        """,
        payload={
            "w": w.tolist(),
            "xj": xj.tolist(),
            "xjs": xjs.tolist(),
            "model_mat": model_mat.tolist(),
            "include_intercept": include_intercept,
            "fixef_id": fe.tolist(),
        },
    )
    py_value = orthogonalize_w(
        w,
        xj,
        xjs,
        model_mat,
        include_intercept=include_intercept,
        fixef_id=fe,
    )
    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)

    r_direct = execute_r_code(
        """
        result <- getFromNamespace(".orthogonalize_W", "scpcR")(
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE),
          unlist(payload$xjs),
          unlist(payload$xjs),
          matrix(unlist(payload$model_mat), nrow = length(payload$model_mat), byrow = TRUE),
          isTRUE(payload$include_intercept),
          unlist(payload$fixef_id)
        )
        """,
        payload={
            "w": w.tolist(),
            "xjs": xjs.tolist(),
            "model_mat": model_mat.tolist(),
            "include_intercept": include_intercept,
            "fixef_id": fe.tolist(),
        },
    )
    py_direct = orthogonalize_w(
        w,
        xjs,
        xjs,
        model_mat,
        include_intercept=include_intercept,
        fixef_id=fe,
    )
    npt.assert_allclose(py_direct, np.array(r_direct), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_orthogonalize_w_two_way_fixef() -> None:
    rng = np.random.default_rng(314159)
    n = 24
    fe1 = np.array([i % 4 + 1 for i in range(n)])
    fe2 = np.array([(i * 3 + (i // 4)) % 3 + 1 for i in range(n)])
    w, _ = np.linalg.qr(rng.normal(size=(n, 4)))
    xj = rng.normal(size=n)
    xjs = np.sign(xj)
    model_mat = rng.normal(size=(n, 2)) + 0.5 * fe1[:, None] - 0.3 * fe2[:, None]

    r_value = execute_r_code(
        """
        W <- matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE)
        xj <- unlist(payload$xj)
        xjs <- unlist(payload$xjs)
        X <- matrix(unlist(payload$model_mat), nrow = length(payload$model_mat), byrow = TRUE)
        fe <- list(fe1 = unlist(payload$fe1), fe2 = unlist(payload$fe2))
        residualize <- getFromNamespace(".make_ols_residualizer", "scpcR")(
          X,
          TRUE,
          fe
        )
        Wx <- W
        Wx[, 1] <- Wx[, 1] * xj * xjs
        Rx <- sweep(Wx[, -1, drop = FALSE], 1, xj, `*`)
        Rr <- residualize(Rx)
        Wx[, -1] <- sweep(Rr, 1, xjs, `*`)
        result <- Wx
        """,
        payload={
            "w": w.tolist(),
            "xj": xj.tolist(),
            "xjs": xjs.tolist(),
            "model_mat": model_mat.tolist(),
            "fe1": fe1.tolist(),
            "fe2": fe2.tolist(),
        },
    )
    py_value = orthogonalize_w(
        w,
        xj,
        xjs,
        model_mat,
        include_intercept=True,
        fixef_id=np.column_stack((fe1, fe2)),
    )

    npt.assert_allclose(py_value, np.array(r_value), atol=ATOL, rtol=RTOL)
