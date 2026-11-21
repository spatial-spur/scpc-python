from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from scpc.utils.matrix import (
    make_fixef_demeaner,
    make_ols_residualizer,
    orthogonalize_w,
)
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def _dummy_matrix(groups: np.ndarray, *, drop_first: bool = False) -> np.ndarray:
    levels = np.unique(groups)
    dummies = np.equal.outer(groups, levels).astype(float)
    return dummies[:, 1:] if drop_first else dummies


def test_absorbed_ols_preserves_representable_variation_near_large_anchor() -> None:
    fe = np.repeat([0, 1, 2], 3)
    x = np.array([0, 1, 2, 0, 2, 1, 1, 0, 2], dtype=float)[:, None]
    demean = make_fixef_demeaner(fe, len(fe))
    values = np.arange(9.0) ** 2
    expected = make_ols_residualizer(x, demean=demean, require_full_rank=True)(values)
    actual = make_ols_residualizer(x + 2.0**52, demean=demean, require_full_rank=True)(
        values
    )
    npt.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_one_way_absorption_preserves_extreme_contrasts_in_each_group() -> None:
    maximum = np.finfo(float).max
    previous = np.nextafter(maximum, 0.0)
    spacing = maximum - previous
    smallest = np.finfo(float).smallest_subnormal
    fe = np.tile([0, 1, 2], 3)
    values = np.array(
        [
            maximum,
            1e-300,
            smallest,
            previous,
            2e-300,
            2 * smallest,
            np.nextafter(previous, 0.0),
            3e-300,
            3 * smallest,
        ]
    )
    result = make_fixef_demeaner(fe, len(fe))(
        values, context="one-way fixed effects", drop_absorbed=True
    )
    npt.assert_array_equal(result[fe == 0], [spacing, 0, -spacing])
    npt.assert_allclose(
        result[fe == 1], [-1e-300, 0, 1e-300], atol=np.spacing(2e-300), rtol=1e-14
    )
    npt.assert_array_equal(result[fe == 2], [-smallest, 0, smallest])


def test_two_way_absorption_preserves_large_offsets_in_either_effect() -> None:
    fe = np.column_stack((np.repeat(np.arange(4), 5), np.tile(np.arange(5), 4)))
    rng = np.random.default_rng(1143)
    x = rng.integers(-3, 4, size=20)
    values = rng.normal(size=(20, 2))
    dummies = np.column_stack([_dummy_matrix(fe[:, i]) for i in range(2)])
    residualize = make_ols_residualizer(np.column_stack((x, dummies)))
    expected = residualize(values)
    design = np.column_stack((x + 1e15 * fe[:, 0], x + 1e15 * fe[:, 1]))
    for order in (fe, fe[:, ::-1]):
        demean = make_fixef_demeaner(order, len(fe))
        for regressor in design.T:
            actual = make_ols_residualizer(
                regressor, demean=demean, require_full_rank=True
            )(values)
            npt.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_two_way_absorption_refines_a_banded_panel_with_large_mixed_offsets() -> None:
    group = np.arange(1, 81)
    fe = np.column_stack(
        (np.r_[group, group[:-1], group], np.r_[group, group[1:], group])
    )
    contrast = np.resize([1, 2], len(group))
    x = np.r_[contrast, np.zeros(len(group) - 1), -contrast]
    dummies = np.column_stack([_dummy_matrix(fe[:, i]) for i in range(2)])
    y = np.sin(np.arange(len(fe)))
    expected = make_ols_residualizer(np.column_stack((x, dummies)))(y)
    x += 2.0**42 * fe.sum(axis=1)
    for order in (fe, fe[:, ::-1]):
        demean = make_fixef_demeaner(order, len(fe))
        actual = make_ols_residualizer(x, demean=demean, require_full_rank=True)(y)
        npt.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        with pytest.raises(ValueError, match="rank deficient"):
            make_ols_residualizer(
                fe[:, 0] + 0.75 * fe[:, 1], demean=demean, require_full_rank=True
            )


def test_one_way_absorption_rejects_sub_ulp_leakage_as_a_design_direction() -> None:
    fe = np.repeat([1.0, 2.0, 3.0], 4)
    y = np.tile([0.4, -0.75, 0.8, 1.25], 3)
    rounded = (fe + y) - y
    assert not np.array_equal(rounded, fe)
    demean = make_fixef_demeaner(fe, len(fe))
    with pytest.raises(ValueError, match="rank deficient"):
        make_ols_residualizer(rounded, demean=demean, require_full_rank=True)


def test_ols_residualizer_accepts_an_integer_design() -> None:
    x = np.array([[1, 0], [1, 1], [1, 2]])
    y = np.array([1, 0, 1])
    actual = make_ols_residualizer(x)(y)
    npt.assert_allclose(actual, [1 / 3, -2 / 3, 1 / 3], rtol=0, atol=1e-12)


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

    dummies = np.column_stack((_dummy_matrix(fe1), _dummy_matrix(fe2, drop_first=True)))
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


def test_orthogonalize_w_multiway_absorption_detects_collapse() -> None:
    groups = np.arange(75)
    g1 = np.repeat(groups, 4)
    g2 = np.repeat(np.column_stack((groups, groups + 1)).ravel(), 2)
    n = len(g1)
    values = np.random.default_rng(95).normal(size=len(groups) + 1)[g2]
    w = np.column_stack((np.ones(n), values))
    with pytest.raises(ValueError, match="collapsed"):
        orthogonalize_w(
            w,
            np.ones(n),
            np.ones(n),
            np.empty((n, 0)),
            fixef_id=[list(row) for row in zip(g1, g2)],
        )


def test_fixef_demeaner_handles_binary_ulp_single_group_offset_and_balanced_2way() -> (
    None
):
    # 1. Binary 0/1 contrast shifted by 2^52 (1 ULP peak-to-peak, 0.5 ULP demeaned amplitude)
    fe_bin = np.array([0, 0, 1, 1, 2, 2])
    x_bin = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])[:, None]
    demean_bin = make_fixef_demeaner(fe_bin, len(fe_bin))
    vals_bin = np.arange(6.0) ** 2
    exp_bin = make_ols_residualizer(x_bin, demean=demean_bin, require_full_rank=True)(
        vals_bin
    )
    act_bin = make_ols_residualizer(
        x_bin + 2.0**52, demean=demean_bin, require_full_rank=True
    )(vals_bin)
    npt.assert_allclose(act_bin, exp_bin, rtol=0, atol=1e-12)

    # 2. Single-group huge offset (1e15 in group 0, 0.001*z in groups 1..9)
    rng = np.random.default_rng(42)
    fe_sg = np.repeat(np.arange(10), 10)
    z = rng.normal(size=100)
    x_sg = (0.001 * z * (fe_sg > 0) + 1e15 * (fe_sg == 0))[:, None]
    x_clean = (0.001 * z * (fe_sg > 0))[:, None]
    demean_sg = make_fixef_demeaner(fe_sg, len(fe_sg))
    vals_sg = rng.normal(size=100)
    exp_sg = make_ols_residualizer(x_clean, demean=demean_sg, require_full_rank=True)(
        vals_sg
    )
    act_sg = make_ols_residualizer(x_sg, demean=demean_sg, require_full_rank=True)(
        vals_sg
    )
    npt.assert_allclose(act_sg, exp_sg, rtol=0, atol=1e-12)

    # 3. Balanced 2-way FE with large offsets in both factors (1e12 * g + 1e6 * h)
    g = np.repeat(np.arange(5), 5)
    h = np.tile(np.arange(5), 5)
    c = ((g - 2) * (h - 2)).astype(float)
    x_2w = (c + 1e12 * g + 1e6 * h)[:, None]
    demean_2w = make_fixef_demeaner(np.column_stack([g, h]), 25)
    res_2w = demean_2w(x_2w, context="test", drop_absorbed=True)
    npt.assert_allclose(res_2w[:, 0], c, rtol=0, atol=1e-9)


def test_cluster_direction_normalization_handles_extreme_scales() -> None:
    from scpc.utils.matrix import (
        make_iv_residualizer,
        orthogonalize_w_cluster,
        orthogonalize_w_cluster_iv,
    )

    rng = np.random.default_rng(199)
    nclust = 6
    cl_idx = np.repeat(np.arange(nclust), 3)
    n = len(cl_idx)
    w, _ = np.linalg.qr(rng.normal(size=(nclust, 3)))
    x = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 2)) + 0.5 * x
    xj_base = rng.normal(size=n)
    iv_res = make_iv_residualizer(x, z, transpose=True)

    base_ols = orthogonalize_w_cluster(w, cl_idx, xj_base, x)
    base_iv = orthogonalize_w_cluster_iv(w, cl_idx, xj_base, iv_res)
    for scale in (1e200, 1e-200):
        scaled_ols = orthogonalize_w_cluster(w, cl_idx, xj_base * scale, x)
        scaled_iv = orthogonalize_w_cluster_iv(w, cl_idx, xj_base * scale, iv_res)
        npt.assert_allclose(scaled_ols / scale, base_ols, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(scaled_iv / scale, base_iv, rtol=1e-12, atol=1e-12)
