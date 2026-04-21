from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.types import SpatialSetup
from scpc.utils.spatial import set_oms_wfin
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def assert_columns_allclose_up_to_sign(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    npt.assert_allclose(left[:, 0], right[:, 0], atol=ATOL, rtol=RTOL)
    for column in range(1, left.shape[1]):
        if np.allclose(left[:, column], right[:, column], atol=ATOL, rtol=RTOL):
            continue
        npt.assert_allclose(left[:, column], -right[:, column], atol=ATOL, rtol=RTOL)


def get_column_signs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    signs = np.ones(left.shape[1])
    for column in range(1, left.shape[1]):
        if not np.allclose(left[:, column], right[:, column], atol=ATOL, rtol=RTOL):
            signs[column] = -1.0
    return signs


def test_set_oms_wfin_returns_a_coherent_exact_spatial_setup() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ]
    )

    result = set_oms_wfin(coords, 0.1, latlong=False, method="exact", large_n_seed=1)

    assert isinstance(result, SpatialSetup)
    assert result.wfin.shape[0] == coords.shape[0]
    assert result.wfin.shape[1] >= 2
    assert result.cvfin > 0
    assert result.c0 > 0
    assert result.cmax > result.c0
    assert len(result.omsfin) >= 2
    assert result.distmat is not None
    assert result.method == "exact"
    assert result.large_n is False
    npt.assert_array_equal(result.perm, np.array([0, 1, 2]))
    npt.assert_allclose(result.coords, coords, atol=1e-12, rtol=0.0)
    assert result.random_state is None


def test_set_oms_wfin_returns_a_coherent_approximate_spatial_setup() -> None:
    coords = np.array([[0.0], [1.0], [2.0], [3.0]])

    result = set_oms_wfin(coords, 0.1, latlong=False, method="approx", large_n_seed=1)

    assert isinstance(result, SpatialSetup)
    assert result.wfin.shape[0] == 4
    assert result.wfin.shape[1] >= 2
    assert result.cvfin > 0
    assert result.c0 > 0
    assert result.cmax > result.c0
    assert len(result.omsfin) >= 2
    assert result.distmat is None
    assert result.method == "approx"
    assert result.large_n is True
    assert result.random_state is not None
    assert np.array_equal(np.sort(result.perm), np.arange(4))


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_set_oms_wfin_exact() -> None:
    coords = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ]

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".setOmsWfin", "scpcR")(
          matrix(unlist(payload$coords), nrow = length(payload$coords), byrow = TRUE),
          payload$avc0,
          payload$latlong,
          method = payload$method,
          large_n_seed = payload$large_n_seed
        )
        result <- list(
          wfin = unname(split(out$Wfin, row(out$Wfin))),
          cvfin = out$cvfin,
          omsfin = lapply(out$Omsfin, function(om) unname(split(om, row(om)))),
          c0 = out$c0,
          cmax = out$cmax,
          coords = unname(split(out$coords, row(out$coords))),
          perm = out$perm,
          distmat = unname(split(out$distmat, row(out$distmat))),
          method = out$method,
          large_n = out$large_n,
          random_state = out$random_state
        )
        """,
        payload={
            "coords": coords,
            "avc0": 0.1,
            "latlong": False,
            "method": "exact",
            "large_n_seed": 1,
        },
    )
    py_value = set_oms_wfin(
        np.array(coords), 0.1, latlong=False, method="exact", large_n_seed=1
    )

    assert py_value.cvfin == pytest.approx(r_value["cvfin"], abs=ATOL, rel=RTOL)
    assert py_value.c0 == pytest.approx(r_value["c0"], abs=ATOL, rel=RTOL)
    assert py_value.cmax == pytest.approx(r_value["cmax"], abs=ATOL, rel=RTOL)
    assert py_value.method == r_value["method"]
    assert py_value.large_n is r_value["large_n"]
    assert py_value.random_state == r_value["random_state"]
    npt.assert_array_equal(py_value.perm, np.array(r_value["perm"]) - 1)
    assert py_value.distmat is not None
    npt.assert_allclose(
        py_value.coords, np.array(r_value["coords"]), atol=ATOL, rtol=RTOL
    )
    npt.assert_allclose(
        py_value.distmat, np.array(r_value["distmat"]), atol=ATOL, rtol=RTOL
    )
    r_wfin = np.array(r_value["wfin"])
    assert_columns_allclose_up_to_sign(py_value.wfin, r_wfin)
    assert len(py_value.omsfin) == len(r_value["omsfin"])
    signs = get_column_signs(py_value.wfin, r_wfin)
    sign_matrix = np.diag(signs)
    for py_om, r_om in zip(py_value.omsfin, r_value["omsfin"], strict=True):
        npt.assert_allclose(
            py_om,
            sign_matrix @ np.array(r_om) @ sign_matrix,
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_set_oms_wfin_approx() -> None:
    coords = [[0.0], [1.0], [2.0], [3.0]]

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".setOmsWfin", "scpcR")(
          matrix(unlist(payload$coords), nrow = length(payload$coords), byrow = TRUE),
          payload$avc0,
          payload$latlong,
          method = payload$method,
          large_n_seed = payload$large_n_seed
        )
        result <- list(
          wfin = unname(split(out$Wfin, row(out$Wfin))),
          cvfin = out$cvfin,
          omsfin = lapply(out$Omsfin, function(om) unname(split(om, row(om)))),
          c0 = out$c0,
          cmax = out$cmax,
          coords = unname(split(out$coords, row(out$coords))),
          perm = out$perm,
          distmat = out$distmat,
          method = out$method,
          large_n = out$large_n,
          random_state = out$random_state
        )
        """,
        payload={
            "coords": coords,
            "avc0": 0.1,
            "latlong": False,
            "method": "approx",
            "large_n_seed": 1,
        },
    )
    py_value = set_oms_wfin(
        np.array(coords), 0.1, latlong=False, method="approx", large_n_seed=1
    )

    assert py_value.cvfin == pytest.approx(r_value["cvfin"], abs=ATOL, rel=RTOL)
    assert py_value.c0 == pytest.approx(r_value["c0"], abs=ATOL, rel=RTOL)
    assert py_value.cmax == pytest.approx(r_value["cmax"], abs=ATOL, rel=RTOL)
    assert py_value.method == r_value["method"]
    assert py_value.large_n is r_value["large_n"]
    assert py_value.distmat is None
    assert py_value.random_state == r_value["random_state"]
    npt.assert_array_equal(py_value.perm, np.array(r_value["perm"]) - 1)
    r_coords = np.array(r_value["coords"])
    if r_coords.ndim == 1:
        r_coords = r_coords.reshape(-1, 1)
    npt.assert_allclose(py_value.coords, r_coords, atol=ATOL, rtol=RTOL)
    r_wfin = np.array(r_value["wfin"])
    assert_columns_allclose_up_to_sign(py_value.wfin, r_wfin)
    assert len(py_value.omsfin) == len(r_value["omsfin"])
    signs = get_column_signs(py_value.wfin, r_wfin)
    sign_matrix = np.diag(signs)
    for py_om, r_om in zip(py_value.omsfin, r_value["omsfin"], strict=True):
        npt.assert_allclose(
            py_om,
            sign_matrix @ np.array(r_om) @ sign_matrix,
            atol=ATOL,
            rtol=RTOL,
        )
