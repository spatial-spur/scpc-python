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


def test_set_oms_wfin_returns_a_coherent_spatial_setup() -> None:
    distmat = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    result = set_oms_wfin(distmat, 0.1)

    assert isinstance(result, SpatialSetup)
    assert result.wfin.shape[0] == distmat.shape[0]
    assert result.wfin.shape[1] >= 2
    assert result.cvfin > 0
    assert result.c0 > 0
    assert result.cmax > result.c0
    assert len(result.omsfin) >= 2


@pytest.mark.skipif(R is None, reason="Rscript not installed")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "scpcR v0.1.0b1 changed .setOmsWfin to take coords + latlong and "
        "to support the newer large-n branch; Python still ports the older "
        "distance-matrix exact helper"
    ),
)
def test_python_r_parity_set_oms_wfin() -> None:
    distmat = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ]
    avc0 = 0.1

    # TODO: this parity test is intentionally pinned to the older r helper contract.
    # when python is upgraded to match the current public scpcr interface and
    # the >3500 observation branch, remove this xfail and rewrite the r call.
    r_value = execute_r_code(
        """
        out <- getFromNamespace(".setOmsWfin", "scpcR")(
          matrix(unlist(payload$distmat), nrow = length(payload$distmat), byrow = TRUE),
          payload$avc0
        )
        result <- list(
          wfin = unname(split(out$Wfin, row(out$Wfin))),
          cvfin = out$cvfin,
          omsfin = lapply(out$Omsfin, function(om) unname(split(om, row(om)))),
          c0 = out$c0,
          cmax = out$cmax,
          q = ncol(out$Wfin) - 1
        )
        """,
        payload={"distmat": distmat, "avc0": avc0},
    )
    py_value = set_oms_wfin(np.array(distmat), avc0)

    assert py_value.cvfin == pytest.approx(r_value["cvfin"], abs=ATOL, rel=RTOL)
    assert py_value.c0 == pytest.approx(r_value["c0"], abs=ATOL, rel=RTOL)
    assert py_value.cmax == pytest.approx(r_value["cmax"], abs=ATOL, rel=RTOL)
    assert py_value.wfin.shape[1] - 1 == r_value["q"]
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
