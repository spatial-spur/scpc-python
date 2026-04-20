from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import get_oms
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_oms_builds_the_expected_omega_grid() -> None:
    distmat = np.array([[0.0, 1.0], [1.0, 0.0]])
    w = np.eye(2)
    c0 = math.log(2.0)
    cmax = 2 * c0
    cgridfac = 2.0

    oms = get_oms(distmat, c0, cmax, w, cgridfac)

    assert len(oms) == 2
    npt.assert_allclose(
        oms[0], np.array([[1.0, 0.5], [0.5, 1.0]]), atol=1e-12, rtol=0.0
    )
    npt.assert_allclose(
        oms[1], np.array([[1.0, 0.25], [0.25, 1.0]]), atol=1e-12, rtol=0.0
    )


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_oms() -> None:
    distmat = [[0.0, 1.0], [1.0, 0.0]]
    w = [[1.0, 0.0], [0.0, 1.0]]
    c0 = math.log(2.0)
    cmax = 2 * c0
    cgridfac = 2.0

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".getOms", "scpcR")(
          matrix(unlist(payload$distmat), nrow = length(payload$distmat), byrow = TRUE),
          payload$c0,
          payload$cmax,
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE),
          payload$cgridfac
        )
        result <- lapply(out, function(om) unname(split(om, row(om))))
        """,
        payload={
            "distmat": distmat,
            "c0": c0,
            "cmax": cmax,
            "w": w,
            "cgridfac": cgridfac,
        },
    )
    py_value = get_oms(np.array(distmat), c0, cmax, np.array(w), cgridfac)

    assert len(py_value) == len(r_value)
    for py_om, r_om in zip(py_value, r_value, strict=True):
        npt.assert_allclose(py_om, np.array(r_om), atol=ATOL, rtol=RTOL)
