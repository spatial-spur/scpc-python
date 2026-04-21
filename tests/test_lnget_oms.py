from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import get_nc, lnget_oms
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_lnget_oms_returns_symmetric_sampled_omega_matrices() -> None:
    s = np.array([[0.0], [1.0], [2.0], [3.0]])
    w = np.array(
        [
            [0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [0.5, -0.5],
        ]
    )

    oms, random_t = lnget_oms(
        s,
        0.5,
        0.72,
        w,
        1.2,
        capM=8,
        random_t=1,
        latlong=False,
    )

    assert len(oms) == get_nc(0.5, 0.72, 1.2)
    for om in oms:
        assert om.shape == (2, 2)
        npt.assert_allclose(om, om.T, atol=1e-12, rtol=0.0)
    assert isinstance(random_t, int)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_lnget_oms() -> None:
    s = [[0.0], [1.0], [2.0], [3.0]]
    w = [
        [0.5, 0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [0.5, -0.5],
    ]

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".lnget_Oms", "scpcR")(
          matrix(unlist(payload$s), nrow = length(payload$s), byrow = TRUE),
          payload$c0,
          payload$cmax,
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE),
          payload$cgridfac,
          capM = payload$capM,
          random_t = payload$random_t,
          latlong = payload$latlong
        )
        result <- list(
          Oms = lapply(out$Oms, function(om) unname(split(om, row(om)))),
          state = out$state
        )
        """,
        payload={
            "s": s,
            "c0": 0.5,
            "cmax": 0.72,
            "w": w,
            "cgridfac": 1.2,
            "capM": 8,
            "random_t": 1,
            "latlong": False,
        },
    )
    py_oms, py_state = lnget_oms(
        np.array(s),
        0.5,
        0.72,
        np.array(w),
        1.2,
        capM=8,
        random_t=1,
        latlong=False,
    )

    assert len(py_oms) == len(r_value["Oms"])
    for py_om, r_om in zip(py_oms, r_value["Oms"], strict=True):
        npt.assert_allclose(py_om, np.array(r_om), atol=ATOL, rtol=RTOL)
    assert py_state == r_value["state"]
