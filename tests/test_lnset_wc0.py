from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import lnset_wc0
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def assert_columns_allclose_up_to_sign(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    npt.assert_allclose(left[:, 0], right[:, 0], atol=ATOL, rtol=RTOL)
    for column in range(1, left.shape[1]):
        if np.allclose(left[:, column], right[:, column], atol=ATOL, rtol=RTOL):
            continue
        npt.assert_allclose(left[:, column], -right[:, column], atol=ATOL, rtol=RTOL)


def test_lnset_wc0_returns_a_coherent_large_n_setup() -> None:
    s = np.array([[0.0], [1.0], [2.0], [3.0]])

    w, c0, cmax, random_t = lnset_wc0(
        s,
        0.1,
        2,
        0.00001,
        latlong=False,
        capN=3,
        m=3,
        random_t=1,
    )

    assert w.shape == (4, 3)
    npt.assert_allclose(w[:, 0], np.full(4, 1 / math.sqrt(4)), atol=1e-12, rtol=0.0)
    assert c0 > 0
    assert cmax > c0
    assert isinstance(random_t, int)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_lnset_wc0() -> None:
    s = [[0.0], [1.0], [2.0], [3.0]]

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".lnset_wc0", "scpcR")(
          matrix(unlist(payload$s), nrow = length(payload$s), byrow = TRUE),
          payload$avc0,
          payload$qmax,
          payload$minavc,
          payload$latlong,
          capN = payload$capN,
          m = payload$m,
          random_t = payload$random_t
        )
        result <- list(
          W = unname(split(out$W, row(out$W))),
          c0 = out$c0,
          cmax = out$cmax,
          random_t = out$random_t
        )
        """,
        payload={
            "s": s,
            "avc0": 0.1,
            "qmax": 2,
            "minavc": 0.00001,
            "latlong": False,
            "capN": 3,
            "m": 3,
            "random_t": 1,
        },
    )
    py_w, py_c0, py_cmax, py_random_t = lnset_wc0(
        np.array(s),
        0.1,
        2,
        0.00001,
        latlong=False,
        capN=3,
        m=3,
        random_t=1,
    )

    assert_columns_allclose_up_to_sign(py_w, np.array(r_value["W"]))
    assert py_c0 == pytest.approx(r_value["c0"], abs=ATOL, rel=RTOL)
    assert py_cmax == pytest.approx(r_value["cmax"], abs=ATOL, rel=RTOL)
    assert py_random_t == r_value["random_t"]
