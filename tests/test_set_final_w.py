from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from scipy import stats

from scpc.utils.spatial import set_final_w
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_set_final_w_returns_the_only_admissible_projection_when_qmax_is_one() -> None:
    oms = [np.zeros((2, 2)), np.zeros((2, 2))]
    w = np.array([[1.0, 0.0], [0.0, 1.0]])

    selected_w, cv, q = set_final_w(oms, w, 1)

    npt.assert_allclose(selected_w, w, atol=1e-12, rtol=0.0)
    assert cv == pytest.approx(stats.t.ppf(0.975, df=1), abs=1e-10)
    assert q == 1


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_set_final_w() -> None:
    oms = [
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]
    w = [[1.0, 0.0], [0.0, 1.0]]

    r_value = execute_r_code(
        """
        oms <- lapply(
          payload$oms,
          function(om) matrix(unlist(om), nrow = length(om), byrow = TRUE)
        )
        w <- matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE)
        out <- getFromNamespace(".setfinalW", "scpcR")(oms, w, payload$qmax)
        result <- list(w = unname(split(out$W, row(out$W))), cv = out$cv, q = out$q)
        """,
        payload={"oms": oms, "w": w, "qmax": 1},
    )
    py_w, py_cv, py_q = set_final_w([np.array(om) for om in oms], np.array(w), 1)

    npt.assert_allclose(py_w, np.array(r_value["w"]), atol=ATOL, rtol=RTOL)
    assert py_cv == pytest.approx(r_value["cv"], abs=ATOL, rel=RTOL)
    assert py_q == r_value["q"]
