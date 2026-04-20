from __future__ import annotations

import numpy as np
import pytest

from scpc.utils.spatial import max_rp
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_max_rp_returns_the_first_zero_case_when_all_cases_match() -> None:
    oms = [np.zeros((2, 2)), np.zeros((2, 2))]

    max_value, max_index = max_rp(oms, 1, 2.0)

    assert max_value == pytest.approx(0.0, abs=1e-12)
    assert max_index == 0


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_max_rp() -> None:
    oms = [
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.2], [0.2, 0.5]],
    ]
    q = 1
    cv = 1.5

    r_value = execute_r_code(
        """
        oms <- lapply(payload$oms, function(om) matrix(unlist(om), nrow = length(om), byrow = TRUE))
        out <- getFromNamespace(".maxrp", "scpcR")(oms, payload$q, payload$cv)
        result <- list(max = out$max, i = out$i)
        """,
        payload={"oms": oms, "q": q, "cv": cv},
    )
    py_max, py_index = max_rp([np.array(om) for om in oms], q, cv)

    assert py_max == pytest.approx(r_value["max"], abs=ATOL, rel=RTOL)
    assert py_index + 1 == r_value["i"]
