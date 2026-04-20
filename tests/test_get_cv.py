from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from scpc.utils.spatial import get_cv
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_cv_matches_the_t_critical_value_for_zero_omega_matrices() -> None:
    oms = [np.zeros((2, 2)), np.zeros((2, 2))]
    q = 1
    level = 0.05

    expected = stats.t.ppf(1 - level / 2, df=q)

    assert get_cv(oms, q, level) == pytest.approx(expected, abs=1e-10)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_cv() -> None:
    oms = [
        [[1.0, 0.1], [0.1, 0.8]],
        [[0.9, 0.05], [0.05, 0.6]],
    ]
    q = 1
    level = 0.05

    r_value = execute_r_code(
        """
        oms <- lapply(
          payload$oms,
          function(om) matrix(unlist(om), nrow = length(om), byrow = TRUE)
        )
        result <- getFromNamespace(".getcv", "scpcR")(oms, payload$q, payload$level)
        """,
        payload={"oms": oms, "q": q, "level": level},
    )
    py_value = get_cv([np.array(om) for om in oms], q, level)

    assert py_value == pytest.approx(r_value, abs=ATOL, rel=RTOL)
