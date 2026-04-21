from __future__ import annotations

import pytest

from scpc.utils.spatial import next_u
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_next_u_advances_the_large_n_random_state() -> None:
    value, state = next_u(1)

    assert state == 64390
    assert value == pytest.approx(64390 / 2**32, abs=1e-18, rel=0.0)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_next_u() -> None:
    r_value = execute_r_code(
        """
        out <- getFromNamespace(".next_u", "scpcR")(payload$random_t)
        result <- list(value = out$value, state = out$state)
        """,
        payload={"random_t": 1},
    )
    py_value, py_state = next_u(1)

    assert py_value == pytest.approx(r_value["value"], abs=ATOL, rel=RTOL)
    assert py_state == r_value["state"]
