from __future__ import annotations

import numpy as np
import pytest

from scpc.utils.spatial import get_rp
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_rp_is_zero_for_the_zero_omega_matrix() -> None:
    om = np.zeros((2, 2))

    assert get_rp(om, 2.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_rp() -> None:
    om = np.array([[1.0, 0.2], [0.2, 0.5]])
    cv = 1.5

    r_value = execute_r_code(
        'result <- getFromNamespace(".getrp", "scpcR")(matrix(unlist(payload$om), nrow = length(payload$om), byrow = TRUE), payload$cv)',
        payload={"om": om.tolist(), "cv": cv},
    )
    py_value = get_rp(om, cv)

    assert py_value == pytest.approx(r_value, abs=ATOL, rel=RTOL)
