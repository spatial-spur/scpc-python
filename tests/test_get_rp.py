from __future__ import annotations

import numpy as np
import pytest

from scpc.utils.spatial import get_rp
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_rp_is_zero_for_the_zero_omega_matrix() -> None:
    om = np.zeros((2, 2))

    assert get_rp(om, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_strict_gaussian_rejection_at_and_near_a_one_factor_tie() -> None:
    for r in (1.0 - 5e-15, -1.0 + np.finfo(float).eps):
        om = np.array([[1.0, r], [r, 1.0]])
        assert get_rp(om, 1.0) == 0.5
        assert get_rp(np.pad(om, (0, 1)), 1.0) == 0.5
    for loadings in ([1, -1], [1, 1], [5, 3, 4]):
        om = np.outer(loadings, loadings)
        assert get_rp(om, 1.0) == 0.0
        assert get_rp(om, 1.0 - 1e-15) == 1.0
    for scale in (2.0**-900, 1.0, 2.0**900):
        a, b = 2401 * scale, 49 * scale
        assert get_rp([[a, b], [b, scale]], 49.0) == 0
        near = np.nextafter(b, 0.0)
        assert get_rp([[a, near], [near, scale]], 49.0) == 0.5
    near = np.outer([5.0, 3.0, 4.0], [5.0, 3.0, 4.0])
    near[0, 0] = np.nextafter(25.0, np.inf)
    assert get_rp(near, 1.0) == pytest.approx(0.5, abs=2e-7)
    with pytest.raises(ValueError, match="positive semidefinite"):
        get_rp([[0, 1], [1, 0]], 1.0)


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
