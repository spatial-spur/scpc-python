from __future__ import annotations

import pytest

from scpc.utils.spatial import get_nc
from tests.utils import R, execute_r_code


def test_get_nc_matches_the_closed_form_grid_count() -> None:
    assert get_nc(1.0, 4.0, 2.0) == 2


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_nc() -> None:
    c0 = 0.5
    cmax = 10.0
    cgridfac = 1.2

    r_value = execute_r_code(
        'result <- getFromNamespace(".getnc", "scpcR")(payload$c0, payload$cmax, payload$cgridfac)',
        payload={"c0": c0, "cmax": cmax, "cgridfac": cgridfac},
    )
    py_value = get_nc(c0, cmax, cgridfac)

    assert py_value == r_value
