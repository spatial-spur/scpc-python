from __future__ import annotations

import math

import numpy as np
import pytest

from scpc.utils.spatial import get_avc
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_avc_matches_exponential_average() -> None:
    c = math.log(2.0)
    dist = np.array([1.0, 2.0])

    assert get_avc(c, dist) == pytest.approx(0.375, abs=1e-12)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_avc() -> None:
    c = 0.03
    dist = [0.1, 0.2, 0.4]

    r_value = execute_r_code(
        'result <- getFromNamespace(".getavc", "scpcR")(payload$c, unlist(payload$dist))',
        payload={"c": c, "dist": dist},
    )
    py_value = get_avc(c, np.array(dist))

    assert py_value == pytest.approx(r_value, abs=ATOL, rel=RTOL)
