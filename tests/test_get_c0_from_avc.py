from __future__ import annotations

import math

import numpy as np
import pytest

from scpc.utils.spatial import get_avc, get_c0_from_avc
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_get_c0_from_avc_inverts_the_target_average_correlation() -> None:
    dist = np.array([1.0, 2.0])
    avc0 = 0.375

    c0 = get_c0_from_avc(dist, avc0)

    assert c0 > 0
    assert c0 == pytest.approx(math.log(2.0), abs=1e-3)
    assert get_avc(c0, dist) == pytest.approx(avc0, abs=3e-4)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_c0_from_avc() -> None:
    dist = [0.1, 0.2, 0.4, 0.8]
    avc0 = 0.2

    r_value = execute_r_code(
        'result <- getFromNamespace(".getc0fromavc", "scpcR")(unlist(payload$dist), payload$avc0)',
        payload={"dist": dist, "avc0": avc0},
    )
    py_value = get_c0_from_avc(np.array(dist), avc0)

    assert py_value == pytest.approx(r_value, abs=ATOL, rel=RTOL)
