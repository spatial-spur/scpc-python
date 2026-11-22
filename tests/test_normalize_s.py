from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import normalize_s
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_normalize_s_normalizes_one_dimensional_euclidean_coords() -> None:
    s = np.array([[3.0], [1.0], [2.0]])

    coords, perm = normalize_s(s, latlong=False)

    npt.assert_allclose(coords, np.array([[0.0], [0.5], [1.0]]), atol=1e-12, rtol=0.0)
    npt.assert_array_equal(perm, np.array([1, 2, 0]))


def test_normalize_s_centers_large_coordinates_before_summing() -> None:
    base = np.array([[-1.5, 0.25], [0.0, -0.75], [2.0, 1.5], [-0.5, 2.25]])
    expected, expected_perm = normalize_s(base, latlong=False)
    actual, perm = normalize_s(1e308 + 1e306 * base, latlong=False)
    npt.assert_array_equal(perm, expected_perm)
    npt.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
@pytest.mark.parametrize(
    ("coords", "latlong"),
    [
        (
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            False,
        ),
        (
            [
                [170.0, 10.0],
                [-175.0, -5.0],
                [160.0, 0.0],
                [-165.0, 20.0],
            ],
            True,
        ),
    ],
)
def test_python_r_parity_normalize_s(coords: list[list[float]], latlong: bool) -> None:
    r_value = execute_r_code(
        """
        out <- getFromNamespace(".normalize_s", "scpcR")(
          matrix(unlist(payload$coords), nrow = length(payload$coords), byrow = TRUE),
          payload$latlong
        )
        result <- list(
          coords = unname(split(out$coords, row(out$coords))),
          perm = out$perm
        )
        """,
        payload={"coords": coords, "latlong": latlong},
    )
    py_coords, py_perm = normalize_s(np.array(coords), latlong=latlong)

    npt.assert_allclose(py_coords, np.array(r_value["coords"]), atol=ATOL, rtol=RTOL)
    npt.assert_array_equal(py_perm, np.array(r_value["perm"]) - 1)
