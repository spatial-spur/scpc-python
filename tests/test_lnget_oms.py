from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from scpc.utils.spatial import (
    InvalidSampledCovarianceError,
    lnget_oms,
    sample_covariance_pairs,
    validate_sampled_covariance,
)
from tests.config import ATOL, RTOL
from tests.utils import R, execute_r_code


def test_lnget_oms_returns_symmetric_sampled_omega_matrices() -> None:
    s = np.array([[0.0], [1.0], [2.0], [3.0]])
    w = np.array(
        [
            [0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [0.5, -0.5],
        ]
    )

    oms, random_t = lnget_oms(
        s,
        0.5,
        0.72,
        w,
        1.2,
        capM=20,
        random_t=1,
        latlong=False,
    )

    assert len(oms) == 3
    (indices, distances), _ = sample_covariance_pairs(s, 20, 1, False)
    cross = w[indices[:-1]].T @ (w[indices[1:]] * np.exp(-0.72 * distances)[:, None])
    np.testing.assert_allclose(oms[-1], w.T @ w + (6 / 20) * (cross + cross.T))
    for om in oms:
        assert om.shape == (2, 2)
        npt.assert_allclose(om, om.T, atol=1e-12, rtol=0.0)
    assert isinstance(random_t, int)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_lnget_oms() -> None:
    s = [[0.0], [1.0], [2.0], [3.0]]
    w = [
        [0.5, 0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [0.5, -0.5],
    ]

    r_value = execute_r_code(
        """
        out <- getFromNamespace(".lnget_Oms", "scpcR")(
          matrix(unlist(payload$s), nrow = length(payload$s), byrow = TRUE),
          payload$c0,
          payload$cmax,
          matrix(unlist(payload$w), nrow = length(payload$w), byrow = TRUE),
          payload$cgridfac,
          capM = payload$capM,
          random_t = payload$random_t,
          latlong = payload$latlong
        )
        result <- list(
          Oms = lapply(out$Oms, function(om) unname(split(om, row(om)))),
          state = out$state
        )
        """,
        payload={
            "s": s,
            "c0": 0.5,
            "cmax": 0.73,
            "w": w,
            "cgridfac": 1.2,
            "capM": 20,
            "random_t": 1,
            "latlong": False,
        },
    )
    py_oms, py_state = lnget_oms(
        np.array(s),
        0.5,
        0.73,
        np.array(w),
        1.2,
        capM=20,
        random_t=1,
        latlong=False,
    )

    # The release fixture covers the geometric candidates before the endpoint.
    released = r_value["Oms"]
    assert len(py_oms) == len(released) + 1
    for py_om, r_om in zip(py_oms[:-1], released, strict=True):
        npt.assert_allclose(py_om, np.array(r_om), atol=ATOL, rtol=RTOL)
    assert py_state == r_value["state"]


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_validate_sampled_covariance_uses_relative_scale(scale: float) -> None:
    validate_sampled_covariance(scale * np.ones((2, 2)), 0)
    indefinite = scale * np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(InvalidSampledCovarianceError, match="positive-semidefinite"):
        validate_sampled_covariance(indefinite, 0)


def test_validate_sampled_covariance_checks_both_triangles() -> None:
    with pytest.raises(InvalidSampledCovarianceError, match="grid point 2"):
        validate_sampled_covariance(np.array([[1.0, 0.5], [0.0, 1.0]]), 1)
