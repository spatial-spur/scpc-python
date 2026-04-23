from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from scpc.types import SCPCResult


def make_result(*, cvs: bool = False) -> SCPCResult:
    return SCPCResult(
        scpcstats=np.array(
            [
                [1.0, 0.5, 2.0, 0.05, 0.0, 2.0],
                [-1.0, 0.25, -4.0, 0.01, -1.5, -0.5],
            ]
        ),
        scpccvs=np.array(
            [
                [1.0, 1.5, 2.0, 3.0],
                [1.1, 1.6, 2.1, 3.1],
            ]
        )
        if cvs
        else None,
        w=np.ones((3, 2)),
        avc=0.03,
        c0=1.2,
        cv=2.0,
        q=1,
        coef_names=["Intercept", "x"],
        method="exact",
        large_n_seed=1,
    )


def test_repr_returns_compact_result_description() -> None:
    assert repr(make_result()) == "SCPCResult(ncoef=2, q=1, avc=0.03, method='exact')"


def test_str_returns_main_inference_table() -> None:
    text = str(make_result())

    assert "SCPC Inference (ncoef = 2, q = 1)" in text
    assert "Intercept" in text
    assert "P>|t|" in text
    assert "95% Confidence Intervals" not in text


def test_summary_returns_main_table_and_confidence_intervals() -> None:
    text = make_result(cvs=True).summary()

    assert "SCPC Inference (ncoef = 2, q = 1, avc = 0.03)" in text
    assert "95% Confidence Intervals:" in text
    assert "Two-sided critical values:" in text


def test_coef_returns_named_coefficient_series() -> None:
    pdt.assert_series_equal(
        make_result().coef(),
        pd.Series([1.0, -1.0], index=["Intercept", "x"], name="Coef"),
    )


def test_confint_returns_stored_95_percent_intervals() -> None:
    pdt.assert_frame_equal(
        make_result().confint(),
        pd.DataFrame(
            [[0.0, 2.0], [-1.5, -0.5]],
            index=["Intercept", "x"],
            columns=["2.5 %", "97.5 %"],
        ),
    )


def test_confint_selects_by_name_and_index() -> None:
    expected = pd.DataFrame(
        [[-1.5, -0.5]],
        index=["x"],
        columns=["2.5 %", "97.5 %"],
    )

    pdt.assert_frame_equal(make_result().confint(parm="x"), expected)
    pdt.assert_frame_equal(make_result().confint(parm=1), expected)


def test_confint_uses_stored_critical_values_for_non_95_levels() -> None:
    pdt.assert_frame_equal(
        make_result(cvs=True).confint(parm=["x"], level=0.90),
        pd.DataFrame([[-1.4, -0.6]], index=["x"], columns=["5 %", "95 %"]),
    )


def test_confint_requires_stored_critical_values_for_non_95_levels() -> None:
    with pytest.raises(ValueError, match="not available"):
        make_result().confint(level=0.90)
