from __future__ import annotations

import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf

from scpc.utils.data import get_coef_names
from tests.utils import make_basic_iv_data, make_one_way_fe_iv_data


def test_get_coef_names_returns_statsmodels_names() -> None:
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = smf.ols("y ~ x", data=data).fit()

    assert get_coef_names(model) == ["Intercept", "x"]


def test_get_coef_names_returns_pyfixest_names() -> None:
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = pf.feols("y ~ x", data=data)

    assert get_coef_names(model) == ["Intercept", "x"]


def test_get_coef_names_returns_pyfixest_iv_names() -> None:
    data = make_basic_iv_data(2001)
    model = pf.feols("y ~ w | x ~ z", data=data)

    assert get_coef_names(model) == ["Intercept", "w", "x"]


def test_get_coef_names_returns_pyfixest_fe_iv_names() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | fe | x ~ z", data=data)

    assert get_coef_names(model) == ["w", "x"]
