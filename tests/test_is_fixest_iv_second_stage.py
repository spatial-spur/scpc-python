from __future__ import annotations

import pyfixest as pf
import pytest

from scpc.utils.data import is_fixest_iv_second_stage
from tests.utils import R, execute_r_code, make_one_way_fe_iv_data


def test_is_fixest_iv_second_stage_distinguishes_pyfixest_ols_and_iv_models() -> None:
    data = make_one_way_fe_iv_data(2002)
    fit_ols = pf.feols("y ~ w", data=data)
    fit_iv = pf.feols("y ~ w | x ~ z", data=data)
    fit_fe_iv = pf.feols("y ~ w | fe | x ~ z", data=data)

    assert is_fixest_iv_second_stage(fit_ols) is False
    assert is_fixest_iv_second_stage(fit_iv) is True
    assert is_fixest_iv_second_stage(fit_fe_iv) is True


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_is_fixest_iv_second_stage() -> None:
    data = make_one_way_fe_iv_data(2002)
    fit_ols = pf.feols("y ~ w", data=data)
    fit_iv = pf.feols("y ~ w | x ~ z", data=data)
    fit_fe_iv = pf.feols("y ~ w | fe | x ~ z", data=data)

    r_value = execute_r_code(
        """
        suppressPackageStartupMessages(library(fixest))
        d <- data.frame(
          y = unlist(payload$data$y),
          x = unlist(payload$data$x),
          w = unlist(payload$data$w),
          z = unlist(payload$data$z),
          fe = unlist(payload$data$fe)
        )
        fit_ols <- fixest::feols(y ~ w, data = d)
        fit_iv <- fixest::feols(y ~ w | x ~ z, data = d)
        fit_fe_iv <- fixest::feols(y ~ w | fe | x ~ z, data = d)
        result <- list(
          ols = getFromNamespace(".is_fixest_iv_second_stage", "scpcR")(fit_ols),
          iv = getFromNamespace(".is_fixest_iv_second_stage", "scpcR")(fit_iv),
          fe_iv = getFromNamespace(".is_fixest_iv_second_stage", "scpcR")(fit_fe_iv)
        )
        """,
        payload={"data": data.to_dict(orient="list")},
    )

    assert is_fixest_iv_second_stage(fit_ols) is r_value["ols"]
    assert is_fixest_iv_second_stage(fit_iv) is r_value["iv"]
    assert is_fixest_iv_second_stage(fit_fe_iv) is r_value["fe_iv"]
