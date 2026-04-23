from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pyfixest as pf
import pytest

from scpc.utils.data import get_fixest_iv_design
from tests.config import ATOL, RTOL
from tests.utils import (
    R,
    execute_r_code,
    make_basic_iv_data,
    make_one_way_fe_iv_data,
    make_two_way_fe_iv_data,
    reorder_r_columns_to_py,
)


def test_get_fixest_iv_design_extracts_the_stored_pyfixest_matrices() -> None:
    data = make_basic_iv_data(2001)
    model = pf.feols("y ~ w | x ~ z", data=data)
    design = get_fixest_iv_design(model)
    coef_names = [str(name) for name in model._coefnames]
    z_names = [str(name) for name in model._coefnames_z]
    endo_idx = [i for i, name in enumerate(coef_names) if name not in z_names]
    exo_idx = [i for i, name in enumerate(coef_names) if name in z_names]
    raw_x = np.column_stack(
        (
            np.ones(len(data), dtype=float),
            np.asarray(data["w"], dtype=float),
            np.asarray(data["x"], dtype=float),
        )
    )
    raw_z = np.column_stack(
        (
            np.ones(len(data), dtype=float),
            np.asarray(data["w"], dtype=float),
            np.asarray(data["z"], dtype=float),
        )
    )
    fit_x = np.asarray(model._model_1st_stage._Y_hat_link, dtype=float)[:, None]

    npt.assert_allclose(design["X"], raw_x, atol=1e-12, rtol=0.0)
    npt.assert_allclose(design["Z"], raw_z, atol=1e-12, rtol=0.0)
    npt.assert_allclose(
        design["model_mat"][:, exo_idx],
        raw_x[:, exo_idx],
        atol=1e-12,
        rtol=0.0,
    )
    npt.assert_allclose(
        design["model_mat"][:, endo_idx],
        fit_x,
        atol=1e-12,
        rtol=0.0,
    )
    assert not np.allclose(
        design["model_mat"][:, endo_idx],
        raw_x[:, endo_idx],
    )
    assert design["coef_names"] == coef_names
    assert design["has_fixef"] is False
    assert design["fixef_id"] is None


def test_get_fixest_iv_design_marks_one_way_absorbed_fixed_effects() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | fe | x ~ z", data=data)
    design = get_fixest_iv_design(model)
    coef_names = [str(name) for name in model._coefnames]
    z_names = [str(name) for name in model._coefnames_z]
    endo_idx = [i for i, name in enumerate(coef_names) if name not in z_names]
    exo_idx = [i for i, name in enumerate(coef_names) if name in z_names]
    raw_x = np.column_stack(
        (
            np.asarray(data["w"], dtype=float),
            np.asarray(data["x"], dtype=float),
        )
    )
    raw_z = np.column_stack(
        (
            np.asarray(data["w"], dtype=float),
            np.asarray(data["z"], dtype=float),
        )
    )
    fit_x = np.asarray(model._model_1st_stage._Y_hat_link, dtype=float)[:, None]

    assert design["has_fixef"] is True
    assert design["fixef_id"] is None
    npt.assert_allclose(design["X"], raw_x, atol=1e-12, rtol=0.0)
    npt.assert_allclose(design["Z"], raw_z, atol=1e-12, rtol=0.0)
    npt.assert_allclose(
        design["model_mat"][:, exo_idx],
        raw_x[:, exo_idx],
        atol=1e-12,
        rtol=0.0,
    )
    npt.assert_allclose(
        design["model_mat"][:, endo_idx],
        fit_x,
        atol=1e-12,
        rtol=0.0,
    )
    assert not np.allclose(
        design["model_mat"][:, endo_idx],
        raw_x[:, endo_idx],
    )


def test_get_fixest_iv_design_marks_two_way_absorbed_fixed_effects() -> None:
    data = make_two_way_fe_iv_data(2005)
    model = pf.feols("y ~ w | fe1 + fe2 | x ~ z", data=data)
    design = get_fixest_iv_design(model)

    assert design["has_fixef"] is True
    assert design["fixef_id"] is None
    assert design["X"].shape[0] == len(data)
    assert design["Z"].shape[0] == len(data)
    assert design["model_mat"].shape[0] == len(data)
    assert design["model_mat"].shape[1] == len(model._coefnames)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_get_fixest_iv_design() -> None:
    data = make_one_way_fe_iv_data(2002)
    model = pf.feols("y ~ w | fe | x ~ z", data=data)

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
        fit <- fixest::feols(y ~ w | fe | x ~ z, data = d)
        design <- getFromNamespace(".get_fixest_iv_design", "scpcR")(fit)
        result <- list(
          coef_names = unname(design$coef_names),
          x_names = unname(colnames(design$X)),
          z_names = unname(colnames(design$Z)),
          model_names = unname(colnames(design$model_mat)),
          X = unname(split(design$X, row(design$X))),
          Z = unname(split(design$Z, row(design$Z))),
          model_mat = unname(split(design$model_mat, row(design$model_mat))),
          has_fixef = design$has_fixef
        )
        """,
        payload={"data": data.to_dict(orient="list")},
    )
    py_value = get_fixest_iv_design(model)
    py_coef_names = [str(name) for name in model._coefnames]
    r_x = reorder_r_columns_to_py(
        py_coef_names,
        [str(name) for name in r_value["x_names"]],
        np.array(r_value["X"]),
    )
    r_model = reorder_r_columns_to_py(
        py_coef_names,
        [str(name) for name in r_value["model_names"]],
        np.array(r_value["model_mat"]),
    )
    z_order = ["w", "z"]
    r_z = reorder_r_columns_to_py(
        z_order, [str(name) for name in r_value["z_names"]], np.array(r_value["Z"])
    )

    npt.assert_allclose(py_value["X"], r_x, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(py_value["Z"], r_z, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(py_value["model_mat"], r_model, atol=ATOL, rtol=RTOL)
    assert py_value["has_fixef"] is r_value["has_fixef"]
