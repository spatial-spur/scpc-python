from __future__ import annotations

import pytest

from tests.utils import (
    FIXEST_R_VERSION,
    R,
    SANDWICH_R_VERSION,
    SCPC_R_VERSION,
    execute_r_code,
)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_r_reference_versions_are_pinned_for_parity() -> None:
    r_value = execute_r_code(
        """
        result <- list(
          fixest = as.character(utils::packageDescription("fixest")$Version),
          sandwich = as.character(utils::packageDescription("sandwich")$Version),
          scpcR = as.character(utils::packageDescription("scpcR")$Version)
        )
        """
    )

    assert r_value["fixest"] == FIXEST_R_VERSION
    assert r_value["sandwich"] == SANDWICH_R_VERSION
    assert r_value["scpcR"] == SCPC_R_VERSION
