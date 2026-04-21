from __future__ import annotations

import pytest

from scpc.utils.spatial import validate_scpc_method
from tests.utils import R, execute_r_code


def test_validate_scpc_method_accepts_supported_methods() -> None:
    assert validate_scpc_method("auto") == "auto"
    assert validate_scpc_method("exact") == "exact"
    assert validate_scpc_method("approx") == "approx"


def test_validate_scpc_method_rejects_unknown_methods() -> None:
    with pytest.raises(ValueError, match=r"`method` must be"):
        validate_scpc_method("bad")


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_validate_scpc_method() -> None:
    r_value = execute_r_code(
        """
        result <- tryCatch(
          list(ok = TRUE, value = getFromNamespace(".validate_scpc_method", "scpcR")(payload$method)),
          error = function(e) list(ok = FALSE, message = conditionMessage(e))
        )
        """,
        payload={"method": "bad"},
    )

    with pytest.raises(ValueError) as excinfo:
        validate_scpc_method("bad")

    assert r_value["ok"] is False
    assert str(excinfo.value) == r_value["message"]
