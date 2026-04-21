from __future__ import annotations

from typing import Any

import pytest

from scpc.utils.spatial import validate_large_n_seed
from tests.utils import R, execute_r_code


def test_validate_large_n_seed_accepts_valid_integer_states() -> None:
    assert validate_large_n_seed(0) == 0
    assert validate_large_n_seed(2**32 - 1) == 2**32 - 1


def test_validate_large_n_seed_rejects_invalid_values() -> None:
    bad_seed: Any = 1.5

    with pytest.raises(ValueError, match=r"`large_n_seed` must be"):
        validate_large_n_seed(-1)

    with pytest.raises(ValueError, match=r"`large_n_seed` must be"):
        validate_large_n_seed(2**32)

    with pytest.raises(ValueError, match=r"`large_n_seed` must be"):
        validate_large_n_seed(bad_seed)


@pytest.mark.skipif(R is None, reason="Rscript not installed")
def test_python_r_parity_validate_large_n_seed() -> None:
    bad_seed: Any = 1.5

    r_value = execute_r_code(
        """
        result <- tryCatch(
          list(ok = TRUE, value = getFromNamespace(".validate_large_n_seed", "scpcR")(payload$seed)),
          error = function(e) list(ok = FALSE, message = conditionMessage(e))
        )
        """,
        payload={"seed": 1.5},
    )

    with pytest.raises(ValueError) as excinfo:
        validate_large_n_seed(bad_seed)

    assert r_value["ok"] is False
    assert str(excinfo.value) == r_value["message"]
