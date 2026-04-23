from __future__ import annotations

from scpc.utils.results import resolve_parm_indices


def test_resolve_parm_indices_selects_all_for_none() -> None:
    assert resolve_parm_indices(["Intercept", "x", "w"], None) == [0, 1, 2]


def test_resolve_parm_indices_selects_by_name() -> None:
    assert resolve_parm_indices(["Intercept", "x", "w"], "w") == [2]
    assert resolve_parm_indices(["Intercept", "x", "w"], ["w", "x"]) == [2, 1]


def test_resolve_parm_indices_selects_by_index() -> None:
    assert resolve_parm_indices(["Intercept", "x", "w"], 2) == [2]
    assert resolve_parm_indices(["Intercept", "x", "w"], [2, 1]) == [2, 1]
