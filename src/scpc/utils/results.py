from __future__ import annotations
from collections.abc import Sequence


def resolve_parm_indices(
    coef_names: Sequence[str],
    parm: str | int | Sequence[str] | Sequence[int] | None,
) -> list[int]:
    """Resolve `confint(parm=...)` to zero-based coefficient indices.
    TODO: overcomplicates things but makes it more similar to R behavior
    when it comes to the behavior in the SCPCResult methods
    """
    # return all if no parameter selected
    if parm is None:
        return list(range(len(coef_names)))

    # single coef by name
    if isinstance(parm, str):
        return [coef_names.index(parm)]

    # index of coef
    if isinstance(parm, int):
        return [parm]

    values = list(parm)
    if not values:
        return []

    # check if list of names or indices
    if isinstance(values[0], str):
        return [coef_names.index(str(value)) for value in values]

    return [int(value) for value in values]
