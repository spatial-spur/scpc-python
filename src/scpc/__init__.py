from __future__ import annotations
import sys
from types import ModuleType
from typing import Any
from .core import scpc
from .types import SCPCResult


class SCPC(ModuleType):
    """module type forwarding the scpc() function to enable both `import scpc` import (instead of `from scpc import scpc`)"""

    def __call__(self, *args: Any, **kwargs: Any) -> SCPCResult:
        return scpc(*args, **kwargs)


module = sys.modules[__name__]
module.__class__ = SCPC

__all__ = ["scpc", "SCPCResult"]
