"""EqDeepRx reproduction package."""

from .provenance import LOADED_SOURCE_FINGERPRINT
from .config import EqDeepRxConfig, paper_config

__all__ = ["EqDeepRx", "EqDeepRxConfig", "paper_config"]


def __getattr__(name):
    # The CLI configures compiler/driver storage before importing PyTorch.
    if name == "EqDeepRx":
        from .model import EqDeepRx
        return EqDeepRx
    raise AttributeError(name)
