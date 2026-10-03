"""Import Sionna's physical layer without requiring its unused ray tracer.

Sionna 2.1 imports ``sionna.rt`` at package initialization.  The TR 38.901
channel code only needs ``sionna.phy``; on macOS 14 the RT dependency DrJit
cannot initialize its Metal backend.  The temporary finder below skips RT
only while importing PHY and removes the placeholder immediately afterward,
so a later, explicit ``sionna.rt`` import still uses the real package.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from contextlib import contextmanager
from types import ModuleType


_IMPORT_LOCK = threading.RLock()


class _UnusedRTLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__path__ = []


class _UnusedRTFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sionna.rt":
            return importlib.machinery.ModuleSpec(
                fullname, _UnusedRTLoader(), is_package=True
            )
        return None


@contextmanager
def _physical_layer_import():
    if sys.platform != "darwin" or "sionna" in sys.modules:
        yield
        return
    with _IMPORT_LOCK:
        finder = _UnusedRTFinder()
        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            sys.meta_path.remove(finder)
            sys.modules.pop("sionna.rt", None)
            package = sys.modules.get("sionna")
            if package is not None and hasattr(package, "rt"):
                delattr(package, "rt")


def import_sionna_phy(module_name: str):
    """Return a ``sionna.phy`` module, leaving RT imports unmodified."""

    if module_name != "sionna.phy" and not module_name.startswith("sionna.phy."):
        raise ValueError("only sionna.phy modules can be imported here")
    with _physical_layer_import():
        return importlib.import_module(module_name)
