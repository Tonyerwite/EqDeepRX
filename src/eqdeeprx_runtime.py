"""Runtime storage routing for long-running EqDeepRx jobs.

This module intentionally has no third-party imports.  The training entrypoint
calls it before importing PyTorch so compiler and driver caches are created on
the configured run volume instead of the system volume.
"""

from __future__ import annotations

import os
from pathlib import Path


_CACHE_SUBDIRECTORIES = {
    "TEMP": "temp",
    "TMP": "temp",
    "TORCH_HOME": "torch",
    "XDG_CACHE_HOME": "xdg",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUDA_CACHE_PATH": "cuda",
    "MPLCONFIGDIR": "mpl",
}


def _default_runtime_root() -> Path:
    configured = os.environ.get("EQDEEP_RX_RUNTIME_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt" and Path("D:/").exists():
        return Path(r"D:\EqDeepRxRuns\runtime")
    return Path.home() / ".cache" / "eqdeeprx-runtime"


def configure_runtime_storage(root: str | os.PathLike[str] | None = None) -> Path:
    """Route Python, PyTorch, CUDA, Triton and plotting caches to ``root``.

    The explicit ``root`` argument is used by tests and callers that manage
    their own run volume.  Otherwise ``EQDEEP_RX_RUNTIME_DIR`` is honored,
    falling back to the D: run volume on Windows when it is available.
    """

    runtime_root = Path(root) if root is not None else _default_runtime_root()
    runtime_root.mkdir(parents=True, exist_ok=True)
    for variable, relative in _CACHE_SUBDIRECTORIES.items():
        path = runtime_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    return runtime_root

