from __future__ import annotations

import os
from pathlib import Path


def test_runtime_storage_routes_training_caches_to_requested_drive(tmp_path, monkeypatch):
    from eqdeeprx_runtime import configure_runtime_storage

    runtime_root = tmp_path / "eqdeeprx-runtime"
    monkeypatch.delenv("EQDEEP_RX_RUNTIME_DIR", raising=False)

    result = configure_runtime_storage(runtime_root)

    assert result == runtime_root
    expected = {
        "TEMP": runtime_root / "temp",
        "TMP": runtime_root / "temp",
        "TORCH_HOME": runtime_root / "torch",
        "XDG_CACHE_HOME": runtime_root / "xdg",
        "TRITON_CACHE_DIR": runtime_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": runtime_root / "torchinductor",
        "CUDA_CACHE_PATH": runtime_root / "cuda",
        "MPLCONFIGDIR": runtime_root / "mpl",
    }
    for name, path in expected.items():
        assert Path(path).is_dir()
        assert Path(os.environ[name]) == path
