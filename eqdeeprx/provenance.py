"""Bind a process to the Python sources it initially imported."""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


LOADED_SOURCE_FINGERPRINT = source_fingerprint()


def require_source_unchanged() -> str:
    if source_fingerprint() != LOADED_SOURCE_FINGERPRINT:
        raise RuntimeError("source changed after this process started; restart before continuing")
    return LOADED_SOURCE_FINGERPRINT
