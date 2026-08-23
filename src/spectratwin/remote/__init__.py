"""Provider-independent remote staging and persistence."""

from spectratwin.remote.staging import (
    VerifiedFileTransfer,
    persist_file,
    sha256_file,
    stage_file,
)

__all__ = ["VerifiedFileTransfer", "persist_file", "sha256_file", "stage_file"]
