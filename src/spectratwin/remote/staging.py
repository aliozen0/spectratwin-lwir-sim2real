"""Checksum-verified file transfer across persistent and ephemeral storage.

The caller supplies ordinary filesystem paths. This module deliberately has
no Google Drive/Colab dependency: a mounted persistent store is just a source
or destination adapter at the orchestration boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COPY_BUFFER_BYTES = 1024 * 1024


class VerifiedFileTransfer(BaseModel):
    """Compact transfer evidence safe to print in run logs."""

    model_config = ConfigDict(frozen=True)

    artifact_name: str
    sha256: str
    size_bytes: int
    reused_existing: bool
    completion_marker_name: str | None = None


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""
    if not path.is_file():
        raise FileNotFoundError(f"regular file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_checksum(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return normalized


def _copy_atomically(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, temporary, length=_COPY_BUFFER_BYTES)
            temporary.flush()
            os.fsync(temporary.fileno())

        copied_sha256 = sha256_file(temporary_path)
        if copied_sha256 != expected_sha256:
            raise OSError(
                f"copied file checksum mismatch (expected {expected_sha256}, got {copied_sha256})"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def stage_file(source: Path, destination: Path, expected_sha256: str) -> VerifiedFileTransfer:
    """Verify and atomically stage one persistent file into execution-local storage.

    A matching destination is safely reused after checksum verification. An
    existing mismatched destination is never overwritten because it may be a
    valid artifact belonging to another identity.
    """
    expected = _validated_checksum(expected_sha256)
    source_sha256 = sha256_file(source)
    if source_sha256 != expected:
        raise ValueError(f"source checksum mismatch (expected {expected}, got {source_sha256})")

    reused = False
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected:
            raise FileExistsError(
                "destination already exists with a different identity; refusing to overwrite"
            )
        reused = True
    else:
        _copy_atomically(source, destination, expected)

    return VerifiedFileTransfer(
        artifact_name=destination.name,
        sha256=expected,
        size_bytes=source.stat().st_size,
        reused_existing=reused,
    )


def _write_completion_marker(marker_path: Path, transfer: VerifiedFileTransfer) -> None:
    payload = json.dumps(
        {
            "schema_version": "spectratwin-persist-marker-v1",
            "artifact_name": transfer.artifact_name,
            "sha256": transfer.sha256,
            "size_bytes": transfer.size_bytes,
        },
        indent=2,
        sort_keys=True,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{marker_path.name}.",
            suffix=".partial",
            dir=marker_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, marker_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def persist_file(source: Path, destination: Path) -> VerifiedFileTransfer:
    """Persist one output file and create a marker only after verified transfer."""
    source_sha256 = sha256_file(source)
    marker_path = destination.with_name(f"{destination.name}.COMPLETED.json")
    if marker_path.exists():
        existing = json.loads(marker_path.read_text())
        if (
            existing.get("sha256") != source_sha256
            or existing.get("size_bytes") != source.stat().st_size
            or existing.get("artifact_name") != destination.name
        ):
            raise FileExistsError(
                "completion marker already exists with a different identity; refusing to overwrite"
            )
        if not destination.is_file():
            raise FileNotFoundError(
                "completion marker exists but its persisted artifact is missing; refusing repair"
            )

    transfer = stage_file(source, destination, source_sha256)
    marked_transfer = transfer.model_copy(update={"completion_marker_name": marker_path.name})
    if marker_path.exists():
        return marked_transfer
    else:
        _write_completion_marker(marker_path, marked_transfer)
    return marked_transfer
