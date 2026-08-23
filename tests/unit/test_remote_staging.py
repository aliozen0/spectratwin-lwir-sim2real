import json
from pathlib import Path

import pytest

from spectratwin.remote.staging import persist_file, sha256_file, stage_file


def test_stage_file_verifies_and_copies_atomically(tmp_path: Path):
    source = tmp_path / "persistent" / "dataset.tar"
    source.parent.mkdir()
    source.write_bytes(b"dataset archive")
    destination = tmp_path / "content" / "dataset.tar"

    result = stage_file(source, destination, sha256_file(source))

    assert destination.read_bytes() == source.read_bytes()
    assert result.sha256 == sha256_file(destination)
    assert result.reused_existing is False
    assert not list(destination.parent.glob("*.partial"))


def test_stage_file_reuses_matching_destination(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"same")
    destination.write_bytes(b"same")

    result = stage_file(source, destination, sha256_file(source))

    assert result.reused_existing is True


def test_stage_file_rejects_bad_source_checksum_without_output(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"source")

    with pytest.raises(ValueError, match="source checksum mismatch"):
        stage_file(source, destination, "0" * 64)

    assert not destination.exists()


def test_stage_file_never_overwrites_mismatched_destination(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"source")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        stage_file(source, destination, sha256_file(source))

    assert destination.read_bytes() == b"existing"


def test_persist_file_writes_marker_after_verified_copy(tmp_path: Path):
    source = tmp_path / "working" / "result.json"
    source.parent.mkdir()
    source.write_text('{"status":"ok"}')
    destination = tmp_path / "persistent" / "result.json"

    result = persist_file(source, destination)

    marker = destination.with_name("result.json.COMPLETED.json")
    marker_payload = json.loads(marker.read_text())
    assert destination.read_text() == source.read_text()
    assert result.completion_marker_name == marker.name
    assert marker_payload["schema_version"] == "spectratwin-persist-marker-v1"
    assert marker_payload["sha256"] == sha256_file(destination)


def test_persist_file_rejects_conflicting_completion_marker(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "persistent" / "result.bin"
    source.write_bytes(b"result")
    destination.parent.mkdir()
    marker = destination.with_name("result.bin.COMPLETED.json")
    marker.write_text('{"sha256":"wrong"}')

    with pytest.raises(FileExistsError, match="completion marker"):
        persist_file(source, destination)

    assert not destination.exists()


def test_persist_file_rejects_marker_with_missing_artifact(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "persistent" / "result.bin"
    source.write_bytes(b"result")
    destination.parent.mkdir()
    marker = destination.with_name("result.bin.COMPLETED.json")
    marker.write_text(
        json.dumps(
            {
                "artifact_name": destination.name,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    )

    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        persist_file(source, destination)
