import struct
import zlib

import numpy as np
import pytest

from spectratwin.annotation.coco import build_frame_annotations
from spectratwin.annotation.overlay import OverlayError, compose_overlay, write_overlay_png
from spectratwin.annotation.policy import AnnotationPolicy


def _annotation():
    instance_map = np.zeros((6, 7), dtype=np.uint8)
    instance_map[1:5, 1:6] = 1
    return build_frame_annotations(
        sample_index=0,
        file_name="image.png",
        instance_map=instance_map,
        instance_categories={0: "person"},
        policy=AnnotationPolicy(),
    ).annotations[0]


def _png_chunks(payload: bytes) -> list[tuple[bytes, bytes]]:
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    chunks: list[tuple[bytes, bytes]] = []
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
        assert zlib.crc32(chunk_type + data) & 0xFFFFFFFF == expected_crc
        chunks.append((chunk_type, data))
        offset += 12 + length
    return chunks


def test_overlay_tints_mask_draws_bbox_and_does_not_mutate_input() -> None:
    image = np.zeros((6, 7), dtype=np.uint8)
    original = image.copy()

    overlay = compose_overlay(image, [_annotation()])

    assert np.array_equal(image, original)
    assert overlay.shape == (6, 7, 3)
    assert overlay.dtype == np.uint8
    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[3, 3].tolist() != [0, 0, 0]  # mask tint
    assert overlay[1, 1].tolist() == [255, 64, 64]  # bbox edge


def test_png_is_valid_rgb_scanline_data_using_only_stdlib_decode(tmp_path) -> None:
    path = tmp_path / "overlay.png"
    overlay = write_overlay_png(path, np.zeros((6, 7, 3), dtype=np.uint8), [_annotation()])

    chunks = _png_chunks(path.read_bytes())
    assert [chunk_type for chunk_type, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1]
    )
    assert (width, height, bit_depth, color_type, compression, filtering, interlace) == (
        7,
        6,
        8,
        2,
        0,
        0,
        0,
    )
    scanlines = zlib.decompress(chunks[1][1])
    assert len(scanlines) == 6 * (1 + 7 * 3)
    assert all(scanlines[row * (1 + 7 * 3)] == 0 for row in range(6))
    assert overlay[1, 1].tolist() == [255, 64, 64]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2,), dtype=np.uint8),
    ],
)
def test_overlay_rejects_unsupported_image_contracts(image: np.ndarray) -> None:
    with pytest.raises(OverlayError):
        compose_overlay(image, [])


def test_overlay_rejects_annotation_whose_mask_size_differs() -> None:
    with pytest.raises(OverlayError, match="shape"):
        compose_overlay(np.zeros((2, 2), dtype=np.uint8), [_annotation()])
