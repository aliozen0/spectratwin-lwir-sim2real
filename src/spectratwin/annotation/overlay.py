"""Dependency-light COCO mask/bbox overlays and RGB PNG encoding (SPEC-005)."""

from __future__ import annotations

import os
import struct
import tempfile
import zlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from spectratwin.annotation.coco import CocoAnnotationRecord
from spectratwin.annotation.rle import decode_rle

MASK_ALPHA = 0.35
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (64, 224, 96),
    (64, 144, 255),
)


class OverlayError(ValueError):
    """Raised when image/annotation inputs cannot form a diagnostic overlay."""


def _rgb_copy(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise OverlayError(f"overlay image must use uint8, got {array.dtype}")
    if array.ndim == 2:
        return np.repeat(array[:, :, np.newaxis], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.copy()
    raise OverlayError(f"overlay image must have shape HxW or HxWx3, got {array.shape}")


def compose_overlay(
    image: np.ndarray,
    annotations: Iterable[CocoAnnotationRecord],
) -> np.ndarray:
    """Return an RGB copy with deterministic category-coloured masks and boxes."""
    overlay = _rgb_copy(image)
    height, width, _ = overlay.shape
    for annotation in sorted(annotations, key=lambda record: record.id):
        mask = decode_rle(annotation.segmentation)
        if mask.shape != (height, width):
            raise OverlayError(
                f"annotation {annotation.id} mask shape {mask.shape} differs from image "
                f"shape {(height, width)}"
            )
        colour = np.asarray(_PALETTE[annotation.category_id % len(_PALETTE)], dtype=np.float32)
        if np.any(mask):
            tinted = np.rint(
                overlay[mask].astype(np.float32) * (1.0 - MASK_ALPHA) + colour * MASK_ALPHA
            )
            overlay[mask] = np.clip(tinted, 0, 255).astype(np.uint8)

        x, y, box_width, box_height = annotation.bbox
        x0 = min(max(x, 0), width - 1)
        y0 = min(max(y, 0), height - 1)
        x1 = min(max(x + box_width - 1, 0), width - 1)
        y1 = min(max(y + box_height - 1, 0), height - 1)
        edge_colour = np.asarray(_PALETTE[annotation.category_id % len(_PALETTE)], dtype=np.uint8)
        overlay[y0, x0 : x1 + 1] = edge_colour
        overlay[y1, x0 : x1 + 1] = edge_colour
        overlay[y0 : y1 + 1, x0] = edge_colour
        overlay[y0 : y1 + 1, x1] = edge_colour
    return overlay


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def encode_rgb_png(image: np.ndarray) -> bytes:
    """Encode one uint8 RGB array as a non-interlaced PNG using stdlib only."""
    rgb = _rgb_copy(image)
    height, width, _ = rgb.shape
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes(order="C") for row in rgb)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def write_overlay_png(
    path: Path,
    image: np.ndarray,
    annotations: Iterable[CocoAnnotationRecord],
) -> np.ndarray:
    """Compose and atomically write a diagnostic PNG, returning the RGB array."""
    overlay = compose_overlay(image, annotations)
    payload = encode_rgb_png(overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return overlay
