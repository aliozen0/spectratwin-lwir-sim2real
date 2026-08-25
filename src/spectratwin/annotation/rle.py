"""NumPy-only uncompressed COCO run-length encoding (SPEC-005)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


class RleError(ValueError):
    """Raised when a mask or RLE violates the uncompressed COCO contract."""


class UncompressedRle(BaseModel):
    """JSON-compatible uncompressed COCO RLE in column-major order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    size: tuple[PositiveInt, PositiveInt]
    counts: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def _runs_cover_the_declared_image(self) -> Self:
        expected = self.size[0] * self.size[1]
        if not self.counts:
            raise ValueError("RLE counts must contain at least one run")
        if sum(self.counts) != expected:
            raise ValueError(f"RLE counts cover {sum(self.counts)} pixels, expected {expected}")
        return self


def encode_rle(binary_mask: np.ndarray) -> UncompressedRle:
    """Encode a 2D binary mask using COCO's column-major run convention."""
    mask = np.asarray(binary_mask)
    if mask.ndim != 2 or 0 in mask.shape:
        raise RleError(f"binary mask must have a non-empty 2D shape, got {mask.shape}")
    if np.issubdtype(mask.dtype, np.bool_):
        pass
    elif np.issubdtype(mask.dtype, np.integer):
        if np.any((mask != 0) & (mask != 1)):
            raise RleError("binary mask values must be only 0 or 1")
    else:
        raise RleError(f"binary mask must have a bool or integer dtype, got {mask.dtype}")

    flat = mask.astype(np.uint8, copy=False).ravel(order="F")
    counts: list[int] = []
    current_value = 0
    current_run = 0
    for raw_value in flat:
        value = int(raw_value)
        if value == current_value:
            current_run += 1
        else:
            counts.append(current_run)
            current_run = 1
            current_value = value
    counts.append(current_run)
    return UncompressedRle(size=mask.shape, counts=tuple(counts))


def decode_rle(rle: UncompressedRle | Mapping[str, object]) -> np.ndarray:
    """Decode uncompressed COCO RLE to a 2D boolean mask."""
    if isinstance(rle, UncompressedRle):
        validated = rle
    else:
        try:
            # JSON arrays deserialize as lists; the model's element fields are
            # strict, while normal validation intentionally converts only the
            # container to the persisted tuple representation.
            validated = UncompressedRle.model_validate(dict(rle))
        except (TypeError, ValidationError) as exc:
            raise RleError(f"invalid uncompressed COCO RLE: {exc}") from exc

    flat = np.zeros(validated.size[0] * validated.size[1], dtype=bool)
    offset = 0
    foreground = False
    for count in validated.counts:
        if foreground:
            flat[offset : offset + count] = True
        offset += count
        foreground = not foreground
    return flat.reshape(validated.size, order="F")
