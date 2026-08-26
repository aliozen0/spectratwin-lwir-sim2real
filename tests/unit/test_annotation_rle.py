import numpy as np
import pytest
from pydantic import ValidationError

from spectratwin.annotation.rle import RleError, UncompressedRle, decode_rle, encode_rle


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 4), dtype=bool),
        np.ones((3, 4), dtype=bool),
        np.array([[0, 0, 0], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
    ],
)
def test_uncompressed_rle_round_trips_binary_masks(mask: np.ndarray) -> None:
    encoded = encode_rle(mask)

    decoded = decode_rle(encoded)

    assert decoded.dtype == np.bool_
    assert np.array_equal(decoded, mask.astype(bool))


def test_rle_uses_coco_column_major_order() -> None:
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    encoded = encode_rle(mask)

    assert encoded.size == (2, 3)
    assert encoded.counts == (0, 1, 2, 2, 1)
    assert encoded.model_dump(mode="json") == {
        "size": [2, 3],
        "counts": [0, 1, 2, 2, 1],
    }


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((0, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        np.array([[0, 2]], dtype=np.uint8),
        np.array([[0, -1]], dtype=np.int8),
    ],
)
def test_encoder_rejects_non_binary_or_invalid_shapes(mask: np.ndarray) -> None:
    with pytest.raises(RleError):
        encode_rle(mask)


@pytest.mark.parametrize(
    "payload",
    [
        {"size": [2], "counts": [4]},
        {"size": [2, 2], "counts": [3]},
        {"size": [2, 2], "counts": [0, 1, -1, 4]},
        {"size": [2, 2], "counts": [0, 1.0, 3]},
        {"size": [0, 2], "counts": [0]},
    ],
)
def test_decoder_rejects_malformed_runs(payload: dict[str, list[int | float]]) -> None:
    with pytest.raises(RleError):
        decode_rle(payload)


def test_persisted_rle_model_rejects_invalid_total() -> None:
    with pytest.raises(ValidationError):
        UncompressedRle(size=(2, 2), counts=(1, 2))
