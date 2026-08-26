import numpy as np
import pytest

from spectratwin.sensor.agc import apply_agc
from spectratwin.sensor.blur import apply_gaussian_blur
from spectratwin.sensor.noise import apply_sensor_noise
from spectratwin.sensor.quantization import quantize_to_bit_depth

# --- Blur -----------------------------------------------------------------------


def test_blur_disabled_is_exact_copy_not_same_object():
    frame = np.random.default_rng(0).uniform(0, 1, size=(16, 16))
    blurred = apply_gaussian_blur(frame, sigma_px=0.0)
    assert np.array_equal(blurred, frame)
    assert blurred is not frame


def test_blur_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_px"):
        apply_gaussian_blur(np.zeros((4, 4)), sigma_px=-1.0)


def test_blur_rejects_non_2d_frame():
    with pytest.raises(ValueError, match="2D"):
        apply_gaussian_blur(np.zeros((4, 4, 3)), sigma_px=1.0)


def test_blur_preserves_shape_and_constant_frame():
    frame = np.full((10, 10), 42.0)
    blurred = apply_gaussian_blur(frame, sigma_px=2.0)
    assert blurred.shape == frame.shape
    assert np.allclose(blurred, 42.0)


def test_blur_reduces_variance_of_noisy_frame():
    rng = np.random.default_rng(0)
    frame = rng.normal(loc=0.0, scale=10.0, size=(64, 64))
    blurred = apply_gaussian_blur(frame, sigma_px=3.0)
    assert np.var(blurred) < np.var(frame)


# --- Noise -----------------------------------------------------------------------


def test_noise_disabled_is_exact_copy_and_does_not_consume_rng():
    frame = np.zeros((8, 8))
    rng = np.random.default_rng(0)
    result = apply_sensor_noise(frame, rng, sigma_read=0.0, signal_dependent_gain=0.0)
    assert np.array_equal(result, frame)

    # rng state must be untouched: first draw must match a freshly-seeded rng.
    fresh = np.random.default_rng(0)
    assert rng.normal() == fresh.normal()


def test_noise_rejects_negative_parameters():
    frame = np.zeros((4, 4))
    with pytest.raises(ValueError, match="sigma_read"):
        apply_sensor_noise(frame, np.random.default_rng(0), sigma_read=-1.0)
    with pytest.raises(ValueError, match="signal_dependent_gain"):
        apply_sensor_noise(frame, np.random.default_rng(0), signal_dependent_gain=-1.0)


def test_noise_deterministic_replay_with_same_seed():
    frame = np.full((32, 32), 5.0)
    result_a = apply_sensor_noise(frame, np.random.default_rng(7), sigma_read=1.0)
    result_b = apply_sensor_noise(frame, np.random.default_rng(7), sigma_read=1.0)
    assert np.array_equal(result_a, result_b)


def test_noise_signal_independent_std_matches_sigma_read():
    frame = np.zeros((200, 200))
    result = apply_sensor_noise(frame, np.random.default_rng(0), sigma_read=5.0)
    assert np.std(result) == pytest.approx(5.0, rel=0.05)


# --- Quantization ------------------------------------------------------------------


def test_quantize_maps_range_bounds_to_full_scale():
    frame = np.array([[0.0, 5.0, 10.0]])
    quantized = quantize_to_bit_depth(frame, bit_depth=8, value_range=(0.0, 10.0))
    assert quantized.tolist() == [[0, 128, 255]]
    assert quantized.dtype == np.uint8


def test_quantize_clips_out_of_range_values():
    frame = np.array([[-100.0, 1000.0]])
    quantized = quantize_to_bit_depth(frame, bit_depth=8, value_range=(0.0, 10.0))
    assert quantized.tolist() == [[0, 255]]


def test_quantize_uses_uint16_above_8_bits():
    frame = np.array([[0.0, 16383.0, 16383.0]])
    quantized = quantize_to_bit_depth(frame, bit_depth=14, value_range=(0.0, 16383.0))
    assert quantized.dtype == np.uint16
    assert quantized[0, 0] == 0
    assert quantized[0, 1] == 2**14 - 1


@pytest.mark.parametrize("bit_depth", [0, 17])
def test_quantize_rejects_invalid_bit_depth(bit_depth):
    with pytest.raises(ValueError, match="bit_depth"):
        quantize_to_bit_depth(np.zeros((2, 2)), bit_depth=bit_depth, value_range=(0.0, 1.0))


def test_quantize_rejects_inverted_value_range():
    with pytest.raises(ValueError, match="value_range"):
        quantize_to_bit_depth(np.zeros((2, 2)), bit_depth=8, value_range=(1.0, 0.0))


# --- AGC ---------------------------------------------------------------------------


def test_agc_maps_gradient_into_output_range():
    frame = np.linspace(0.0, 100.0, 101).reshape(1, -1)
    mapped, params = apply_agc(frame, low_percentile=0.0, high_percentile=100.0)
    assert mapped.min() == pytest.approx(0.0, abs=1e-6)
    assert mapped.max() == pytest.approx(255.0, abs=1e-6)
    assert params.is_near_constant is False


def test_agc_constant_frame_has_no_nan_or_inf_and_uses_midpoint():
    frame = np.full((10, 10), 7.0)
    mapped, params = apply_agc(frame)
    assert np.all(np.isfinite(mapped))
    assert np.allclose(mapped, 127.5)
    assert params.is_near_constant is True


def test_agc_near_constant_frame_is_detected():
    frame = np.full((10, 10), 100.0)
    frame[0, 0] += 1e-13
    _, params = apply_agc(frame)
    assert params.is_near_constant is True


def test_agc_output_always_within_range_for_random_frame():
    rng = np.random.default_rng(0)
    frame = rng.normal(loc=50.0, scale=1000.0, size=(50, 50))
    mapped, _ = apply_agc(frame, output_range=(0.0, 1.0))
    assert mapped.min() >= 0.0
    assert mapped.max() <= 1.0


def test_agc_is_deterministic_for_same_input():
    rng = np.random.default_rng(0)
    frame = rng.normal(size=(20, 20))
    mapped_a, _ = apply_agc(frame)
    mapped_b, _ = apply_agc(frame)
    assert np.array_equal(mapped_a, mapped_b)


def test_agc_rejects_inverted_percentiles():
    with pytest.raises(ValueError, match="percentile"):
        apply_agc(np.zeros((4, 4)), low_percentile=90.0, high_percentile=10.0)


def test_agc_rejects_inverted_output_range():
    with pytest.raises(ValueError, match="output_range"):
        apply_agc(np.zeros((4, 4)), output_range=(1.0, 0.0))
