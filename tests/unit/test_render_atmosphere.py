import numpy as np
import pytest

from spectratwin.render.atmosphere import apply_atmosphere
from spectratwin.render.parameters import blackbody_band_radiance
from spectratwin.thermal.radiometry import band_radiance


def test_unit_transmittance_leaves_the_frame_unchanged() -> None:
    frame = np.array([[10.0, 20.0], [30.0, 40.0]])
    result, parameters = apply_atmosphere(frame, transmittance=1.0, atmospheric_temperature_k=290.0)
    assert np.array_equal(result, frame)
    assert parameters.enabled is False


def test_the_stage_does_not_mutate_its_input() -> None:
    frame = np.array([[10.0, 20.0]])
    original = frame.copy()
    apply_atmosphere(frame, transmittance=0.5, atmospheric_temperature_k=290.0)
    assert np.array_equal(frame, original)


def test_the_stage_reproduces_the_reference_model_end_to_end() -> None:
    """A rendered pixel carrying the reference bracket, pushed through this
    stage, must equal the full reference expression including tau."""
    emissivity, object_k, reflected_k, atmospheric_k, tau = 0.7, 315.0, 275.0, 288.0, 0.8

    bracket = emissivity * blackbody_band_radiance(temperature_k=object_k) + (
        1.0 - emissivity
    ) * blackbody_band_radiance(temperature_k=reflected_k)
    rendered = np.array([[bracket]])

    result, _ = apply_atmosphere(
        rendered, transmittance=tau, atmospheric_temperature_k=atmospheric_k
    )
    reference = band_radiance(
        object_temperature_k=object_k,
        reflected_temperature_k=reflected_k,
        atmospheric_temperature_k=atmospheric_k,
        emissivity=emissivity,
        transmittance=tau,
    )
    assert result[0, 0] == pytest.approx(reference, rel=1e-12)


def test_zero_transmittance_yields_only_path_radiance() -> None:
    frame = np.array([[999.0]])
    result, parameters = apply_atmosphere(frame, transmittance=0.0, atmospheric_temperature_k=290.0)
    expected = blackbody_band_radiance(temperature_k=290.0)
    assert result[0, 0] == pytest.approx(expected, rel=1e-12)
    assert parameters.path_radiance_w_sr_m2 == pytest.approx(expected, rel=1e-12)


def test_transmittance_outside_the_unit_interval_is_rejected() -> None:
    frame = np.array([[1.0]])
    with pytest.raises(ValueError, match="transmittance"):
        apply_atmosphere(frame, transmittance=1.5, atmospheric_temperature_k=290.0)


def test_non_physical_atmospheric_temperature_is_rejected() -> None:
    frame = np.array([[1.0]])
    with pytest.raises(ValueError, match="atmospheric_temperature_k"):
        apply_atmosphere(frame, transmittance=0.9, atmospheric_temperature_k=0.0)
