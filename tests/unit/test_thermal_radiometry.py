import math
from typing import Any

import pytest

from spectratwin.thermal.radiometry import band_radiance, planck_spectral_radiance

# --- Planck spectral radiance --------------------------------------------------


def test_planck_radiance_increases_with_temperature_at_fixed_wavelength():
    wavelength_m = 10e-6
    radiances = [planck_spectral_radiance(wavelength_m, t) for t in (250.0, 300.0, 350.0, 400.0)]
    assert radiances == sorted(radiances)
    assert radiances[0] < radiances[-1]


def test_planck_rejects_non_positive_wavelength():
    with pytest.raises(ValueError, match="wavelength_m"):
        planck_spectral_radiance(0.0, 300.0)


def test_planck_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature_k"):
        planck_spectral_radiance(10e-6, 0.0)


# --- Band radiance: core acceptance criterion ----------------------------------


def test_band_radiance_increases_with_object_temperature_across_range():
    radiances = [
        band_radiance(
            object_temperature_k=to,
            reflected_temperature_k=280.0,
            atmospheric_temperature_k=250.0,
            emissivity=0.95,
            transmittance=0.9,
        )
        for to in (250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0)
    ]
    assert radiances == sorted(radiances)
    assert radiances[0] < radiances[-1]


def test_band_radiance_finite_over_configured_bounds():
    for emissivity in (0.0, 0.5, 1.0):
        for transmittance in (0.0, 0.5, 1.0):
            for temperature_k in (200.0, 300.0, 400.0):
                value = band_radiance(
                    object_temperature_k=temperature_k,
                    reflected_temperature_k=temperature_k,
                    atmospheric_temperature_k=temperature_k,
                    emissivity=emissivity,
                    transmittance=transmittance,
                )
                assert math.isfinite(value)


# --- Equation-structure correctness (edge cases where a term vanishes) --------


def test_zero_emissivity_full_transmittance_reduces_to_reflected_term_only():
    pure_reflected = band_radiance(
        object_temperature_k=999.0,
        reflected_temperature_k=300.0,
        atmospheric_temperature_k=999.0,
        emissivity=0.0,
        transmittance=1.0,
    )
    pure_blackbody_300k = band_radiance(
        object_temperature_k=300.0,
        reflected_temperature_k=999.0,
        atmospheric_temperature_k=999.0,
        emissivity=1.0,
        transmittance=1.0,
    )
    assert pure_reflected == pytest.approx(pure_blackbody_300k)


def test_zero_transmittance_reduces_to_atmospheric_term_only():
    pure_atmosphere = band_radiance(
        object_temperature_k=999.0,
        reflected_temperature_k=999.0,
        atmospheric_temperature_k=300.0,
        emissivity=0.5,
        transmittance=0.0,
    )
    pure_blackbody_300k = band_radiance(
        object_temperature_k=300.0,
        reflected_temperature_k=300.0,
        atmospheric_temperature_k=300.0,
        emissivity=1.0,
        transmittance=1.0,
    )
    assert pure_atmosphere == pytest.approx(pure_blackbody_300k)


def test_narrow_band_approximates_point_spectral_radiance_times_width():
    center_m = 10e-6
    width_m = 1e-9
    narrow = band_radiance(
        object_temperature_k=300.0,
        reflected_temperature_k=300.0,
        atmospheric_temperature_k=300.0,
        emissivity=1.0,
        transmittance=1.0,
        wavelength_band_m=(center_m, center_m + width_m),
    )
    expected = planck_spectral_radiance(center_m, 300.0) * width_m
    assert narrow == pytest.approx(expected, rel=1e-3)


# --- Regression fixture (frozen numeric output) --------------------------------


def test_band_radiance_regression_fixture_default_lwir_band():
    value = band_radiance(
        object_temperature_k=300.0,
        reflected_temperature_k=280.0,
        atmospheric_temperature_k=260.0,
        emissivity=0.95,
        transmittance=0.9,
    )
    assert value == pytest.approx(51.492910142693, rel=1e-9)


# --- Invalid input rejection ----------------------------------------------------


@pytest.mark.parametrize("emissivity", [-0.01, 1.01])
def test_band_radiance_rejects_invalid_emissivity(emissivity):
    with pytest.raises(ValueError, match="emissivity"):
        band_radiance(
            object_temperature_k=300.0,
            reflected_temperature_k=280.0,
            atmospheric_temperature_k=260.0,
            emissivity=emissivity,
            transmittance=0.9,
        )


@pytest.mark.parametrize("transmittance", [-0.01, 1.01])
def test_band_radiance_rejects_invalid_transmittance(transmittance):
    with pytest.raises(ValueError, match="transmittance"):
        band_radiance(
            object_temperature_k=300.0,
            reflected_temperature_k=280.0,
            atmospheric_temperature_k=260.0,
            emissivity=0.95,
            transmittance=transmittance,
        )


@pytest.mark.parametrize(
    "field", ["object_temperature_k", "reflected_temperature_k", "atmospheric_temperature_k"]
)
def test_band_radiance_rejects_non_positive_temperature(field):
    kwargs: dict[str, Any] = {
        "object_temperature_k": 300.0,
        "reflected_temperature_k": 280.0,
        "atmospheric_temperature_k": 260.0,
        "emissivity": 0.95,
        "transmittance": 0.9,
    }
    kwargs[field] = 0.0
    with pytest.raises(ValueError, match=field):
        band_radiance(**kwargs)


def test_band_radiance_rejects_inverted_wavelength_band():
    with pytest.raises(ValueError, match="wavelength_band_m"):
        band_radiance(
            object_temperature_k=300.0,
            reflected_temperature_k=280.0,
            atmospheric_temperature_k=260.0,
            emissivity=0.95,
            transmittance=0.9,
            wavelength_band_m=(14e-6, 8e-6),
        )


def test_band_radiance_rejects_too_few_samples():
    with pytest.raises(ValueError, match="n_samples"):
        band_radiance(
            object_temperature_k=300.0,
            reflected_temperature_k=280.0,
            atmospheric_temperature_k=260.0,
            emissivity=0.95,
            transmittance=0.9,
            n_samples=1,
        )
