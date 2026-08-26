import pytest
from pydantic import ValidationError

from spectratwin.render.settings import RenderSettings


def test_reference_check_settings_disable_everything_that_perturbs_a_pixel() -> None:
    settings = RenderSettings.for_reference_check(
        width_px=64, height_px=64, seed=11, diffuse_bounces=1
    )
    assert settings.device == "CPU"
    assert settings.diffuse_bounces == 1
    assert settings.max_bounces == 1
    assert settings.filter_width_px == pytest.approx(0.01)
    assert settings.seed == 11


def test_zero_bounces_is_allowed_for_the_emissive_only_check() -> None:
    settings = RenderSettings.for_reference_check(
        width_px=16, height_px=16, seed=0, diffuse_bounces=0
    )
    assert settings.diffuse_bounces == 0
    assert settings.max_bounces == 0


def test_negative_bounces_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RenderSettings(
            width_px=64,
            height_px=64,
            samples=8,
            max_bounces=-1,
            diffuse_bounces=0,
            seed=0,
        )


def test_diffuse_bounces_may_not_exceed_max_bounces() -> None:
    with pytest.raises(ValidationError):
        RenderSettings(
            width_px=64,
            height_px=64,
            samples=8,
            max_bounces=1,
            diffuse_bounces=4,
            seed=0,
        )


def test_zero_resolution_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RenderSettings(
            width_px=0,
            height_px=64,
            samples=8,
            max_bounces=1,
            diffuse_bounces=1,
            seed=0,
        )


def test_settings_are_frozen_and_serialisable() -> None:
    settings = RenderSettings.for_reference_check(
        width_px=32, height_px=32, seed=3, diffuse_bounces=1
    )
    assert settings.schema_version == "spectratwin-render-settings-v1"
    payload = settings.model_dump(mode="json")
    assert RenderSettings(**payload) == settings
