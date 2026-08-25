import json

import numpy as np
import pytest

from spectratwin.render.atmosphere import apply_atmosphere
from spectratwin.render.frame import ThermalFrameMetadata, build_frame_metadata
from spectratwin.render.parameters import derive_surface_parameters
from spectratwin.render.settings import RenderSettings
from spectratwin.sensor.agc import apply_agc
from spectratwin.thermal.materials import ResolvedThermalSurface


def _metadata() -> ThermalFrameMetadata:
    raw = np.array([[30.0, 60.0], [45.0, 90.0]])
    corrected, atmosphere = apply_atmosphere(
        raw, transmittance=0.9, atmospheric_temperature_k=288.0
    )
    _, agc = apply_agc(corrected)
    surface = derive_surface_parameters(
        surface=ResolvedThermalSurface(
            material_name="asphalt", emissivity=0.94, temperature_k=310.0
        )
    )
    return build_frame_metadata(
        sample_seed=5,
        master_seed=0,
        ambient_temperature_k=293.15,
        sky_temperature_k=273.15,
        render_settings=RenderSettings.for_reference_check(
            width_px=64, height_px=52, seed=5, diffuse_bounces=1
        ),
        atmosphere=atmosphere,
        agc=agc,
        surfaces=(surface,),
        renderer_identity="blenderproc-2.8.0/blender-4.2.1",
    )


def test_metadata_records_the_settings_that_change_a_pixel() -> None:
    metadata = _metadata()
    assert metadata.render_settings.diffuse_bounces == 1
    assert metadata.render_settings.samples > 0
    assert metadata.render_settings.device == "CPU"
    assert metadata.renderer_identity == "blenderproc-2.8.0/blender-4.2.1"


def test_metadata_records_the_atmosphere_state_explicitly() -> None:
    metadata = _metadata()
    assert metadata.atmosphere.enabled is True
    assert metadata.atmosphere.transmittance == pytest.approx(0.9)


def test_metadata_round_trips_through_json() -> None:
    metadata = _metadata()
    payload = json.loads(metadata.model_dump_json())
    assert ThermalFrameMetadata(**payload) == metadata


def test_metadata_is_versioned() -> None:
    assert _metadata().schema_version == "spectratwin-thermal-frame-v1"


def test_agc_parameters_are_carried_so_display_mapping_is_reproducible() -> None:
    metadata = _metadata()
    assert metadata.agc.input_low <= metadata.agc.input_high
