import numpy as np
import pytest
from pydantic import ValidationError

from spectratwin.thermal.materials import (
    THERMAL_MATERIALS,
    EmissivityBasis,
    ThermalMaterial,
    get_thermal_material,
)


def test_registry_covers_the_documented_material_families() -> None:
    expected = {
        "asphalt",
        "concrete",
        "painted_metal",
        "rubber",
        "automotive_glass",
        "fabric",
        "skin",
        "vegetation",
    }
    assert set(THERMAL_MATERIALS) == expected


def test_every_material_declares_its_provenance_basis_and_notes() -> None:
    for material in THERMAL_MATERIALS.values():
        assert material.basis is EmissivityBasis.LITERATURE_TYPICAL
        assert len(material.notes) >= 20


def test_emissivity_range_is_constrained_to_the_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ThermalMaterial(
            name="impossible",
            emissivity_min=0.9,
            emissivity_max=1.4,
            temperature_offset_k_min=0.0,
            temperature_offset_k_max=1.0,
            basis=EmissivityBasis.LITERATURE_TYPICAL,
            notes="an emissivity above one is not physical",
        )


def test_inverted_emissivity_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ThermalMaterial(
            name="inverted",
            emissivity_min=0.95,
            emissivity_max=0.90,
            temperature_offset_k_min=0.0,
            temperature_offset_k_max=1.0,
            basis=EmissivityBasis.LITERATURE_TYPICAL,
            notes="max below min must be rejected at construction",
        )


def test_draw_is_reproducible_from_the_same_seed() -> None:
    material = get_thermal_material("asphalt")
    first = material.draw(np.random.default_rng(7), ambient_temperature_k=293.15)
    second = material.draw(np.random.default_rng(7), ambient_temperature_k=293.15)
    assert first == second


def test_draw_stays_inside_the_declared_ranges() -> None:
    material = get_thermal_material("asphalt")
    rng = np.random.default_rng(0)
    for _ in range(200):
        surface = material.draw(rng, ambient_temperature_k=293.15)
        assert material.emissivity_min <= surface.emissivity <= material.emissivity_max
        offset = surface.temperature_k - 293.15
        assert material.temperature_offset_k_min <= offset <= material.temperature_offset_k_max
        assert surface.material_name == "asphalt"


def test_draw_rejects_a_non_physical_ambient_temperature() -> None:
    material = get_thermal_material("asphalt")
    with pytest.raises(ValueError, match="ambient_temperature_k"):
        material.draw(np.random.default_rng(0), ambient_temperature_k=0.0)


def test_unknown_material_name_is_a_typed_failure() -> None:
    with pytest.raises(KeyError, match="unknown thermal material"):
        get_thermal_material("unobtainium")
