"""FLIR category -> project taxonomy mapping.

docs/DATA_STRATEGY.md fixes the v1 taxonomy to person/car/bicycle. Every
other FLIR category is explicitly ignored rather than silently remapped.
"""

from __future__ import annotations

CATEGORY_MAPPING_VERSION = "flir-taxonomy-v1"

#: Fixed, 0-indexed project taxonomy. Order defines the numeric category ID
#: every downstream consumer (training targets, model ``num_labels``,
#: evaluation) must agree on.
PROJECT_CATEGORIES: tuple[str, ...] = ("person", "car", "bicycle")

_MAPPING = {
    "person": "person",
    "car": "car",
    "bike": "bicycle",
    "bicycle": "bicycle",
}


def map_category(flir_category_name: str) -> str | None:
    """Return the project category, or ``None`` if the category is ignored."""
    return _MAPPING.get(flir_category_name.strip().lower())


def category_id_for(project_category: str) -> int:
    """Return the fixed numeric ID for a project category name."""
    return PROJECT_CATEGORIES.index(project_category)
