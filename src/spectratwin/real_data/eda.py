"""EDA statistics over normalized FLIR records.

Pure computation over already-scanned :class:`FlirSampleRecord` objects.
Numbers only exist once a caller supplies real scanned records - this
module never invents or assumes counts.
"""

from __future__ import annotations

import statistics
from collections import Counter

from pydantic import BaseModel, ConfigDict

from spectratwin.real_data.records import FlirSampleRecord


class DistributionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    min: float
    max: float
    mean: float
    median: float


class EdaSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_count: int
    samples_without_annotations: int
    class_counts: dict[str, int]
    bbox_width_px: DistributionSummary | None
    bbox_height_px: DistributionSummary | None
    bbox_aspect_ratio: DistributionSummary | None


def _summarize(values: list[float]) -> DistributionSummary | None:
    if not values:
        return None
    return DistributionSummary(
        count=len(values),
        min=min(values),
        max=max(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
    )


def compute_eda(records: list[FlirSampleRecord]) -> EdaSummary:
    class_counts: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    aspect_ratios: list[float] = []
    samples_without_annotations = 0

    for record in records:
        if not record.annotations:
            samples_without_annotations += 1
        for ann in record.annotations:
            class_counts[ann.project_category] += 1
            _, _, w, h = ann.bbox_xywh
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

    return EdaSummary(
        sample_count=len(records),
        samples_without_annotations=samples_without_annotations,
        class_counts=dict(sorted(class_counts.items())),
        bbox_width_px=_summarize(widths),
        bbox_height_px=_summarize(heights),
        bbox_aspect_ratio=_summarize(aspect_ratios),
    )
