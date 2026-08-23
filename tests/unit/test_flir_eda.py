from spectratwin.real_data.eda import compute_eda
from spectratwin.real_data.records import FlirAnnotationRecord, FlirSampleRecord


def _record(source_id: str, annotations: tuple[FlirAnnotationRecord, ...]) -> FlirSampleRecord:
    return FlirSampleRecord(
        source_id=source_id,
        sequence_key=source_id,
        relative_image_path=source_id,
        width=100,
        height=100,
        image_sha256="0" * 64,
        annotations=annotations,
    )


def test_compute_eda_counts_classes_and_bbox_stats():
    records = [
        _record(
            "a.jpg",
            (FlirAnnotationRecord(project_category="person", bbox_xywh=(0, 0, 10, 20)),),
        ),
        _record(
            "b.jpg",
            (FlirAnnotationRecord(project_category="car", bbox_xywh=(0, 0, 30, 10)),),
        ),
        _record("c.jpg", ()),
    ]

    summary = compute_eda(records)

    assert summary.sample_count == 3
    assert summary.samples_without_annotations == 1
    assert summary.class_counts == {"car": 1, "person": 1}
    assert summary.bbox_width_px is not None
    assert summary.bbox_height_px is not None
    assert summary.bbox_width_px.mean == 20
    assert summary.bbox_height_px.mean == 15


def test_compute_eda_handles_empty_input():
    summary = compute_eda([])
    assert summary.sample_count == 0
    assert summary.class_counts == {}
    assert summary.bbox_width_px is None
