"""RT-DETRv2 model construction via Hugging Face Transformers.

``RTDetrV2ForObjectDetection`` is the only RT-DETRv2 integration for v1. The
base checkpoint id/revision is a configuration prior, not a measured result,
and is recorded in every run's metadata rather than left implicit.
"""

from __future__ import annotations

from transformers import AutoImageProcessor, RTDetrImageProcessor, RTDetrV2ForObjectDetection

from spectratwin.real_data.taxonomy import PROJECT_CATEGORIES

#: Smallest official RT-DETRv2 checkpoint (ResNet-18 backbone), pinned to an
#: exact revision so every run initializes from the identical weights.
DEFAULT_CHECKPOINT_ID = "PekingU/rtdetr_v2_r18vd"
DEFAULT_CHECKPOINT_REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"

ID2LABEL: dict[int, str] = dict(enumerate(PROJECT_CATEGORIES))
LABEL2ID: dict[str, int] = {name: idx for idx, name in ID2LABEL.items()}


def build_pretrained_model(
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
    checkpoint_revision: str = DEFAULT_CHECKPOINT_REVISION,
) -> tuple[RTDetrV2ForObjectDetection, RTDetrImageProcessor]:
    """Load the pretrained RT-DETRv2 checkpoint, reshaped to the project taxonomy.

    ``ignore_mismatched_sizes`` is required because the checkpoint's
    classification head was trained on 80 COCO categories; the detection
    backbone/encoder/decoder weights transfer, the head is reinitialized.
    """
    processor = AutoImageProcessor.from_pretrained(checkpoint_id, revision=checkpoint_revision)
    model = RTDetrV2ForObjectDetection.from_pretrained(
        checkpoint_id,
        revision=checkpoint_revision,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model, processor
