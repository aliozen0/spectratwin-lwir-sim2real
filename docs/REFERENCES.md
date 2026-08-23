# References and Current Official Documentation

This file records the main sources used to choose the initial architecture. Re-check versions before pinning dependencies.

## Synthetic data / BlenderProc

- BlenderProc documentation: https://dlr-rm.github.io/BlenderProc/
- COCO annotations example: https://dlr-rm.github.io/BlenderProc/examples/advanced/coco_annotations/README.html
- Camera sampling example: https://dlr-rm.github.io/BlenderProc/examples/basics/camera_sampling/README.html

Key design implication: BlenderProc already provides procedural scene/camera/material capabilities, RGB/depth/segmentation rendering and COCO writing, so SpectraTwin should build its differentiating thermal/sensor layer on top rather than recreating these generic features.

## Real thermal benchmark

- Teledyne FLIR ADAS dataset: https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/

At the time of planning, the official page lists 26,442 fully annotated frames, about 520k boxes over 15 categories, 9,711 thermal and 9,233 RGB training/validation images, 7,498 matched video frames, pre-AGC thermal data, and a Tau 2 640×512 / 13 mm reference thermal camera specification. Re-check the page before final dataset documentation.

## Detector

- RT-DETR official repository: https://github.com/lyuwenyu/RT-DETR
- Hugging Face RT-DETRv2 docs: https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2

## Python reproducibility / quality

- uv project structure and lockfile: https://docs.astral.sh/uv/concepts/projects/layout/
- uv locking/sync: https://docs.astral.sh/uv/concepts/projects/sync/
- Ruff configuration: https://docs.astral.sh/ruff/configuration/
- Pyright configuration: https://github.com/microsoft/pyright/blob/main/docs/configuration.md
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- pre-commit: https://pre-commit.com/

## Config/data contracts

- Hydra structured config: https://hydra.cc/docs/1.3/advanced/terminology/
- Pydantic validation: https://pydantic.dev/docs/validation/latest/get-started/

## Data and experiment lineage

- DVC command/workflow reference: https://dvc.org/doc/command-reference/
- MLflow tracking: https://mlflow.org/docs/latest/ml/tracking/

## Google Colab execution

- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Google Colab local runtimes: https://research.google.com/colaboratory/local-runtimes.html

Design implications from the FAQ:

- managed resource limits and GPU types can vary,
- managed runtimes are ephemeral,
- reduce repeated mounted-Drive read/write operations,
- many small Drive I/O operations can fail or hit quota,
- for many files, copy an archive to the Colab VM and unpack locally,
- use GPU runtimes only when work actually benefits from GPU acceleration.
