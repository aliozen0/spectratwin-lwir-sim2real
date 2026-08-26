# SpectraTwin: LWIR Sim-to-Real

Reproducible, physically motivated RGB-LWIR synthetic-data tooling for
sim-to-real object detection.

SpectraTwin studies whether synthetic long-wave infrared (LWIR) data can reduce
the amount of labeled real thermal imagery needed for urban object detection.
The project fixes the detector family and treats data generation, thermal
modeling and sensor simulation as the research variables.

> Status: active research and engineering. Real-data ingestion, training smoke,
> deterministic scene/camera sampling, reference LWIR radiometry, the Cycles
> thermal path, core sensor effects and ground-truth contracts are implemented.
> Production dataset orchestration and meaningful benchmark experiments remain
> open. No benchmark-quality or sim-to-real result is claimed yet.

## What is implemented

- typed configuration and machine-readable environment reports,
- deterministic seed derivation without hidden global random state,
- FLIR ADAS v2 metadata scanning, three-class taxonomy and frozen manifests,
- dataset EDA and sequence-aware split validation,
- a minimal RT-DETRv2 real-data train/eval smoke loop,
- resumable model and optimizer checkpoints with identity validation,
- deterministic urban-intersection scene descriptions and collision-aware
  placement for people, cars and bicycles,
- camera intrinsics, bounded poses and explicit world/camera transforms,
- NumPy reference LWIR radiometry and bounded thermal-material priors,
- a plain-`bpy` Cycles thermal render path executed under BlenderProc,
- independently testable blur, noise, quantization and AGC stages,
- visible instance/semantic masks, optical-axis depth, COCO RLE/bboxes,
  deterministic IDs, strict schema validation and diagnostic overlays,
- checksum-verified, atomic staging and persistence helpers,
- unit and offline smoke tests for the implemented paths.

## Research scope

- Domain: urban roads and intersections
- Classes: `person`, `car`, `bicycle`
- Modalities: RGB and synthetic LWIR
- Detector: RT-DETRv2 through Hugging Face Transformers
- Real reference: Teledyne FLIR ADAS thermal dataset
- Implemented synthetic contracts: thermal radiance, instance/semantic masks,
  optical-axis depth, COCO boxes/RLE and reproducibility metadata

## Current limitations

- Scene rendering uses deterministic placeholder primitives rather than a
  production registry of reviewed meshes and textures.
- RGB dataset rendering, geometric distortion and multi-region thermal objects
  remain incomplete.
- The ground-truth contract passed a bounded 20-frame pilot; the resumable,
  sharded generator and required 500-frame pilot are not implemented yet.
- The real-only reference baseline, synthetic-only baseline, low-real-data
  experiments and ablations have not run.
- Thermal outputs are physically motivated configuration priors, not calibrated
  equivalence to a named FLIR camera.

The LWIR model is intended to be physically motivated, not a calibrated clone
of any commercial camera. Real benchmark data must not be used for iterative
model selection.

## Architecture

```text
scene + camera + materials + seed
                 |
                 v
          RGB / geometry
                 |
                 v
      thermal + sensor pipeline
                 |
                 v
     versioned synthetic dataset
                 |
        +--------+--------+
        |                 |
        v                 v
 synthetic training   real FLIR data
        |                 |
        +--------+--------+
                 v
              RT-DETRv2
                 |
                 v
        frozen real evaluation
```

See [Architecture](docs/ARCHITECTURE.md),
[Data Strategy](docs/DATA_STRATEGY.md), and
[Reproducibility](docs/REPRODUCIBILITY.md) for the main contracts.

## Setup

Requirements:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

From the repository root:

```bash
uv sync --frozen --extra train
uv run spectratwin --help
uv run spectratwin env doctor
```

The real-data smoke command requires operator-supplied FLIR manifests and
dataset roots. A portable CPU example is:

```bash
export SPECTRATWIN_PROFILE=cpu-dev
export SPECTRATWIN_MASTER_SEED=0
export SPECTRATWIN_DATA_ROOT=/path/to/data
export SPECTRATWIN_CACHE_ROOT=/path/to/cache
export SPECTRATWIN_ARTIFACT_ROOT=/path/to/run-output

uv run spectratwin train real-smoke \
  --train-manifest /path/to/real_train.json \
  --dev-manifest /path/to/real_dev.json \
  --flir-train-root /path/to/flir-train \
  --flir-dev-root /path/to/flir-dev \
  --device cpu
```

These paths are runtime inputs and are never part of portable dataset or run
identity. The smoke command proves execution, not benchmark quality.

Run the quality suite:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
```

## Data and model artifacts

Datasets, model weights, checkpoints and run artifacts are intentionally not
stored in Git. Configure roots at runtime and keep licensed FLIR content outside
the repository. See [Real Data Card](docs/REAL_DATA_CARD.md) and
[Licensing and Provenance](docs/LICENSING_AND_PROVENANCE.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Configuration Standard](docs/CONFIGURATION_STANDARD.md)
- [Technology Stack](docs/TECH_STACK.md)
- [Thermal Model](docs/THERMAL_MODEL.md)
- [Sensor Model](docs/SENSOR_MODEL.md)
- [Test Strategy](docs/TEST_STRATEGY.md)
- [References](docs/REFERENCES.md)

## License

Code in this repository is licensed under the MIT License. External datasets,
assets and pretrained weights retain their own terms and are not redistributed
by this repository.
