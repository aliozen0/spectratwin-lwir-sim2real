# Architecture

## Architectural goals

1. Reproduce any valid synthetic sample from config + assets + code + seed.
2. Keep rendering, thermal physics, sensor effects, dataset writing, training and evaluation separable.
3. Allow small unit tests without launching Blender when possible.
4. Make persisted artifacts schema-versioned and self-describing.
5. Keep the research variable in the data-generation path while the model family stays fixed.

## Bounded contexts

### Configuration
Composes YAML experiment/scene/sensor choices and validates the final runtime configuration. Hydra provides composition; Pydantic/dataclass schemas enforce domain invariants where persisted/external contracts require stronger validation.

### Scene
Produces a scene description: roads, static environment, objects, placements, thermal attributes and semantic classes. Scene generation must not directly write datasets.

### Camera
Owns intrinsics, extrinsics and camera/optics parameters. Conversion between HFOV/focal representation is centralized and tested.

### Rendering
Bridges domain objects to Blender/BlenderProc. It may produce RGB, depth, masks and raw thermal shader outputs but must not contain training logic.

### Thermal
Contains the reference radiometric model and material/temperature property contracts. CPU reference code is the scientific source of truth; shader implementation is validated against controlled reference cases.

### Sensor
Transforms idealized/raw thermal signal into training/display representations using independently configurable effects: PSF/blur, distortion, noise, quantization and AGC.

### Annotation
Converts instance/semantic render products into COCO-compatible labels and checks geometry.

### Dataset
Coordinates atomic sample persistence, manifest creation, sharding, resumability and fingerprinting.

### Training
Consumes a validated dataset adapter and trains the fixed detector family. Training has no dependency on Blender.

### Evaluation
Owns COCO metrics, dataset slices, failure classification and result reports. Evaluation must never mutate the benchmark dataset.

## Dependency direction

Core domain/data contracts should not import Blender. Preferred dependency direction:

```text
schemas/config/domain
        ^
        |
scene camera thermal sensor annotation
        ^
        |
render adapters / blenderproc
        ^
        |
dataset orchestration

training/evaluation -> dataset adapters/contracts
```

Avoid `utils.py` becoming a dumping ground. Domain-specific utilities live with their subsystem.

## Process boundaries

Renderer and trainer should be treated as separate runtime environments. Blender/BlenderProc can impose its own Python/runtime constraints; PyTorch/CUDA dependencies evolve independently. Artifacts and schemas connect the two processes.

```text
[Renderer container/process]
        |
        v
Versioned dataset artifacts
        |
        v
[Trainer container/process]
        |
        v
MLflow artifacts + result reports
```

## Persisted artifacts

### FrameRecord
Minimum provenance:

- schema_version
- sample_id
- master_seed/sample_seed
- generator version or Git SHA
- resolved config hash
- asset manifest fingerprint
- scene/camera/thermal/sensor fields
- object list and class mapping
- output file checksums/relative paths
- generation status and timestamps

### DatasetManifest
- dataset ID/version
- schema version
- generator commit/config fingerprint
- asset fingerprint
- shard count
- sample counts: requested/valid/failed
- class taxonomy
- output modalities
- aggregate statistics location

### ExperimentRecord / MLflow tags
- run name/ID
- code Git SHA
- model ID/checkpoint
- training dataset fingerprint
- real dataset split fingerprint
- full resolved config artifact
- seed
- metrics and artifacts

## Atomicity

A sample is rendered to a temporary directory, validated, checksummed and atomically renamed into the valid dataset namespace. Interrupted temp directories are not listed in the final manifest.

## Resumability

On restart, orchestrator verifies an existing sample against expected metadata/fingerprint. Valid samples are skipped; incomplete/mismatched samples are regenerated or quarantined. Never blindly trust file existence.

## Scaling

Scale by shards, not by introducing a distributed platform early. A shard is a deterministic sample-index range. Separate workers can render independent shards; a final manifest merge verifies no duplicated IDs and compatible schema/config fingerprints.

## Observability

Use structured events such as `generation_started`, `sample_completed`, `sample_failed`, `shard_completed`, `validation_failed`. Record durations, sample IDs and reproducibility metadata. Do not log secrets or full proprietary paths when unnecessary.

## Deployment / execution topology

SpectraTwin's runtime topology is part of the architecture, not an afterthought:

```text
WSL2 local environment
    |
    v
WSL2 development control plane
    |  Git SHA + config + manifests
    v
Colab ephemeral worker
    |  versioned artifacts/checkpoints/shards
    v
Persistent artifact/data storage
    |
    v
WSL result ingestion and release
```

### Control plane vs compute plane

WSL acts as the local control plane: source, specifications, validation, run definitions and final analysis records. Colab acts as a replaceable compute plane. Scientific code does not branch on "running in Colab" except through declared execution/storage/resource adapters.

### Remote job contract

A remote job must be reconstructable from:

- exact Git ref,
- resolved config,
- dataset fingerprint,
- asset fingerprint when rendering,
- seed(s),
- persisted resume state where relevant.

Runtime-local absolute paths are not scientific identity.

### Storage adapters

Domain/training code receives dataset/artifact roots via configuration. It must not import Google Drive APIs or assume `/content/drive`. Notebook/bootstrap code handles staging into a normal local filesystem path; the application sees a portable path.

### Hardware abstraction

The system records actual detected hardware but should not make research
semantics depend on a specific local or remote GPU. Resource-only knobs
(micro-batch, workers, precision) are separated from experiment-defining knobs.
