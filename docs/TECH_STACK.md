# Technology Stack and Selection Rationale

## Selection principles

- Prefer mature tools with clear official documentation.
- Avoid duplicating capabilities that are not the research contribution.
- Pin exact resolved dependencies after compatibility is verified.
- Keep Blender runtime and ML runtime separable.

## Rendering: Blender + BlenderProc + Cycles

BlenderProc is selected because it already supports procedural object/camera/material handling and output modalities such as RGB, depth, normals and segmentation, plus COCO writing. This allows project effort to focus on the thermal/sensor/sim-to-real research contribution rather than rebuilding generic synthetic-data plumbing.

Cycles is preferred for RGB/PBR fidelity. The planned thermal rendering path
will not simply convert RGB to grayscale; it will use a separate physically
motivated response.

## Shader: OSL where appropriate

OSL provides an explicit shader implementation path for thermal response logic. The scientific reference should first exist as testable Python/Numpy equations. Shader outputs are cross-checked on controlled cases rather than treated as self-validating.

## Python target

Application code targets Python 3.11 unless Blender/BlenderProc compatibility requires a stricter runtime boundary. Do not force one Python environment to satisfy incompatible renderer and trainer constraints; use separate environment/container definitions.

## Dependency management: uv

`uv` is used for project dependency management and lockfile-based reproducibility. Commit `uv.lock`. CI should check that the lockfile is current. Avoid ad-hoc `pip install` commands that create state not represented in project metadata.

## Config composition: Hydra

Hydra supports composable configuration groups and command-line overrides. Use it to express orthogonal experiment choices (scene/camera/thermal/sensor/dataset/training) without creating giant duplicated YAML files.

Hydra is not a replacement for domain validation. Final composed runtime config should be validated against typed schemas and cross-field invariants.

## Runtime contracts: Pydantic v2

Pydantic is used for persisted records and externally sourced data because validation and JSON schema support make contracts explicit. Prefer strict validation for IDs, schema versions, physical ranges and paths that must not be silently coerced.

## Model: RT-DETRv2

Use one RT-DETRv2 model size that fits the available GPU. The project is a data/sim-to-real study, not an architecture benchmark. Keep the model fixed across main comparisons.

Hugging Face Transformers supports RT-DETRv2 fine-tuning/inference; the upstream RT-DETR repository also provides a PyTorch implementation. Pick one integration and stick to it for v1 to avoid reproducibility drift.

## Evaluation: COCO metrics

Primary: `AP@[0.50:0.95]` on the frozen real benchmark. Report AP50/AP75 and per-class AP. Add slice analysis without replacing the primary metric.

## Data versioning: DVC

Use DVC to version large dataset artifacts and preprocessing outputs without committing them directly to Git. `dvc.yaml` may later encode repeatable data stages. Every experiment references a dataset fingerprint/version.

## Experiment tracking: MLflow

MLflow records parameters, metrics, artifacts, dataset references and code provenance. Start with a local tracking setup; use PostgreSQL/object storage only when it improves reproducibility/team-like workflow rather than as infrastructure theatre.

Recommended maturity:

1. local MLflow first,
2. Docker Compose tracking server when training is stable,
3. PostgreSQL + MinIO if useful for the portfolio/demo.

## Code quality

- Ruff: formatting + linting.
- Pyright: static type checking; start strict on core contracts/physics and expand.
- pytest: unit/integration/regression/smoke tests.
- pre-commit: fast local checks before commits.
- GitHub Actions: fast CPU-safe validation on pull requests.

## Tools intentionally deferred

- Airflow/Prefect: not needed for a single-machine research pipeline.
- Kubernetes: no v1 operational need.
- Spark: dataset scale does not justify it.
- Web dashboard: generated reports + MLflow UI are sufficient.
- Multiple experiment trackers: one tracker only.

## Development and remote compute stack

### Development

- Linux-first Git/Python development environment.
- CPU or a compatible local GPU for inexpensive smoke/debug work.

### Heavy compute

- Google Colab managed runtime for GPU-intensive training/rendering experiments.
- Preferred accelerator: A100 when actually allocated and detected.
- Colab local `/content`: hot ephemeral job scratch.
- Google Drive initially: persistent transport/artifact storage, not hot per-file training I/O.

### Environment portability

No core module may depend on personal WSL paths or Colab-specific paths. Execution profiles and bootstrap layers translate environment-local storage into portable application roots.
