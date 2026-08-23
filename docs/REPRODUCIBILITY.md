# Reproducibility Standard

## Goal

A reviewer should be able to answer: “Which exact code, data, configuration, assets and random seeds produced this figure/model?”

## Reproducibility tuple

For every dataset version:

```text
(code Git SHA,
 resolved generation config hash,
 asset manifest hash,
 schema version,
 master seed,
 sample index range)
```

For every training run:

```text
(code Git SHA,
 training config,
 training dataset fingerprint,
 benchmark manifest fingerprint,
 model initialization,
 seed)
```

## Dependency reproducibility

- Commit `uv.lock` for normal Python environments.
- CI uses locked/frozen dependency install behavior.
- Renderer/trainer container images should be tagged and preferably record immutable base/image digests in final release notes.
- Never rely on “pip install latest”.

## Randomness

Create an explicit seed hierarchy:

```text
master_seed
  ├─ scene_seed(sample_index)
  ├─ camera_seed(sample_index)
  ├─ thermal_seed(sample_index)
  └─ sensor_seed(sample_index)
```

Derive sub-seeds deterministically using a stable documented method. Do not use Python's process-dependent hash for persistent seed derivation.

## Resolved configs

Every dataset/train run persists the fully resolved config, not just the original YAML fragments. This prevents later confusion when config defaults change.

## Fingerprints

Prefer cryptographic hashes of canonicalized manifests/configs for identity. Do not hash mutable absolute local paths into portable identities unless necessary.

Real-data manifest schema `spectratwin-real-manifest-v2` hashes canonicalized
portable record fields, normalized project annotations and SHA-256 digests of
referenced image bytes. It does not store licensed image/label content or local
absolute roots. Schema-less membership-only manifests are legacy evidence and
must be regenerated from a fresh scan before training.

## Reproduction levels

### L1 — Functional
Commands run and produce valid outputs.

### L2 — Deterministic data
Same dataset inputs/config/seed reproduce the same sample metadata and ideally identical deterministic outputs under a documented environment.

### L3 — Experiment lineage
A reported metric links to exact dataset/model/code/config.

### L4 — Clean-room audit
The release is reproduced from a clean clone/environment using documented steps.

Portfolio v1 should meet at least L3 and demonstrate L4 for the smoke path.


## Cross-environment reproducibility

A run may be authored in WSL and executed in Colab. Therefore environment-specific path strings are excluded from scientific identity where possible. Record both:

- **portable identity:** Git SHA, config hash, dataset/asset fingerprints, seed, model config;
- **execution evidence:** OS, hostname/runtime class, actual GPU, VRAM, CUDA, PyTorch/Python versions, precision mode.

The actual GPU is measured at runtime; an expected A100 allocation is never written as if observed.

## Ephemeral-runtime reproducibility

A Colab runtime disappearing must not destroy the ability to reproduce/resume a run. The minimum recovery tuple is:

```text
(Git ref,
 resolved config,
 data/assets persistent identity,
 run ID,
 last persisted checkpoint/shard state)
```

## Notebook reproducibility

Colab notebooks contain orchestration only. Any algorithmic code used to produce a reported result must exist in the Git-tracked package at the recorded commit. Final metrics from notebook-only experimental edits are not release-eligible.
