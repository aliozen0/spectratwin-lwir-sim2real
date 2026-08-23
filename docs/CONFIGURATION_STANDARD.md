# Configuration Standard

## Principle

Configuration is an externalized experiment contract, not a dumping ground for arbitrary knobs.

## Groups

Recommended Hydra groups:

- scene
- camera
- thermal
- sensor
- dataset
- model
- training
- experiment

## Rules

- Every config key has a type and documented unit where applicable.
- Physical values use explicit suffixes where ambiguity is likely (`temperature_k`, `distance_m`, `hfov_deg`).
- Cross-field invariants are validated after composition.
- Final resolved config is persisted with every run/dataset.
- Do not reference personal absolute paths in committed defaults.
- Secrets do not belong in YAML.
- Changing default behavior that affects comparability must be documented and
  considered in dataset versioning.

## Overrides

CLI overrides are useful for development, but final experiments should save the resolved result. A command alone is not sufficient provenance because defaults can later change.


## Environment-specific roots

Committed experiment configs MUST NOT contain one developer's WSL or Colab absolute paths. Use logical roots/IDs plus environment-local settings.

Example conceptual separation:

```yaml
# committed experiment config
dataset:
  id: flir-prepared-v1

# local uncommitted/runtime environment mapping
storage:
  datasets_root: /home/<user>/data/spectratwin-cache
```

On Colab the same dataset ID may resolve to `/content/data`. The resolved config artifact may record the execution path for debugging, but portable identity/fingerprint must not depend on that personal path.
