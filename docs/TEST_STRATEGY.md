# Test Strategy

## Pyramid

Most tests should run without Blender/GPU. Expensive rendering/training smoke tests are few and explicit.

### Unit

- Pydantic/config validators,
- seed derivation,
- camera matrices/projection helpers,
- bbox/mask geometry,
- Planck/band radiance,
- sensor transforms,
- manifest fingerprints.

### Integration

- scene description → render adapter → tiny output,
- render outputs → annotation writer → validator,
- prepared dataset → data loader,
- one training/evaluation step.

### Regression

Use tiny deterministic fixtures to detect unexpected changes in metadata, annotation counts, reference-radiance values or selected image statistics. Avoid fragile exact-image comparisons across GPU/renderer versions unless environment is strictly fixed.

### Smoke

- `render-smoke`: a handful of samples,
- `train-smoke`: one/few optimizer steps,
- `eval-smoke`: deterministic mini metric path.

## Test naming

Tests describe domain behavior: `test_band_radiance_increases_with_temperature`, not `test_function1`.

## CI split

Fast CI runs CPU-safe unit/schema/quality tests. Render/GPU tests can be optional/manual/self-hosted and must not make normal contributions impossible.


## Deployment-boundary tests

Add cheap tests for:

- path registry can resolve different local roots to the same portable dataset ID,
- environment doctor works with no GPU present,
- environment doctor parses/reports GPU facts when available,
- resume logic works when local scratch starts empty but persisted checkpoint/shard state exists.

A periodic Colab smoke run is an integration/deployment test, not a substitute for unit tests.
