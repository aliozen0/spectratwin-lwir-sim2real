# Physically Motivated LWIR Model

## Positioning

SpectraTwin does **not** claim to reproduce a specific calibrated FLIR camera. The thermal subsystem is a physically motivated approximation designed to test whether better thermal/sensor priors improve downstream sim-to-real detection.

That limitation must remain visible in README, reports and interviews.

## Target band

The v1 model uses a configurable LWIR band, with the default research setting around **8–14 µm**. The exact sensor response is represented as a generic/configurable function unless source-backed calibration data is available.

## Reference image-formation model

A simplified spectral radiance formulation can be expressed as:

```text
observed spectral radiance
  = transmitted object emission
  + transmitted reflected environmental radiance
  + path/atmospheric radiance
```

For wavelength λ:

```text
Lλ = τ * [ ε * Bλ(To) + (1 - ε) * Bλ(Tr) ]
     + (1 - τ) * Bλ(Ta)
```

Where:

- `Bλ(T)` is Planck spectral radiance,
- `ε` is surface emissivity,
- `To` object/surface temperature,
- `Tr` reflected apparent/environment temperature,
- `Ta` atmospheric/path temperature,
- `τ` atmospheric transmission.

Band response is approximated by numerical integration over wavelength with configurable sensor response `R(λ)`:

```text
L_band = integral R(λ) * Lλ dλ
```

## Reference implementation first

Implement the scientific model in a pure Python/Numpy module before shader code. This allows unit tests that do not require Blender.

Minimum tests:

- at fixed emissivity/environment, radiance increases with object temperature,
- emissivity remains constrained to `[0, 1]`,
- calculations are finite over configured physical ranges,
- wavelength units are explicit and converted once,
- selected numerical cases remain stable as regression fixtures.

## Thermal attributes

Thermal properties are not identical to visible PBR materials. Create a separate thermal-material layer, for example:

- asphalt,
- concrete,
- painted metal,
- rubber,
- automotive glass,
- fabric/clothing proxy,
- skin proxy where visible,
- vegetation.

Each material class should define source/assumption notes and distributions rather than one undocumented constant.

## Temperature model

Avoid giving every object a single fixed temperature. Define scene-conditioned distributions:

- ambient temperature,
- road/building offsets,
- person body/clothing regions,
- vehicle body/tires/engine/exhaust proxy regions,
- bicycle/rider components.

The model can remain intentionally coarse, but variability should be explicit and reproducible by seed.

## Atmosphere

Start with a simple transmission approximation. Do not build a full atmospheric radiative-transfer system in v1. Make atmosphere effects toggleable so their downstream value can be tested by ablation.

## Shader mapping

The Blender/OSL shader is an implementation of the reference concept, not a separate scientific truth. Create controlled scenes with known temperatures/material properties and compare ordering/scaled responses to the CPU reference.

## Raw versus display image

Persist at least two concepts:

- `thermal_raw`: high-bit-depth/linearized synthetic signal proxy suitable for later processing.
- `thermal_agc`: display/training representation after AGC/normalization.

Do not destroy the raw proxy simply to create a visually pleasing thermal image.

## What constitutes physical validation here?

Without camera calibration, validation is limited to:

1. equation/unit correctness,
2. monotonic/sanity properties,
3. controlled synthetic-scene behavior,
4. statistical comparison with real thermal imagery,
5. downstream transfer experiments.

A good final report states these limits explicitly.
