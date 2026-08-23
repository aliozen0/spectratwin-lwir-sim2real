# Camera and Thermal Sensor Model

## Objective

Separate idealized thermal scene response from camera/sensor artifacts so each factor can be independently enabled, configured and ablated.

## Camera intrinsics

The camera contract stores either an intrinsic matrix or sufficient parameters to derive it, including:

- width/height,
- horizontal/vertical FOV or focal representation,
- principal point,
- optional skew if ever required,
- distortion coefficients when enabled.

The primary FLIR ADAS reference camera published by the dataset uses a 640×512 thermal camera and approximately 45° HFOV. Treat those values as benchmark-informed defaults, not proof that the synthetic sensor is calibrated to that device.

## Camera extrinsics

Persist a clearly documented transform convention. Choose one representation for stored world↔camera matrices and test round trips. Record units and coordinate-system conversion at the Blender boundary.

## Optical/sensor stages

Recommended stage order:

```text
raw band response
→ optical PSF / blur
→ geometric distortion
→ spatial/sensor noise
→ quantization
→ AGC / display mapping
```

Actual ordering should be documented and kept stable for experiments.

## PSF/blur

Start with a small configurable Gaussian approximation. Parameters should be expressed in pixel-domain sigma or a clearly documented proxy. The goal is controlled loss of spatial detail, not pretending to model a complete MTF.

## Distortion

Use a standard radial/tangential camera model if needed. Do not randomize distortion to physically impossible values. Store the applied coefficients in frame metadata.

## Noise

V1 can support simple signal-independent and/or signal-dependent proxies. Every random draw must derive from the sample seed. Tests should verify deterministic replay.

## Quantization

Keep internal calculations floating point, then quantize explicitly into the chosen raw output representation. Record bit depth/normalization range. Avoid confusing a 16-bit PNG container with true radiometric calibration.

## AGC

Automatic gain control strongly changes thermal appearance. Implement at least one deterministic baseline method (for example percentile clipping + normalization) and an optional variability policy for ablation.

Persist AGC parameters per frame so displayed intensity transformations are reproducible.

## Ablation switches

Every stage should be independently toggleable:

- no blur / blur,
- no distortion / distortion,
- clean / noise,
- fixed / variable quantization behavior if studied,
- fixed / randomized AGC.

Ablation must change one conceptual factor at a time whenever feasible.
