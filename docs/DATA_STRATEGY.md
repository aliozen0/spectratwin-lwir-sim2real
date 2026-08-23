# Data Strategy

## 1. Data philosophy

The project studies data quality and domain transfer, so dataset provenance and split integrity are first-class concerns. Every image used for training or evaluation must be attributable to a dataset version and split manifest.

## 2. Primary real benchmark: Teledyne FLIR ADAS

The FLIR ADAS dataset is selected because the official dataset provides annotated thermal and visible data, COCO-format annotations and pre-AGC thermal frames. The project maps its taxonomy to only:

- Person → `person`
- Car → `car`
- Bike → `bicycle`

All other categories are ignored for v1, with ignore behavior documented and validated.

Do not copy FLIR files into the repository. Provide preparation code/instructions later that operate on a user-supplied local download.

## 3. Benchmark split policy

Use the official suggested train/validation split where it cleanly meets the research design. For any video-derived frames or custom split, avoid random frame-level splitting that could place near-adjacent frames across train/test. Prefer sequence/clip-level separation.

Create three conceptual roles:

- `real_train`: available for training/fine-tuning.
- `real_dev`: allowed for implementation debugging/hyperparameter sanity.
- `real_benchmark`: frozen evaluation; never used for iterative model selection.

Every real-data manifest stores stable source IDs and an aggregate fingerprint
covering normalized annotations and referenced image SHA-256 values so both the
split and training-effective source content are auditable.

## 4. Real-data fractions

For the label-efficiency experiment derive deterministic subsets from `real_train`:

- 10%
- 25%
- 50%
- 100%

Subsets should be nested when practical (`10 ⊂ 25 ⊂ 50 ⊂ 100`) and stratified/constructed to avoid pathological class loss. Record subset manifests and seeds.

## 5. Synthetic dataset stages

### Smoke
~20 samples. Proves contracts and visualization.

### Pilot
~500 samples. Used for data validator, distribution comparison and short training.

### Development
~2,000 samples. Used for tuning generation priors and catching systemic artifacts.

### Main research
Target ~10,000 valid paired samples. Increase toward ~20,000 only if compute budget and evidence justify it.

More samples are not automatically better. Distribution and sensor assumptions matter more than a vanity image count.

## 6. Synthetic class/domain priors

Initial priors should be informed by real-data EDA rather than arbitrary uniform randomization. Compare at least:

- classes per frame,
- bbox pixel area and height,
- object distance proxy,
- aspect ratios,
- occlusion/truncation,
- camera FOV/height assumptions,
- day/night conditions,
- intensity/contrast statistics.

The goal is not to copy the real dataset exactly; it is to avoid avoidable distribution mismatch while preserving useful domain randomization.

## 7. Dataset contract

Each valid sample must have:

- RGB output (when enabled),
- raw/linear thermal proxy,
- AGC/display thermal image,
- depth,
- instance/semantic mask as configured,
- COCO annotation linkage,
- `FrameRecord` metadata,
- reproducibility fingerprint.

## 8. Validation gate

Reject/quarantine samples if:

- an output is missing/corrupt,
- dimensions mismatch the manifest,
- bounding boxes leave image bounds or have non-positive area,
- masks/classes disagree,
- required camera matrices are invalid,
- NaN/Inf values exist where prohibited,
- thermal values violate configured output domain,
- metadata does not satisfy schema.

Training consumes only a successfully validated dataset manifest.

## 9. DVC strategy

Git tracks small manifests/configs/code. DVC tracks large generated/prepared datasets and derived outputs. Dataset identifiers should remain human readable (`syn-v0.3-sensor`) while the manifest also stores content/config fingerprints.

## 10. Dataset card

The final synthetic dataset card must state:

- intended use,
- source/provenance,
- license/redistribution limits,
- taxonomy,
- generation process,
- splits,
- known biases,
- synthetic assumptions,
- validation checks,
- versions used in final experiments.


## Physical storage is not dataset identity

The same prepared dataset version may live in WSL cache, persistent remote archive and Colab `/content`. These are replicas/staging locations, not different dataset versions. Dataset identity comes from manifest/schema/fingerprint.

## Remote staging

For Colab, prefer moving prepared data as checksummed archives or deterministic shards to local `/content` before high-frequency reads. The training/dataset APIs receive the local staged root and remain unaware of Google Drive semantics.
