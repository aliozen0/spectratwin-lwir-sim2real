# Real Data Card — FLIR ADAS v2 Benchmark

Provenance and statistics for the real-world reference used across sim-to-real
experiments. This card records the frozen FLIR ADAS v2 split policy.

## Source

- Dataset: Teledyne FLIR Free ADAS Thermal Dataset v2.
- Obtained: Kaggle mirror `samdazel/teledyne-flir-adas-thermal-dataset-v2`,
  2026-08-23. Description and license-agreement link match the official
  Teledyne FLIR release; `video_thermal_test` frame count (3,749) matches
  the official README exactly, cross-checked as a mirror-integrity signal.
- License: subject to
  https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/.
  Not redistributed; it must remain under an operator-configured data root
  outside this repository, per docs/LICENSING_AND_PROVENANCE.md.
- Local footprint: ~12 GB extracted.

## Taxonomy mapping

Project taxonomy (`spectratwin.real_data.taxonomy`, mapping version
`flir-taxonomy-v1`): FLIR `person` -> `person`, `car` -> `car`,
`bike` -> `bicycle`. The other 12 FLIR categories are explicitly ignored,
not remapped.

## Split policy

FLIR's official splits are used directly (no re-splitting):

| role            | FLIR split            | frames | rationale |
|-----------------|------------------------|-------:|-----------|
| `real_train`    | `images_thermal_train` | 10,742 | training/fine-tuning |
| `real_dev`      | `images_thermal_val`   |  1,144 | iterative debugging/hyperparameter sanity |
| `real_benchmark`| `video_thermal_test`   |  3,749 | frozen eval; FLIR samples this split from independent video sequences, avoiding adjacent-frame leakage by construction |

`real_benchmark` is frozen: its manifest was written once and
`write_manifest` refuses to overwrite it. It must never be used for
iterative model selection.

## Statistics (person / car / bicycle counts, `compute_eda` output)

| role            | person | car    | bicycle | frames w/o annotation |
|-----------------|-------:|-------:|--------:|-----------------------:|
| `real_train`    | 50,478 | 73,623 |   7,237 | 519 / 10,742 |
| `real_dev`      |  4,470 |  7,133 |     170 |  48 / 1,144 |
| `real_benchmark`| 12,323 | 30,517 |     113 | 256 / 3,749 |

`scan_flir_dataset` reported 0 issues (no missing images, no invalid
bboxes, no corrupt annotations) across all three splits.

## Manifests

Operator-created manifests remain under the configured data root, outside this
repository. Schema `spectratwin-real-manifest-v2` stores its schema/fingerprint
algorithm versions, mapping/split versions, seed, sorted sample IDs and one
aggregate SHA-256 fingerprint. The fingerprint covers portable record fields,
normalized project annotations and each referenced image's SHA-256 value; it
does not store FLIR bytes, annotation payloads or absolute local paths.

Earlier schema-less manifests and their membership-only fingerprints remain
historical evidence, not valid training inputs. Regenerate v2 manifests from a
fresh scan into new files. In particular, never overwrite the frozen benchmark
manifest automatically; write a side-by-side v2 candidate and freeze it only
after human comparison of membership and scan issues.

## Known limitations

- v2 integrity verification streams every referenced image byte and is
  therefore O(dataset size); run it against staged execution-local storage,
  not a many-small-file mounted remote drive.
- Sequence disjointness was checked and holds across all three roles:
  `real_train` (133 video sequences), `real_dev` (17 sequences) and
  `real_benchmark` (8 sequences) share zero `video_id` values pairwise.
- Full 15-category FLIR annotation quality (occlusion/truncation flags,
  segmentation polygons) is not used; only bbox + the 3-class taxonomy.
