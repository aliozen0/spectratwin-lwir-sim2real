# Licensing and Provenance Policy

## Principle

A professional synthetic-data repository must be able to explain where every non-original asset, dataset and pretrained weight came from and what can be redistributed.

## Real datasets

Do not redistribute the FLIR dataset in Git or release artifacts unless its terms explicitly permit the specific redistribution. Store instructions/adapters/manifests, not the source files.

## 3D assets

Prefer assets with clearly documented permissive/CC0-like licensing. For every asset record:

- stable asset ID,
- source URL/reference,
- author/provider,
- license identifier/text reference,
- download date/version,
- modifications,
- whether redistribution is allowed,
- checksum.

Unknown-license assets are not allowed in the release dataset/demo.

## Pretrained model weights

Record model name, source, upstream license and exact checkpoint identifier. Do not silently vendor weights into Git.

## Code dependencies

Retain upstream notices where required. Generate a dependency inventory/SBOM for release if convenient; `uv` can export standardized dependency representations, but license review remains a separate responsibility.

## Public generated samples

Before publishing synthetic examples, ensure no restricted source texture/asset license prevents redistribution of rendered derivative content. When uncertain, replace the asset with a clearly permissive alternative.

## THIRD_PARTY_NOTICES

The eventual repo should contain a `THIRD_PARTY_NOTICES.md` listing assets, datasets, model code/weights and any relevant attribution requirements.
