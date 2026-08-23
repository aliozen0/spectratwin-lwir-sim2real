# Security and Secrets

SpectraTwin is primarily a local research tool, but it still handles external downloads, large files, optional tracking services and containerized workloads.

## Rules

- Never commit access tokens, MLflow/S3 credentials, API keys or signed download links.
- Store local secrets in environment variables or an ignored `.env`; commit only `.env.example` later.
- Treat external model weights, datasets and 3D assets as untrusted inputs until provenance is checked.
- Pin/lock application dependencies and review updates rather than tracking floating `latest` versions.
- Do not run arbitrary scripts bundled with downloaded assets.
- Validate archive extraction paths to avoid path traversal if automated downloads are implemented.
- Avoid privileged containers; renderer/trainer containers should run with minimal required mounts.
- Dataset directories should be mounted read-only where mutation is not required.

## Reporting

For a public repository, use GitHub private vulnerability reporting when enabled. Do not disclose secrets or privately licensed dataset content in issues.
