"""Real-only smoke training run for a tiny fixture or subset.

Wires the frozen manifest and FLIR source root through the training dataset
adapter into one RT-DETRv2 train/eval loop, logging the identity needed to
reconstruct the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, Subset

from spectratwin.config.settings import Settings
from spectratwin.contracts.environment_report import collect_environment_report
from spectratwin.randomness.seed import derive_subseed
from spectratwin.real_data.manifest import DatasetManifest, read_manifest
from spectratwin.training.config import RealSmokeTrainConfig
from spectratwin.training.dataset import load_training_dataset
from spectratwin.training.hardware import collect_training_hardware_report
from spectratwin.training.model import build_pretrained_model

ModelFactory = Callable[[str, str], tuple[Any, Any]]
CHECKPOINT_SCHEMA_VERSION = "spectratwin-training-checkpoint-v1"


class RealSmokeTrainResult(BaseModel):
    run_id: str
    mlflow_run_id: str
    checkpoint_path: Path
    train_dataset_fingerprint: str
    dev_dataset_fingerprint: str
    git_sha: str | None
    seed: int
    final_train_loss: float
    final_eval_loss: float
    train_sample_count: int
    dev_sample_count: int
    resumed_from_step: int
    completed_steps: int


def _load_manifest(path: Path) -> DatasetManifest:
    return read_manifest(path)


def _make_collate_fn(processor: Any) -> Callable[[list[tuple[Any, dict[str, Any]]]], Any]:
    def collate(batch: list[tuple[Any, dict[str, Any]]]) -> Any:
        images, targets = zip(*batch, strict=True)
        return processor(images=list(images), annotations=list(targets), return_tensors="pt")

    return collate


def _optimization_identity(config: RealSmokeTrainConfig) -> str:
    """Hash fields that must remain stable when a run resumes elsewhere."""
    identity = {
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_revision": config.checkpoint_revision,
        "max_train_samples": config.max_train_samples,
        "max_dev_samples": config.max_dev_samples,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _save_checkpoint_atomically(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(bundle, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _checkpoint_bundle(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    run_id: str,
    config_identity: str,
    seed: int,
    train_fingerprint: str,
    dev_fingerprint: str,
    git_sha: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "completed_steps": completed_steps,
        "config_identity": config_identity,
        "seed": seed,
        "train_dataset_fingerprint": train_fingerprint,
        "dev_dataset_fingerprint": dev_fingerprint,
        "git_sha_at_save": git_sha,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def _load_resume_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    run_id: str | None,
    config_identity: str,
    seed: int,
    train_fingerprint: str,
    dev_fingerprint: str,
    device: torch.device,
) -> tuple[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    bundle = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported resume checkpoint; expected schema {CHECKPOINT_SCHEMA_VERSION!r}"
        )

    expected = {
        "config_identity": config_identity,
        "seed": seed,
        "train_dataset_fingerprint": train_fingerprint,
        "dev_dataset_fingerprint": dev_fingerprint,
    }
    mismatches = [name for name, value in expected.items() if bundle.get(name) != value]
    if run_id is not None and bundle.get("run_id") != run_id:
        mismatches.append("run_id")
    if mismatches:
        raise ValueError(
            "resume checkpoint identity mismatch for: " + ", ".join(sorted(mismatches))
        )

    checkpoint_run_id = bundle.get("run_id")
    completed_steps = bundle.get("completed_steps")
    if not isinstance(checkpoint_run_id, str) or not checkpoint_run_id:
        raise ValueError("resume checkpoint has invalid run_id")
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise ValueError("resume checkpoint has invalid completed_steps")

    model.load_state_dict(bundle["model_state_dict"])
    optimizer.load_state_dict(bundle["optimizer_state_dict"])
    return checkpoint_run_id, completed_steps


def _training_batch_for_step(
    dataset: Subset[Any], processor: Any, batch_size: int, seed: int, step: int
) -> Any:
    """Select a deterministic batch by logical step, independent of runtime restarts."""
    generator = torch.Generator()
    generator.manual_seed(derive_subseed(seed, "train-step", str(step)))
    permutation = torch.randperm(len(dataset), generator=generator)
    indices = permutation[: min(batch_size, len(dataset))].tolist()
    step_dataset = Subset(dataset, indices)
    loader = DataLoader(
        step_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_collate_fn(processor),
    )
    return next(iter(loader))


def run_real_smoke_training(
    config: RealSmokeTrainConfig,
    settings: Settings,
    model_factory: ModelFactory = build_pretrained_model,
) -> RealSmokeTrainResult:
    """Run one real-only smoke train/eval pass and record it to MLflow.

    Not a reportable baseline: ``max_train_samples``/``max_steps`` prove the
    pipeline executes end to end, they do not produce a meaningful metric
    or a frozen-benchmark baseline number.
    """
    if settings.artifact_root is None:
        raise ValueError("settings.artifact_root is required for training")

    train_manifest = _load_manifest(config.train_manifest_path)
    dev_manifest = _load_manifest(config.dev_manifest_path)

    train_dataset = load_training_dataset(train_manifest, config.flir_train_root)
    dev_dataset = load_training_dataset(dev_manifest, config.flir_dev_root)

    train_subset = Subset(train_dataset, range(min(config.max_train_samples, len(train_dataset))))
    dev_subset = Subset(dev_dataset, range(min(config.max_dev_samples, len(dev_dataset))))

    # Seed before model construction: a fresh/mismatched-head model
    # reinitializes weights on load, which must be reproducible too.
    seed = derive_subseed(settings.master_seed, "training", "real-smoke")
    torch.manual_seed(seed)

    hardware = collect_training_hardware_report(precision="fp32")
    device = torch.device(config.device)
    if device.type == "cuda" and (not hardware.cuda_available or hardware.cuda_compatible is False):
        raise RuntimeError(
            "CUDA device requested but PyTorch CUDA is unavailable or incompatible; "
            "run `spectratwin env doctor --profile colab` before remote compute"
        )

    model, processor = model_factory(config.checkpoint_id, config.checkpoint_revision)
    model.to(device)

    collate_fn = _make_collate_fn(processor)
    dev_loader = DataLoader(
        dev_subset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    git_sha = collect_environment_report().git_sha
    config_identity = _optimization_identity(config)
    run_id = config.run_id or uuid.uuid4().hex[:12]
    resumed_from_step = 0
    if config.resume_from_checkpoint is not None:
        run_id, resumed_from_step = _load_resume_checkpoint(
            config.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            run_id=config.run_id,
            config_identity=config_identity,
            seed=seed,
            train_fingerprint=train_manifest.fingerprint,
            dev_fingerprint=dev_manifest.fingerprint,
            device=device,
        )

    checkpoint_dir = settings.artifact_root / "checkpoints" / run_id
    checkpoint_path = checkpoint_dir / "model.pt"

    # NOTE: do not configure `settings.artifact_root` (or any of its parent
    # directories) to be literally named "artifacts" on disk. MLflow 3.9.0's
    # FileStore fails to create a run (raises "Run '<id>' not found" inside
    # create_run) whenever any ancestor path component of the tracking URI
    # is exactly "artifacts" - reproduced independently of this codebase.
    mlflow.set_tracking_uri(f"file:{settings.artifact_root / 'mlruns'}")
    mlflow.set_experiment("real-only-smoke")

    final_train_loss = float("nan")
    with mlflow.start_run(run_name=run_id) as active_run:
        mlflow.log_params(
            {
                **config.model_dump(mode="json"),
                "seed": seed,
                "git_sha": git_sha,
                "train_dataset_fingerprint": train_manifest.fingerprint,
                "dev_dataset_fingerprint": dev_manifest.fingerprint,
                "config_identity": config_identity,
                "logical_run_id": run_id,
                "resumed_from_step": resumed_from_step,
                **hardware.model_dump(),
            }
        )

        model.train()
        step = resumed_from_step
        while step < config.max_steps:
            batch = _training_batch_for_step(
                train_subset, processor, config.batch_size, seed, step
            ).to(device)
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            # RT-DETR's official config clips to 0.1: an unclipped step
            # on the (mismatched-size, reinitialized) detection head can
            # produce NaN weights on the very next forward pass.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            final_train_loss = float(loss.detach().cpu())
            mlflow.log_metric("train_loss", final_train_loss, step=step)
            step += 1
            _save_checkpoint_atomically(
                checkpoint_path,
                _checkpoint_bundle(
                    model=model,
                    optimizer=optimizer,
                    completed_steps=step,
                    run_id=run_id,
                    config_identity=config_identity,
                    seed=seed,
                    train_fingerprint=train_manifest.fingerprint,
                    dev_fingerprint=dev_manifest.fingerprint,
                    git_sha=git_sha,
                ),
            )

        model.eval()
        eval_losses: list[float] = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                outputs = model(**batch)
                eval_losses.append(float(outputs.loss.detach().cpu()))
        final_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else float("nan")
        mlflow.log_metric("eval_loss", final_eval_loss)

        if not checkpoint_path.exists():
            _save_checkpoint_atomically(
                checkpoint_path,
                _checkpoint_bundle(
                    model=model,
                    optimizer=optimizer,
                    completed_steps=step,
                    run_id=run_id,
                    config_identity=config_identity,
                    seed=seed,
                    train_fingerprint=train_manifest.fingerprint,
                    dev_fingerprint=dev_manifest.fingerprint,
                    git_sha=git_sha,
                ),
            )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoint")

        mlflow_run_id = active_run.info.run_id

    return RealSmokeTrainResult(
        run_id=run_id,
        mlflow_run_id=mlflow_run_id,
        checkpoint_path=checkpoint_path,
        train_dataset_fingerprint=train_manifest.fingerprint,
        dev_dataset_fingerprint=dev_manifest.fingerprint,
        git_sha=git_sha,
        seed=seed,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        train_sample_count=len(train_subset),
        dev_sample_count=len(dev_subset),
        resumed_from_step=resumed_from_step,
        completed_steps=step,
    )
