"""Real-only smoke training run for a tiny fixture or subset.

Wires the frozen manifest and FLIR source root through the training dataset
adapter into one RT-DETRv2 train/eval loop, logging the identity needed to
reconstruct the run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping
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
from spectratwin.remote.staging import persist_file
from spectratwin.training.config import RealBaselineTrainConfig, RealSmokeTrainConfig
from spectratwin.training.dataset import load_training_dataset, wrap_with_train_augmentation
from spectratwin.training.hardware import collect_training_hardware_report
from spectratwin.training.model import build_pretrained_model

ModelFactory = Callable[[str, str], tuple[Any, Any]]
CHECKPOINT_SCHEMA_VERSION = "spectratwin-training-checkpoint-v1"
BASELINE_CHECKPOINT_SCHEMA_VERSION = "spectratwin-real-baseline-checkpoint-v1"

#: SPEC-010 real-only baseline: source-backed configuration prior from the
#: cited RT-DETR/RT-DETRv2 recipe's ``ema: {decay: 0.9999}``. Fixed training-
#: loop behavior, not a per-run tunable, so it is a constant, not a config field.
EMA_DECAY = 0.9999


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


class PersistedCheckpointEvidence(BaseModel):
    epoch: int
    artifact_name: str
    sha256: str
    size_bytes: int
    completion_marker_name: str


class RealBaselineCheckpointMetadata(BaseModel):
    run_id: str
    completed_epochs: int
    completed_steps: int
    target_epochs: int
    config_identity: str
    seed: int
    train_dataset_fingerprint: str
    dev_dataset_fingerprint: str
    git_sha_at_save: str
    checkpoint_id: str
    checkpoint_revision: str
    precision: str
    #: Steps already trained into ``completed_epochs`` (the next, still
    #: in-progress epoch); 0 for a checkpoint saved at an epoch boundary.
    #: Absent on checkpoints written before mid-epoch checkpointing existed.
    partial_epoch_steps: int = 0


class RealBaselineTrainResult(BaseModel):
    run_id: str
    mlflow_run_id: str
    checkpoint_path: Path
    resolved_config_path: Path
    config_identity: str
    train_dataset_fingerprint: str
    dev_dataset_fingerprint: str
    git_sha: str
    seed: int
    final_train_loss: float
    final_dev_loss: float
    train_sample_count: int
    dev_sample_count: int
    resumed_from_epoch: int
    completed_epochs: int
    completed_steps: int
    target_epochs: int
    complete: bool
    persisted_checkpoints: list[PersistedCheckpointEvidence]


def _load_manifest(path: Path) -> DatasetManifest:
    return read_manifest(path)


def _make_collate_fn(processor: Any) -> Callable[[list[tuple[Any, dict[str, Any]]]], Any]:
    def collate(batch: list[tuple[Any, dict[str, Any]]]) -> Any:
        images, targets = zip(*batch, strict=True)
        return processor(images=list(images), annotations=list(targets), return_tensors="pt")

    return collate


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in a processor batch, including nested labels."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    return value


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
            batch = _move_batch_to_device(
                _training_batch_for_step(train_subset, processor, config.batch_size, seed, step),
                device,
            )
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
                batch = _move_batch_to_device(batch, device)
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


def _baseline_config_identity(config: RealBaselineTrainConfig) -> str:
    """Hash the portable settings that define baseline optimization semantics."""
    identity = {
        "schema_version": "spectratwin-real-baseline-config-v1",
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_revision": config.checkpoint_revision,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "backbone_learning_rate": config.backbone_learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "gradient_clip_norm": config.gradient_clip_norm,
        "precision": config.precision,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _uses_zero_weight_decay(parameter_name: str) -> bool:
    normalized = parameter_name.lower()
    return parameter_name.endswith(".bias") or "norm" in normalized or ".bn" in normalized


def _build_baseline_optimizer(model: Any, config: RealBaselineTrainConfig) -> torch.optim.AdamW:
    """Build the source-backed RT-DETRv2 AdamW parameter groups."""
    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = (
            config.backbone_learning_rate if "backbone" in name.lower() else config.learning_rate
        )
        weight_decay = 0.0 if _uses_zero_weight_decay(name) else config.weight_decay
        grouped.setdefault((learning_rate, weight_decay), []).append(parameter)

    parameter_groups = [
        {
            "params": parameters,
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "weight_decay": weight_decay,
        }
        for (learning_rate, weight_decay), parameters in sorted(grouped.items())
    ]
    return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999))


def _apply_linear_warmup(
    optimizer: torch.optim.Optimizer, *, completed_steps: int, warmup_steps: int
) -> None:
    scale = 1.0 if warmup_steps == 0 else min(1.0, (completed_steps + 1) / warmup_steps)
    for group in optimizer.param_groups:
        initial_lr = float(group["initial_lr"])
        group["lr"] = initial_lr * scale


def _create_ema_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _update_ema_state(
    ema_state: dict[str, torch.Tensor], model: Any, *, step: int, warmup_steps: int
) -> None:
    """Ramp EMA decay from 0 like the cited recipe's ``ema.warmups``, so noisy
    early-training updates do not dominate the average (a fresh model's EMA
    starting at full decay would barely move for thousands of steps)."""
    tau = max(warmup_steps, 1)
    decay = EMA_DECAY * (1.0 - math.exp(-step / tau))
    model_state = model.state_dict()
    with torch.no_grad():
        for name, averaged in ema_state.items():
            current = model_state[name]
            if averaged.dtype.is_floating_point:
                averaged.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
            else:
                averaged.copy_(current)


def _baseline_checkpoint_bundle(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    completed_steps: int,
    run_id: str,
    config_identity: str,
    seed: int,
    train_fingerprint: str,
    dev_fingerprint: str,
    git_sha: str,
    config: RealBaselineTrainConfig,
    ema_state: dict[str, torch.Tensor],
    partial_epoch_steps: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "completed_epochs": completed_epochs,
        "completed_steps": completed_steps,
        "partial_epoch_steps": partial_epoch_steps,
        "target_epochs": config.epochs,
        "config_identity": config_identity,
        "seed": seed,
        "train_dataset_fingerprint": train_fingerprint,
        "dev_dataset_fingerprint": dev_fingerprint,
        "git_sha_at_save": git_sha,
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_revision": config.checkpoint_revision,
        "precision": config.precision,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema_state,
    }


def _read_baseline_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"baseline checkpoint does not exist: {path}")
    bundle = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != BASELINE_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported baseline checkpoint; expected schema "
            f"{BASELINE_CHECKPOINT_SCHEMA_VERSION!r}"
        )
    return bundle


def load_real_baseline_model_checkpoint(
    path: Path, *, model: Any, device: torch.device
) -> RealBaselineCheckpointMetadata:
    """Load a baseline model state and return its validated portable metadata."""
    bundle = _read_baseline_checkpoint(path, device)
    metadata = RealBaselineCheckpointMetadata.model_validate(
        {key: bundle[key] for key in RealBaselineCheckpointMetadata.model_fields if key in bundle}
    )
    model.load_state_dict(bundle["model_state_dict"])
    return metadata


def load_real_baseline_checkpoint(
    path: Path,
    *,
    device: torch.device,
    model_factory: ModelFactory = build_pretrained_model,
) -> tuple[Any, Any, RealBaselineCheckpointMetadata]:
    """Construct the recorded model and load one completed baseline checkpoint."""
    bundle = _read_baseline_checkpoint(path, device)
    metadata = RealBaselineCheckpointMetadata.model_validate(
        {key: bundle[key] for key in RealBaselineCheckpointMetadata.model_fields if key in bundle}
    )
    model, processor = model_factory(metadata.checkpoint_id, metadata.checkpoint_revision)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    return model, processor, metadata


def _load_baseline_resume_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config: RealBaselineTrainConfig,
    config_identity: str,
    seed: int,
    train_fingerprint: str,
    dev_fingerprint: str,
    git_sha: str,
    device: torch.device,
) -> tuple[int, int, int, dict[str, torch.Tensor]]:
    bundle = _read_baseline_checkpoint(path, device)
    expected = {
        "run_id": config.run_id,
        "config_identity": config_identity,
        "seed": seed,
        "train_dataset_fingerprint": train_fingerprint,
        "dev_dataset_fingerprint": dev_fingerprint,
        "git_sha_at_save": git_sha,
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_revision": config.checkpoint_revision,
        "precision": config.precision,
    }
    mismatches = [name for name, value in expected.items() if bundle.get(name) != value]
    if mismatches:
        raise ValueError(
            "baseline resume checkpoint identity mismatch for: " + ", ".join(sorted(mismatches))
        )
    completed_epochs = bundle.get("completed_epochs")
    completed_steps = bundle.get("completed_steps")
    partial_epoch_steps = bundle.get("partial_epoch_steps", 0)
    if not isinstance(completed_epochs, int) or completed_epochs < 0:
        raise ValueError("baseline checkpoint has invalid completed_epochs")
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise ValueError("baseline checkpoint has invalid completed_steps")
    if not isinstance(partial_epoch_steps, int) or partial_epoch_steps < 0:
        raise ValueError("baseline checkpoint has invalid partial_epoch_steps")
    model.load_state_dict(bundle["model_state_dict"])
    optimizer.load_state_dict(bundle["optimizer_state_dict"])
    # Absent on checkpoints written before EMA existed: fall back to a fresh
    # copy of the just-loaded raw weights rather than failing resume.
    ema_state = bundle.get("ema_state_dict") or _create_ema_state(model)
    return completed_epochs, completed_steps, partial_epoch_steps, ema_state


def _baseline_data_loader(
    dataset: Any,
    *,
    processor: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=_make_collate_fn(processor),
    )


def run_real_baseline_training(
    config: RealBaselineTrainConfig,
    settings: Settings,
    model_factory: ModelFactory = build_pretrained_model,
) -> RealBaselineTrainResult:
    """Train the fixed R100 baseline on complete train/dev manifests."""
    if settings.artifact_root is None:
        raise ValueError("settings.artifact_root is required for training")

    train_manifest = _load_manifest(config.train_manifest_path)
    dev_manifest = _load_manifest(config.dev_manifest_path)
    if train_manifest.role != "real_train" or dev_manifest.role != "real_dev":
        raise ValueError("real baseline requires real_train and real_dev manifests")
    train_dataset = load_training_dataset(train_manifest, config.flir_train_root)
    dev_dataset = load_training_dataset(dev_manifest, config.flir_dev_root)

    seed = derive_subseed(settings.master_seed, "training", "real-baseline")
    torch.manual_seed(seed)
    hardware = collect_training_hardware_report(precision=config.precision)
    device = torch.device(config.device)
    if device.type == "cuda" and (not hardware.cuda_available or hardware.cuda_compatible is False):
        raise RuntimeError(
            "CUDA device requested but PyTorch CUDA is unavailable or incompatible; "
            "run `spectratwin env doctor --profile colab` before remote compute"
        )
    if config.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 precision requires a CUDA device with bf16 support")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, processor = model_factory(config.checkpoint_id, config.checkpoint_revision)
    model.to(device)
    optimizer = _build_baseline_optimizer(model, config)
    git_sha = collect_environment_report().git_sha
    if git_sha is None:
        raise RuntimeError("real baseline training requires a Git checkout with an exact code SHA")
    config_identity = _baseline_config_identity(config)

    resumed_from_epoch = 0
    completed_steps = 0
    resumed_partial_epoch_steps = 0
    ema_state = _create_ema_state(model)
    if config.resume_from_checkpoint is not None:
        resumed_from_epoch, completed_steps, resumed_partial_epoch_steps, ema_state = (
            _load_baseline_resume_checkpoint(
                config.resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                config_identity=config_identity,
                seed=seed,
                train_fingerprint=train_manifest.fingerprint,
                dev_fingerprint=dev_manifest.fingerprint,
                git_sha=git_sha,
                device=device,
            )
        )
    if resumed_from_epoch >= config.epochs:
        raise ValueError("baseline checkpoint already reached the configured target epochs")

    invocation_target = config.epochs
    if config.max_epochs_this_invocation is not None:
        invocation_target = min(
            config.epochs, resumed_from_epoch + config.max_epochs_this_invocation
        )

    run_dir = settings.artifact_root / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / "resolved_config.json"
    resolved_payload = {
        "schema_version": "spectratwin-real-baseline-config-v1",
        "config_identity": config_identity,
        "config": config.model_dump(mode="json"),
    }
    resolved_config_path.write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint_path = settings.artifact_root / "checkpoints" / config.run_id / "model.pt"

    mlflow.set_tracking_uri(f"file:{settings.artifact_root / 'mlruns'}")
    mlflow.set_experiment("real-only-baseline")
    persisted_checkpoints: list[PersistedCheckpointEvidence] = []
    final_train_loss = float("nan")
    final_dev_loss = float("nan")
    pin_memory = device.type == "cuda"

    with mlflow.start_run(run_name=config.run_id) as active_run:
        config_params = {
            key: "" if value is None else value
            for key, value in config.model_dump(mode="json").items()
        }
        mlflow.log_params(
            {
                **config_params,
                "seed": seed,
                "git_sha": git_sha,
                "train_dataset_fingerprint": train_manifest.fingerprint,
                "dev_dataset_fingerprint": dev_manifest.fingerprint,
                "config_identity": config_identity,
                "logical_run_id": config.run_id,
                "resumed_from_epoch": resumed_from_epoch,
                **hardware.model_dump(),
            }
        )
        mlflow.log_artifact(str(resolved_config_path), artifact_path="identity")

        for epoch_index in range(resumed_from_epoch, invocation_target):
            epoch_seed = derive_subseed(seed, "train-epoch", str(epoch_index))
            train_loader = _baseline_data_loader(
                wrap_with_train_augmentation(train_dataset, epoch_seed),
                processor=processor,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=True,
                seed=epoch_seed,
                pin_memory=pin_memory,
            )
            skip_batches = resumed_partial_epoch_steps if epoch_index == resumed_from_epoch else 0
            steps_this_epoch = skip_batches
            batch_iter: Any = (
                itertools.islice(train_loader, skip_batches, None) if skip_batches else train_loader
            )
            model.train()
            epoch_train_losses: list[float] = []
            for batch in batch_iter:
                _apply_linear_warmup(
                    optimizer,
                    completed_steps=completed_steps,
                    warmup_steps=config.warmup_steps,
                )
                batch = _move_batch_to_device(batch, device)
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.precision == "bf16",
                ):
                    outputs = model(**batch)
                    loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.gradient_clip_norm
                )
                optimizer.step()
                _update_ema_state(
                    ema_state, model, step=completed_steps, warmup_steps=config.warmup_steps
                )
                final_train_loss = float(loss.detach().cpu())
                if not math.isfinite(final_train_loss):
                    raise RuntimeError("non-finite training loss; refusing to write a checkpoint")
                epoch_train_losses.append(final_train_loss)
                mlflow.log_metric("train_loss", final_train_loss, step=completed_steps)
                mlflow.log_metric(
                    "learning_rate", float(optimizer.param_groups[0]["lr"]), step=completed_steps
                )
                completed_steps += 1
                steps_this_epoch += 1

                if (
                    config.checkpoint_interval_steps is not None
                    and steps_this_epoch % config.checkpoint_interval_steps == 0
                ):
                    _save_checkpoint_atomically(
                        checkpoint_path,
                        _baseline_checkpoint_bundle(
                            model=model,
                            optimizer=optimizer,
                            completed_epochs=epoch_index,
                            completed_steps=completed_steps,
                            run_id=config.run_id,
                            config_identity=config_identity,
                            seed=seed,
                            train_fingerprint=train_manifest.fingerprint,
                            dev_fingerprint=dev_manifest.fingerprint,
                            git_sha=git_sha,
                            config=config,
                            ema_state=ema_state,
                            partial_epoch_steps=steps_this_epoch,
                        ),
                    )
                    if config.persistent_checkpoint_dir is not None:
                        # Each interval hit gets its own destination: persist_file
                        # refuses to overwrite an already-completed marker with
                        # different content, and a resumed invocation can reach
                        # the same step count as the run it resumed from. The
                        # physical MLflow run id makes the name unique per
                        # invocation even when the step count repeats.
                        mid_epoch_path = (
                            config.persistent_checkpoint_dir
                            / f"model-epoch-{epoch_index:03d}-step-{completed_steps:06d}"
                            f"-{active_run.info.run_id[:8]}.pt"
                        )
                        transfer = persist_file(checkpoint_path, mid_epoch_path)
                        if transfer.completion_marker_name is None:
                            raise RuntimeError(
                                "persisted mid-epoch checkpoint is missing its completion marker"
                            )

            dev_loader = _baseline_data_loader(
                dev_dataset,
                processor=processor,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=False,
                seed=derive_subseed(seed, "dev", str(epoch_index)),
                pin_memory=pin_memory,
            )
            model.eval()
            dev_losses: list[float] = []
            with torch.no_grad():
                for batch in dev_loader:
                    batch = _move_batch_to_device(batch, device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=config.precision == "bf16",
                    ):
                        outputs = model(**batch)
                    dev_losses.append(float(outputs.loss.detach().cpu()))
            final_dev_loss = sum(dev_losses) / len(dev_losses)
            if not math.isfinite(final_dev_loss):
                raise RuntimeError("non-finite development loss; refusing to write a checkpoint")
            completed_epochs = epoch_index + 1
            mlflow.log_metric(
                "epoch_train_loss",
                sum(epoch_train_losses) / len(epoch_train_losses),
                step=completed_epochs,
            )
            mlflow.log_metric("dev_loss", final_dev_loss, step=completed_epochs)

            _save_checkpoint_atomically(
                checkpoint_path,
                _baseline_checkpoint_bundle(
                    model=model,
                    optimizer=optimizer,
                    completed_epochs=completed_epochs,
                    completed_steps=completed_steps,
                    run_id=config.run_id,
                    config_identity=config_identity,
                    seed=seed,
                    train_fingerprint=train_manifest.fingerprint,
                    dev_fingerprint=dev_manifest.fingerprint,
                    git_sha=git_sha,
                    config=config,
                    ema_state=ema_state,
                ),
            )
            should_persist = (
                completed_epochs % config.checkpoint_interval_epochs == 0
                or completed_epochs == invocation_target
                or completed_epochs == config.epochs
            )
            if config.persistent_checkpoint_dir is not None and should_persist:
                persistent_path = (
                    config.persistent_checkpoint_dir / f"model-epoch-{completed_epochs:03d}.pt"
                )
                transfer = persist_file(checkpoint_path, persistent_path)
                if transfer.completion_marker_name is None:
                    raise RuntimeError("persisted checkpoint is missing its completion marker")
                persisted_checkpoints.append(
                    PersistedCheckpointEvidence(
                        epoch=completed_epochs,
                        artifact_name=transfer.artifact_name,
                        sha256=transfer.sha256,
                        size_bytes=transfer.size_bytes,
                        completion_marker_name=transfer.completion_marker_name,
                    )
                )

        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoint")
        mlflow.log_params(
            {
                "completed_epochs": invocation_target,
                "completed_steps": completed_steps,
                "logical_run_complete": invocation_target == config.epochs,
            }
        )
        mlflow_run_id = active_run.info.run_id

    return RealBaselineTrainResult(
        run_id=config.run_id,
        mlflow_run_id=mlflow_run_id,
        checkpoint_path=checkpoint_path,
        resolved_config_path=resolved_config_path,
        config_identity=config_identity,
        train_dataset_fingerprint=train_manifest.fingerprint,
        dev_dataset_fingerprint=dev_manifest.fingerprint,
        git_sha=git_sha,
        seed=seed,
        final_train_loss=final_train_loss,
        final_dev_loss=final_dev_loss,
        train_sample_count=len(train_dataset),
        dev_sample_count=len(dev_dataset),
        resumed_from_epoch=resumed_from_epoch,
        completed_epochs=invocation_target,
        completed_steps=completed_steps,
        target_epochs=config.epochs,
        complete=invocation_target == config.epochs,
        persisted_checkpoints=persisted_checkpoints,
    )
