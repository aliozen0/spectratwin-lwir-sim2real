"""Resolved config for the real-only smoke training run.

Every field here is persisted into MLflow verbatim so a run is reconstructable
from run metadata alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spectratwin.training.model import DEFAULT_CHECKPOINT_ID, DEFAULT_CHECKPOINT_REVISION


class RealSmokeTrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_manifest_path: Path
    dev_manifest_path: Path
    flir_train_root: Path
    flir_dev_root: Path
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    checkpoint_revision: str = DEFAULT_CHECKPOINT_REVISION
    max_train_samples: int = Field(default=8, gt=0)
    max_dev_samples: int = Field(default=4, gt=0)
    max_steps: int = Field(default=2, gt=0)
    batch_size: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    device: str = "cpu"
    run_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    resume_from_checkpoint: Path | None = None


class RealBaselineTrainConfig(BaseModel):
    """Resolved, persisted configuration for an R100 real-only baseline run.

    Unlike :class:`RealSmokeTrainConfig`, this contract always consumes the
    complete train/dev manifests. Scientific settings are explicit and become
    part of the checkpoint identity; runtime paths and worker count do not.
    """

    model_config = ConfigDict(frozen=True)

    train_manifest_path: Path
    dev_manifest_path: Path
    flir_train_root: Path
    flir_dev_root: Path
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    checkpoint_revision: str = DEFAULT_CHECKPOINT_REVISION
    epochs: int = Field(gt=0)
    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    backbone_learning_rate: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    warmup_steps: int = Field(default=2_000, ge=0)
    gradient_clip_norm: float = Field(default=0.1, gt=0)
    checkpoint_interval_epochs: int = Field(default=5, gt=0)
    precision: Literal["fp32", "bf16"] = "fp32"
    device: str = "cuda"
    num_workers: int = Field(default=4, ge=0)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    resume_from_checkpoint: Path | None = None
    persistent_checkpoint_dir: Path | None = None
    max_epochs_this_invocation: int | None = Field(default=None, gt=0)
