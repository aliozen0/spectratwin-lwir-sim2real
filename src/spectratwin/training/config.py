"""Resolved config for the real-only smoke training run.

Every field here is persisted into MLflow verbatim so a run is reconstructable
from run metadata alone.
"""

from __future__ import annotations

from pathlib import Path

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
