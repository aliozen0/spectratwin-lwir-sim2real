"""Typed, portable runtime settings.

Storage roots come from environment variables, never from hardcoded personal
WSL/Colab paths, so the same committed defaults work on any execution
profile (see docs/CONFIGURATION_STANDARD.md and docs/TECH_STACK.md).
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

ENV_DATA_ROOT = "SPECTRATWIN_DATA_ROOT"
ENV_CACHE_ROOT = "SPECTRATWIN_CACHE_ROOT"
ENV_ARTIFACT_ROOT = "SPECTRATWIN_ARTIFACT_ROOT"
ENV_PROFILE = "SPECTRATWIN_PROFILE"
ENV_MASTER_SEED = "SPECTRATWIN_MASTER_SEED"


class ExecutionProfile(StrEnum):
    CPU_DEV = "cpu-dev"
    WSL = "wsl"
    COLAB = "colab"


class Settings(BaseModel):
    """Fully resolved runtime configuration.

    ``cpu-dev`` needs no storage roots so unit tests and lint/typecheck
    commands never require personal paths. Any other profile MUST supply all
    three roots explicitly.
    """

    model_config = ConfigDict(frozen=True)

    execution_profile: ExecutionProfile = ExecutionProfile.CPU_DEV
    master_seed: int = Field(ge=0)
    data_root: Path | None = None
    cache_root: Path | None = None
    artifact_root: Path | None = None

    @model_validator(mode="after")
    def _require_roots_outside_cpu_dev(self) -> Settings:
        if self.execution_profile == ExecutionProfile.CPU_DEV:
            return self
        missing = [
            name
            for name, value in (
                ("data_root", self.data_root),
                ("cache_root", self.cache_root),
                ("artifact_root", self.artifact_root),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"execution_profile={self.execution_profile.value!r} requires "
                f"storage roots to be set, missing: {', '.join(missing)}"
            )
        return self


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from environment variables.

    Raises ``pydantic.ValidationError`` for an invalid/incomplete
    configuration before any expensive work starts.
    """
    source = env if env is not None else os.environ
    raw_seed = source.get(ENV_MASTER_SEED)
    if raw_seed is None:
        raise ValueError(f"{ENV_MASTER_SEED} is required")

    def _path(name: str) -> Path | None:
        value = source.get(name)
        return Path(value) if value else None

    return Settings(
        execution_profile=ExecutionProfile(source.get(ENV_PROFILE, ExecutionProfile.CPU_DEV.value)),
        master_seed=int(raw_seed),
        data_root=_path(ENV_DATA_ROOT),
        cache_root=_path(ENV_CACHE_ROOT),
        artifact_root=_path(ENV_ARTIFACT_ROOT),
    )
