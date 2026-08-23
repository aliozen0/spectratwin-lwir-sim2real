from pathlib import Path

import pytest
from pydantic import ValidationError

from spectratwin.config.settings import ExecutionProfile, Settings, load_settings


def test_cpu_dev_needs_no_storage_roots():
    settings = Settings(execution_profile=ExecutionProfile.CPU_DEV, master_seed=0)
    assert settings.data_root is None


def test_negative_seed_is_rejected():
    with pytest.raises(ValidationError):
        Settings(execution_profile=ExecutionProfile.CPU_DEV, master_seed=-1)


def test_wsl_profile_requires_all_roots():
    with pytest.raises(ValidationError):
        Settings(execution_profile=ExecutionProfile.WSL, master_seed=0, data_root=Path("/x"))


def test_wsl_profile_accepts_all_roots():
    settings = Settings(
        execution_profile=ExecutionProfile.WSL,
        master_seed=0,
        data_root=Path("/data"),
        cache_root=Path("/cache"),
        artifact_root=Path("/artifacts"),
    )
    assert settings.execution_profile == ExecutionProfile.WSL


def test_load_settings_requires_master_seed_env():
    with pytest.raises(ValueError):
        load_settings(env={})


def test_load_settings_reads_env(monkeypatch):
    env = {"SPECTRATWIN_MASTER_SEED": "123"}
    settings = load_settings(env=env)
    assert settings.master_seed == 123
    assert settings.execution_profile == ExecutionProfile.CPU_DEV
