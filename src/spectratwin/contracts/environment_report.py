"""Environment-doctor contract: what a fresh checkout reports about itself.

GPU absence is not an error: a machine with no GPU runs the ``cpu-dev``
profile. GPU fields stay ``None`` in that case.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from spectratwin.config.settings import ExecutionProfile, Settings


class EnvironmentReport(BaseModel):
    os_name: str
    os_release: str
    python_version: str
    git_sha: str | None
    gpu_name: str | None
    gpu_vram_mb: int | None
    gpu_driver_version: str | None = None
    cuda_driver_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_available: bool | None = None
    torch_device_capability: str | None = None
    torch_cuda_compatible: bool | None = None
    free_disk_gb: float
    total_memory_gb: float | None = None


class CapabilityCheck(BaseModel):
    """One machine-readable preflight check without sensitive path values."""

    name: str
    status: Literal["pass", "fail", "info"]
    required: bool
    detail: str


class EnvironmentCapabilityReport(BaseModel):
    """Profile-aware readiness result used before local or remote compute."""

    profile: ExecutionProfile
    ready: bool
    environment: EnvironmentReport
    checks: tuple[CapabilityCheck, ...]


def _git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _gpu_facts() -> tuple[str | None, int | None, str | None, str | None]:
    """Detect GPU model/VRAM via ``nvidia-smi`` if present. Never assumed."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None, None, None, None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None, None, None
    if result.returncode != 0 or not result.stdout.strip():
        return None, None, None, None
    name, vram, driver = result.stdout.strip().splitlines()[0].split(",")

    cuda_version = None
    try:
        summary = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=5)
        match = re.search(r"CUDA Version:\s*([0-9.]+)", summary.stdout)
        if summary.returncode == 0 and match is not None:
            cuda_version = match.group(1)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return name.strip(), int(vram.strip()), driver.strip(), cuda_version


def _total_memory_gb() -> float | None:
    """Return physical memory without adding a platform-specific dependency."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return round((page_size * page_count) / (1024**3), 2)


def _torch_facts() -> tuple[str | None, str | None, bool | None, str | None, bool | None]:
    """Inspect the optional training runtime without making torch a core dependency."""
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError):
        return None, None, None, None, None

    torch_version = str(torch.__version__)
    cuda_version = torch.version.cuda
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return torch_version, cuda_version, False, None, False

    # Torch emits three human warnings for an unsupported architecture while
    # probing it. The doctor replaces those with the structured compatibility
    # field below so JSON-producing commands stay quiet and machine-readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            major, minor = torch.cuda.get_device_capability(0)
        except RuntimeError:
            return torch_version, cuda_version, True, None, False
        capability = f"sm_{major}{minor}"
        try:
            probe = torch.ones(1, device="cuda")
            probe.add_(1)
            torch.cuda.synchronize()
        except RuntimeError:
            compatible = False
        else:
            compatible = True
    return torch_version, cuda_version, True, capability, compatible


def collect_environment_report(repo_root: Path | None = None) -> EnvironmentReport:
    root = repo_root or Path.cwd()
    gpu_name, gpu_vram_mb, gpu_driver_version, cuda_driver_version = _gpu_facts()
    (
        torch_version,
        torch_cuda_version,
        torch_cuda_available,
        torch_device_capability,
        torch_cuda_compatible,
    ) = _torch_facts()
    free_bytes = shutil.disk_usage(root).free
    return EnvironmentReport(
        os_name=platform.system(),
        os_release=platform.release(),
        python_version=platform.python_version(),
        git_sha=_git_sha(root),
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_driver_version=gpu_driver_version,
        cuda_driver_version=cuda_driver_version,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torch_cuda_available=torch_cuda_available,
        torch_device_capability=torch_device_capability,
        torch_cuda_compatible=torch_cuda_compatible,
        free_disk_gb=round(free_bytes / (1024**3), 2),
        total_memory_gb=_total_memory_gb(),
    )


def evaluate_environment_capability(
    report: EnvironmentReport,
    profile: ExecutionProfile,
    *,
    settings: Settings | None = None,
    settings_error: str | None = None,
    required_paths: tuple[tuple[str, Path], ...] = (),
) -> EnvironmentCapabilityReport:
    """Evaluate profile readiness using an already collected fact report.

    Path values are intentionally omitted from output. Callers provide stable
    labels such as ``train_manifest`` so proprietary/personal paths do not leak
    into logs while presence is still verified.
    """
    checks = [
        CapabilityCheck(
            name="git_checkout",
            status="pass" if report.git_sha else "fail",
            required=True,
            detail="exact Git SHA detected" if report.git_sha else "Git SHA unavailable",
        ),
        CapabilityCheck(
            name="disk",
            status="info",
            required=False,
            detail=f"{report.free_disk_gb:.2f} GiB free",
        ),
        CapabilityCheck(
            name="memory",
            status="info",
            required=False,
            detail=(
                f"{report.total_memory_gb:.2f} GiB total"
                if report.total_memory_gb is not None
                else "total memory unavailable"
            ),
        ),
    ]

    if profile == ExecutionProfile.COLAB:
        gpu_ready = report.gpu_name is not None and report.gpu_vram_mb is not None
        checks.append(
            CapabilityCheck(
                name="gpu",
                status="pass" if gpu_ready else "fail",
                required=True,
                detail=(
                    f"{report.gpu_name}, {report.gpu_vram_mb} MiB"
                    if gpu_ready
                    else "GPU/VRAM not detected via nvidia-smi"
                ),
            )
        )
        torch_cuda_ready = (
            report.torch_version is not None
            and report.torch_cuda_available is True
            and report.torch_cuda_compatible is not False
        )
        checks.append(
            CapabilityCheck(
                name="torch_cuda",
                status="pass" if torch_cuda_ready else "fail",
                required=True,
                detail=(
                    f"torch {report.torch_version}, CUDA {report.torch_cuda_version}, "
                    f"capability {report.torch_device_capability}"
                    if torch_cuda_ready
                    else "PyTorch CUDA runtime is unavailable or incompatible with this GPU"
                ),
            )
        )

    if profile != ExecutionProfile.CPU_DEV:
        settings_ready = settings is not None and settings.execution_profile == profile
        checks.append(
            CapabilityCheck(
                name="settings",
                status="pass" if settings_ready else "fail",
                required=True,
                detail=(
                    f"execution profile {profile.value!r} resolved"
                    if settings_ready
                    else (settings_error or f"settings profile must be {profile.value!r}")
                ),
            )
        )
        for field_name in ("data_root", "cache_root", "artifact_root"):
            root = getattr(settings, field_name) if settings is not None else None
            root_ready = root is not None and root.exists() and root.is_dir()
            checks.append(
                CapabilityCheck(
                    name=field_name,
                    status="pass" if root_ready else "fail",
                    required=True,
                    detail="configured directory exists" if root_ready else "directory unavailable",
                )
            )

    for label, path in required_paths:
        exists = path.exists()
        checks.append(
            CapabilityCheck(
                name=f"required_path:{label}",
                status="pass" if exists else "fail",
                required=True,
                detail="present" if exists else "missing",
            )
        )

    ready = all(check.status != "fail" for check in checks if check.required)
    return EnvironmentCapabilityReport(
        profile=profile,
        ready=ready,
        environment=report,
        checks=tuple(checks),
    )
