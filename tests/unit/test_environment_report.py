from pathlib import Path

from spectratwin.config.settings import ExecutionProfile, Settings
from spectratwin.contracts.environment_report import (
    EnvironmentReport,
    evaluate_environment_capability,
)


def _report(*, with_gpu: bool) -> EnvironmentReport:
    return EnvironmentReport(
        os_name="Linux",
        os_release="test",
        python_version="3.13",
        git_sha="a" * 40,
        gpu_name="Test GPU" if with_gpu else None,
        gpu_vram_mb=16_384 if with_gpu else None,
        gpu_driver_version="555.1" if with_gpu else None,
        cuda_driver_version="12.5" if with_gpu else None,
        torch_version="2.9.1",
        torch_cuda_version="12.8",
        torch_cuda_available=with_gpu,
        torch_device_capability="sm_80" if with_gpu else None,
        torch_cuda_compatible=with_gpu,
        free_disk_gb=50.0,
        total_memory_gb=12.0,
    )


def test_colab_capability_requires_detected_gpu(tmp_path: Path):
    settings = Settings(
        execution_profile=ExecutionProfile.COLAB,
        master_seed=0,
        data_root=tmp_path,
        cache_root=tmp_path,
        artifact_root=tmp_path,
    )

    result = evaluate_environment_capability(
        _report(with_gpu=False), ExecutionProfile.COLAB, settings=settings
    )

    assert result.ready is False
    gpu_check = next(check for check in result.checks if check.name == "gpu")
    assert gpu_check.status == "fail"


def test_colab_capability_rejects_torch_incompatible_gpu(tmp_path: Path):
    report = _report(with_gpu=True).model_copy(update={"torch_cuda_compatible": False})
    settings = Settings(
        execution_profile=ExecutionProfile.COLAB,
        master_seed=0,
        data_root=tmp_path,
        cache_root=tmp_path,
        artifact_root=tmp_path,
    )

    result = evaluate_environment_capability(report, ExecutionProfile.COLAB, settings=settings)

    assert result.ready is False
    torch_check = next(check for check in result.checks if check.name == "torch_cuda")
    assert torch_check.status == "fail"


def test_colab_capability_passes_with_gpu_settings_and_inputs(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    settings = Settings(
        execution_profile=ExecutionProfile.COLAB,
        master_seed=0,
        data_root=tmp_path,
        cache_root=tmp_path,
        artifact_root=tmp_path,
    )

    result = evaluate_environment_capability(
        _report(with_gpu=True),
        ExecutionProfile.COLAB,
        settings=settings,
        required_paths=(("manifest", manifest),),
    )

    assert result.ready is True
    assert all(check.status != "fail" for check in result.checks if check.required)


def test_capability_output_does_not_include_required_path(tmp_path: Path):
    secret_path = tmp_path / "private" / "dataset.json"

    result = evaluate_environment_capability(
        _report(with_gpu=False),
        ExecutionProfile.CPU_DEV,
        required_paths=(("dataset", secret_path),),
    )

    assert str(secret_path) not in result.model_dump_json()
