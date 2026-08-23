"""Training-runtime hardware/software facts.

Complements :mod:`spectratwin.contracts.environment_report`: that module
reports OS/Git/GPU-via-``nvidia-smi`` facts without importing torch. This
module additionally reports what *torch itself* detects, since a host GPU
being present does not guarantee the installed torch build can use it.
"""

from __future__ import annotations

import warnings

import torch
import transformers
from pydantic import BaseModel


class TrainingHardwareReport(BaseModel):
    torch_version: str
    transformers_version: str
    cuda_available: bool
    device_name: str | None
    cuda_version: str | None
    device_vram_mb: int | None
    device_capability: str | None
    cuda_compatible: bool | None
    precision: str


def collect_training_hardware_report(precision: str = "fp32") -> TrainingHardwareReport:
    cuda_available = torch.cuda.is_available()
    device_name = None
    device_vram_mb = None
    device_capability = None
    cuda_compatible = False if not cuda_available else None
    if cuda_available:
        # Compatibility is emitted as structured run metadata. Suppress the
        # equivalent human warnings so CLI JSON/log output stays parseable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                properties = torch.cuda.get_device_properties(0)
                major, minor = torch.cuda.get_device_capability(0)
            except RuntimeError:
                cuda_compatible = False
            else:
                device_name = properties.name
                device_vram_mb = round(properties.total_memory / (1024**2))
                device_capability = f"sm_{major}{minor}"
                try:
                    probe = torch.ones(1, device="cuda")
                    probe.add_(1)
                    torch.cuda.synchronize()
                except RuntimeError:
                    cuda_compatible = False
                else:
                    cuda_compatible = True
    return TrainingHardwareReport(
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        cuda_available=cuda_available,
        device_name=device_name,
        cuda_version=torch.version.cuda,
        device_vram_mb=device_vram_mb,
        device_capability=device_capability,
        cuda_compatible=cuda_compatible,
        precision=precision,
    )
