from spectratwin.training.hardware import collect_training_hardware_report


def test_training_hardware_report_records_cuda_compatibility():
    report = collect_training_hardware_report()

    assert report.torch_version
    assert report.transformers_version
    assert report.cuda_compatible in {True, False, None}
    if report.device_capability is not None:
        assert report.device_capability.startswith("sm_")
