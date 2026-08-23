import json

from spectratwin import __version__
from spectratwin.cli import build_parser, main


def test_version_flag_prints_version(capsys):
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == __version__


def test_env_doctor_prints_json_report(capsys):
    exit_code = main(["env", "doctor"])
    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["ready"] is True
    assert report["profile"] == "cpu-dev"
    assert "os_name" in report["environment"]
    assert "python_version" in report["environment"]


def test_env_doctor_rejects_missing_required_path(capsys, tmp_path):
    missing = tmp_path / "missing.json"
    exit_code = main(["env", "doctor", "--require-path", f"manifest={missing}"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 1
    assert report["ready"] is False
    required_check = next(
        check for check in report["checks"] if check["name"] == "required_path:manifest"
    )
    assert required_check["status"] == "fail"
    assert str(missing) not in captured.out


def test_no_args_prints_help_and_fails():
    parser = build_parser()
    assert parser.prog == "spectratwin"


def test_train_real_smoke_parses_required_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "real-smoke",
            "--train-manifest",
            "train.json",
            "--dev-manifest",
            "dev.json",
            "--flir-train-root",
            "/data/train",
            "--flir-dev-root",
            "/data/dev",
        ]
    )
    assert args.train_manifest == "train.json"
    assert args.max_train_samples == 8
    assert args.device == "cpu"
    assert callable(args.func)


def test_remote_stage_parses_required_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "remote",
            "stage",
            "--source",
            "/persistent/data.tar",
            "--destination",
            "/content/data.tar",
            "--sha256",
            "a" * 64,
        ]
    )

    assert args.sha256 == "a" * 64
    assert callable(args.func)
