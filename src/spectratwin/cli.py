"""Minimal CLI bootstrap: ``python -m spectratwin`` / ``spectratwin``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spectratwin import __version__
from spectratwin.config.settings import ExecutionProfile, load_settings
from spectratwin.contracts.environment_report import (
    collect_environment_report,
    evaluate_environment_capability,
)


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _cmd_env_doctor(args: argparse.Namespace) -> int:
    profile = ExecutionProfile(args.profile)
    settings = None
    settings_error = None
    if profile != ExecutionProfile.CPU_DEV:
        try:
            settings = load_settings()
        except (TypeError, ValueError) as exc:
            settings_error = str(exc)

    report = evaluate_environment_capability(
        collect_environment_report(),
        profile,
        settings=settings,
        settings_error=settings_error,
        required_paths=tuple(args.require_path),
    )
    print(report.model_dump_json(indent=2))
    return 0 if report.ready else 1


def _cmd_remote_stage(args: argparse.Namespace) -> int:
    from spectratwin.remote.staging import stage_file

    result = stage_file(Path(args.source), Path(args.destination), args.sha256)
    print(result.model_dump_json(indent=2))
    return 0


def _cmd_remote_persist(args: argparse.Namespace) -> int:
    from spectratwin.remote.staging import persist_file

    result = persist_file(Path(args.source), Path(args.destination))
    print(result.model_dump_json(indent=2))
    return 0


def _cmd_train_real_smoke(args: argparse.Namespace) -> int:
    # Imported lazily: the ``train`` extra (torch/transformers/mlflow) is
    # optional and must not be required for cpu-dev commands like env doctor.
    from spectratwin.training.config import RealSmokeTrainConfig
    from spectratwin.training.run import run_real_smoke_training

    settings = load_settings()
    config = RealSmokeTrainConfig(
        train_manifest_path=Path(args.train_manifest),
        dev_manifest_path=Path(args.dev_manifest),
        flir_train_root=Path(args.flir_train_root),
        flir_dev_root=Path(args.flir_dev_root),
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        run_id=args.run_id,
        resume_from_checkpoint=(
            Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
    )
    result = run_real_smoke_training(config, settings)
    print(result.model_dump_json(indent=2))
    return 0


def _cmd_train_real_baseline(args: argparse.Namespace) -> int:
    from spectratwin.training.config import RealBaselineTrainConfig
    from spectratwin.training.run import run_real_baseline_training

    settings = load_settings()
    config = RealBaselineTrainConfig(
        train_manifest_path=Path(args.train_manifest),
        dev_manifest_path=Path(args.dev_manifest),
        flir_train_root=Path(args.flir_train_root),
        flir_dev_root=Path(args.flir_dev_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_interval_epochs=args.checkpoint_interval_epochs,
        precision=args.precision,
        device=args.device,
        num_workers=args.num_workers,
        run_id=args.run_id,
        resume_from_checkpoint=(
            Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        persistent_checkpoint_dir=(
            Path(args.persistent_checkpoint_dir) if args.persistent_checkpoint_dir else None
        ),
        max_epochs_this_invocation=args.max_epochs_this_invocation,
    )
    result = run_real_baseline_training(config, settings)
    print(result.model_dump_json(indent=2))
    return 0


def _cmd_evaluate_real_benchmark(args: argparse.Namespace) -> int:
    import torch

    from spectratwin.training.evaluate import evaluate_real_baseline_checkpoint

    settings = load_settings()
    if settings.artifact_root is None:
        raise ValueError("SPECTRATWIN_ARTIFACT_ROOT is required for evaluation")
    report = evaluate_real_baseline_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        benchmark_manifest_path=Path(args.benchmark_manifest),
        flir_benchmark_root=Path(args.flir_benchmark_root),
        device=torch.device(args.device),
        artifact_root=settings.artifact_root,
        expected_run_id=args.run_id,
        score_threshold=args.score_threshold,
    )
    print(report.model_dump_json(indent=2))
    return 0


def _cmd_report_training_loss(args: argparse.Namespace) -> int:
    from spectratwin.training.report import build_training_loss_report

    written = build_training_loss_report(
        tracking_uri=f"file:{args.mlruns_dir}",
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


def _cmd_report_confusion_matrix(args: argparse.Namespace) -> int:
    from spectratwin.training.report import build_confusion_matrix_report

    written = build_confusion_matrix_report(
        predictions_artifact_path=Path(args.predictions),
        output_dir=Path(args.output_dir),
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectratwin")
    parser.add_argument("--version", action="store_true", help="print version and exit")

    subparsers = parser.add_subparsers(dest="command")
    env_parser = subparsers.add_parser("env", help="environment utilities")
    env_subparsers = env_parser.add_subparsers(dest="env_command", required=True)
    doctor_parser = env_subparsers.add_parser(
        "doctor", help="report facts and validate an execution profile"
    )
    doctor_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in ExecutionProfile],
        default=ExecutionProfile.CPU_DEV.value,
    )
    doctor_parser.add_argument(
        "--require-path",
        action="append",
        default=[],
        type=_labeled_path,
        metavar="LABEL=PATH",
        help="verify a required input without printing its path (repeatable)",
    )
    doctor_parser.set_defaults(func=_cmd_env_doctor)

    remote_parser = subparsers.add_parser("remote", help="remote staging utilities")
    remote_subparsers = remote_parser.add_subparsers(dest="remote_command", required=True)
    stage_parser = remote_subparsers.add_parser(
        "stage", help="checksum-verify and atomically stage one input file"
    )
    stage_parser.add_argument("--source", required=True)
    stage_parser.add_argument("--destination", required=True)
    stage_parser.add_argument("--sha256", required=True)
    stage_parser.set_defaults(func=_cmd_remote_stage)

    persist_parser = remote_subparsers.add_parser(
        "persist", help="atomically persist one output and write a verified completion marker"
    )
    persist_parser.add_argument("--source", required=True)
    persist_parser.add_argument("--destination", required=True)
    persist_parser.set_defaults(func=_cmd_remote_persist)

    train_parser = subparsers.add_parser("train", help="training utilities")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    real_smoke_parser = train_subparsers.add_parser(
        "real-smoke", help="run one real-only RT-DETRv2 smoke train/eval pass"
    )
    real_smoke_parser.add_argument("--train-manifest", required=True)
    real_smoke_parser.add_argument("--dev-manifest", required=True)
    real_smoke_parser.add_argument("--flir-train-root", required=True)
    real_smoke_parser.add_argument("--flir-dev-root", required=True)
    real_smoke_parser.add_argument("--max-train-samples", type=int, default=8)
    real_smoke_parser.add_argument("--max-dev-samples", type=int, default=4)
    real_smoke_parser.add_argument("--max-steps", type=int, default=2)
    real_smoke_parser.add_argument("--batch-size", type=int, default=2)
    real_smoke_parser.add_argument("--device", default="cpu")
    real_smoke_parser.add_argument("--run-id")
    real_smoke_parser.add_argument("--resume-from-checkpoint")
    real_smoke_parser.set_defaults(func=_cmd_train_real_smoke)

    real_baseline_parser = train_subparsers.add_parser(
        "real-baseline", help="train the fixed R100 baseline on complete train/dev manifests"
    )
    real_baseline_parser.add_argument("--train-manifest", required=True)
    real_baseline_parser.add_argument("--dev-manifest", required=True)
    real_baseline_parser.add_argument("--flir-train-root", required=True)
    real_baseline_parser.add_argument("--flir-dev-root", required=True)
    real_baseline_parser.add_argument("--epochs", type=int, required=True)
    real_baseline_parser.add_argument("--batch-size", type=int, default=16)
    real_baseline_parser.add_argument("--learning-rate", type=float, default=1e-4)
    real_baseline_parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    real_baseline_parser.add_argument("--weight-decay", type=float, default=1e-4)
    real_baseline_parser.add_argument("--warmup-steps", type=int, default=2_000)
    real_baseline_parser.add_argument("--gradient-clip-norm", type=float, default=0.1)
    real_baseline_parser.add_argument("--checkpoint-interval-epochs", type=int, default=5)
    real_baseline_parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    real_baseline_parser.add_argument("--device", default="cuda")
    real_baseline_parser.add_argument("--num-workers", type=int, default=4)
    real_baseline_parser.add_argument("--run-id", required=True)
    real_baseline_parser.add_argument("--resume-from-checkpoint")
    real_baseline_parser.add_argument("--persistent-checkpoint-dir")
    real_baseline_parser.add_argument("--max-epochs-this-invocation", type=int)
    real_baseline_parser.set_defaults(func=_cmd_train_real_baseline)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluation utilities")
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command", required=True)
    real_benchmark_parser = evaluate_subparsers.add_parser(
        "real-benchmark", help="evaluate one completed R100 checkpoint exactly once"
    )
    real_benchmark_parser.add_argument("--checkpoint", required=True)
    real_benchmark_parser.add_argument("--benchmark-manifest", required=True)
    real_benchmark_parser.add_argument("--flir-benchmark-root", required=True)
    real_benchmark_parser.add_argument("--run-id", required=True)
    real_benchmark_parser.add_argument("--device", default="cuda")
    real_benchmark_parser.add_argument("--score-threshold", type=float, default=0.0)
    real_benchmark_parser.set_defaults(func=_cmd_evaluate_real_benchmark)

    report_parser = subparsers.add_parser("report", help="diagnostic plots and tables")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)

    training_loss_parser = report_subparsers.add_parser(
        "training-loss", help="loss-curve plot and metric tables for one MLflow run"
    )
    training_loss_parser.add_argument("--mlruns-dir", required=True)
    training_loss_parser.add_argument("--experiment-name", default="real-only-baseline")
    training_loss_parser.add_argument("--run-id", required=True)
    training_loss_parser.add_argument("--output-dir", required=True)
    training_loss_parser.set_defaults(func=_cmd_report_training_loss)

    confusion_matrix_parser = report_subparsers.add_parser(
        "confusion-matrix", help="confusion-matrix heatmap from a saved predictions artifact"
    )
    confusion_matrix_parser.add_argument("--predictions", required=True)
    confusion_matrix_parser.add_argument("--output-dir", required=True)
    confusion_matrix_parser.add_argument("--iou-threshold", type=float, default=0.5)
    confusion_matrix_parser.add_argument("--score-threshold", type=float, default=0.5)
    confusion_matrix_parser.set_defaults(func=_cmd_report_confusion_matrix)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
