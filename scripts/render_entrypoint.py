# Entry point executed by ``blenderproc run`` (SPEC-007).
#
# Orchestration only. docs/REPRODUCIBILITY.md ("Notebook reproducibility")
# requires that any algorithmic code producing a reported result lives in the
# Git-tracked package, so this file must never grow model or physics logic.
#
# Usage:
#
#     uv run --extra render blenderproc run scripts/render_entrypoint.py -- \
#       --pytest tests/render/test_radiometric_roundtrip.py
#
# The header above is comments rather than a docstring, and there is no
# ``from __future__ import annotations``, because BlenderProc refuses to run a
# script whose first non-comment, non-blank line is not a blenderproc import
# (SetupUtility.check_if_setup_utilities_are_at_the_top). A docstring or a
# __future__ import would both take that slot. Blender 4.2.1 embeds Python
# 3.11, which supports the annotation syntax used here natively.
#
# ruff: noqa: I001
# Import order is fixed by BlenderProc, not by isort: the blenderproc import
# must precede the standard library, so the usual grouping cannot apply.

import blenderproc  # noqa: F401  # side effect: puts Blender's site-packages on sys.path

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpectraTwin render entry point")
    parser.add_argument(
        "--pytest",
        action="append",
        default=[],
        metavar="PATH",
        help="run this pytest target inside Blender; may be repeated",
    )
    arguments = parser.parse_args(argv if argv is not None else sys.argv[1:])

    repository_root = Path(__file__).resolve().parent.parent
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    if not arguments.pytest:
        parser.error("nothing to do: pass at least one --pytest target")

    import pytest

    return pytest.main(["-q", "-m", "renderer", "-p", "no:cacheprovider", *arguments.pytest])


if __name__ == "__main__":
    raise SystemExit(main())
