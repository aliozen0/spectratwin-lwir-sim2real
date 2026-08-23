"""Deterministic seed derivation.

Python's built-in ``hash()`` is randomized per-process (PYTHONHASHSEED) and
must never be used for persistent seed derivation. This module derives
subsystem/sample seeds from a master seed using SHA-256, which is stable
across processes and Python versions.
"""

from __future__ import annotations

import hashlib

import numpy as np

SEED_BITS = 32
SEED_MODULUS = 2**SEED_BITS


def derive_subseed(master_seed: int, *labels: str) -> int:
    """Deterministically derive a subseed from a master seed and label path.

    Same ``master_seed`` and ``labels`` always produce the same subseed,
    independent of process, machine or Python hash randomization.
    """
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    if not labels:
        raise ValueError("at least one label is required")

    digest_input = str(master_seed).encode("utf-8")
    for label in labels:
        digest_input += b"\x00" + label.encode("utf-8")
    digest = hashlib.sha256(digest_input).digest()
    return int.from_bytes(digest[:4], byteorder="big") % SEED_MODULUS


def new_generator(master_seed: int, *labels: str) -> np.random.Generator:
    """Return a NumPy PCG64 generator seeded deterministically."""
    return np.random.default_rng(derive_subseed(master_seed, *labels))
