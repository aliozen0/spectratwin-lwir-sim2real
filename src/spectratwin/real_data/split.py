"""Deterministic, sequence-aware train/dev/benchmark split.

Splits by whole ``sequence_key`` groups, never by individual frame, so
adjacent frames of one clip cannot land on both sides of a split.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from spectratwin.randomness.seed import new_generator
from spectratwin.real_data.records import FlirSampleRecord

SPLIT_POLICY_VERSION = "flir-sequence-split-v1"

REAL_TRAIN = "real_train"
REAL_DEV = "real_dev"
REAL_BENCHMARK = "real_benchmark"


class SplitRatios(BaseModel):
    model_config = ConfigDict(frozen=True)

    real_train: float = 0.7
    real_dev: float = 0.15
    real_benchmark: float = 0.15

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitRatios:
        total = self.real_train + self.real_dev + self.real_benchmark
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


DEFAULT_SPLIT_RATIOS = SplitRatios()


def split_sequences(
    sequence_keys: list[str], master_seed: int, ratios: SplitRatios | None = None
) -> dict[str, list[str]]:
    """Assign each sequence key to exactly one split role, deterministically."""
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    unique_keys = sorted(set(sequence_keys))
    generator = new_generator(master_seed, "real-data-split", SPLIT_POLICY_VERSION)
    shuffled = generator.permutation(len(unique_keys))

    n = len(unique_keys)
    n_train = round(n * ratios.real_train)
    n_dev = round(n * ratios.real_dev)

    ordered = [unique_keys[i] for i in shuffled]
    return {
        REAL_TRAIN: ordered[:n_train],
        REAL_DEV: ordered[n_train : n_train + n_dev],
        REAL_BENCHMARK: ordered[n_train + n_dev :],
    }


def split_records(
    records: list[FlirSampleRecord], master_seed: int, ratios: SplitRatios | None = None
) -> dict[str, list[FlirSampleRecord]]:
    """Split full sample records using the same sequence-level assignment."""
    sequence_to_role = {}
    role_assignment = split_sequences([r.sequence_key for r in records], master_seed, ratios)
    for role, keys in role_assignment.items():
        for key in keys:
            sequence_to_role[key] = role

    result: dict[str, list[FlirSampleRecord]] = {REAL_TRAIN: [], REAL_DEV: [], REAL_BENCHMARK: []}
    for record in records:
        result[sequence_to_role[record.sequence_key]].append(record)
    return result
