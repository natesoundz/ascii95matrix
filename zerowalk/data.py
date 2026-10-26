"""Ground-truth records, context aggregation, and contiguous splits."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Iterator

import numpy as np

from compile_weights import ground_truth_positions, records
from .workbook import WorkbookEncoding


@dataclass(frozen=True)
class AggregatedRows:
    X: np.ndarray
    y: np.ndarray
    counts: np.ndarray
    context_keys: np.ndarray

    @property
    def positions(self) -> int:
        return int(self.counts.sum())


@dataclass(frozen=True)
class PreparedData:
    train: AggregatedRows
    test: AggregatedRows
    accepted_records: int
    rejected_records: int
    train_records: int
    test_records: int


def _strip_line_ending(line: str) -> str:
    return line.rstrip("\r\n")


def _text_lines(path: Path) -> Iterator[str]:
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    has_start = False
    has_end = False
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            has_start |= start_marker in line
            has_end |= end_marker in line
    if has_start != has_end:
        raise ValueError("Gutenberg input must contain both START and END markers")

    inside = not has_start
    with path.open(encoding="utf-8", newline="") as stream:
        for raw_line in stream:
            line = _strip_line_ending(raw_line)
            if start_marker in line:
                inside = True
                continue
            if end_marker in line:
                break
            if inside:
                yield line


def _record_texts(mode: str, path: Path):
    if mode == "text":
        for text in _text_lines(path):
            if text:
                yield text, None
        return
    for record in records(path):
        text = record.get("text")
        if not isinstance(text, str):
            yield None, None
            continue
        try:
            positions = ground_truth_positions(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid ground_truth_positions: {exc}") from exc
        yield text, positions


def _valid_text(text: str | None) -> bool:
    if not text or not text.isascii():
        return False
    values = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    return bool(np.all((values >= 32) & (values <= 126)))


def _count_records(mode: str, path: Path) -> tuple[int, int]:
    accepted = 0
    rejected = 0
    for text, _ in _record_texts(mode, path):
        if _valid_text(text):
            accepted += 1
        else:
            rejected += 1
    return accepted, rejected


def _aggregate_record(
    text: str,
    positions,
    offsets: int,
    profile_id: np.ndarray,
):
    if positions is None:
        positions = np.arange(len(text), dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)

    encoded_text = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    target = encoded_text[positions].astype(np.int64) - 32
    profile_by_ascii = profile_id
    powers = np.power(27, np.arange(offsets, dtype=np.int64), dtype=np.int64)
    context_code = np.zeros(positions.size, dtype=np.int64)
    for offset_index in range(offsets):
        source_positions = positions - offset_index - 1
        present = source_positions >= 0
        profile_values = np.full(positions.size, 26, dtype=np.int64)
        profile_values[present] = profile_by_ascii[
            encoded_text[source_positions[present]].astype(np.int64) - 32
        ]
        context_code += profile_values * powers[offset_index]
    keys = np.column_stack((context_code, target))
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    return unique, counts.astype(np.int64, copy=False)


def _finish_rows(
    key_parts: list[np.ndarray],
    count_parts: list[np.ndarray],
    offsets: int,
    profile_table: np.ndarray,
) -> AggregatedRows:
    if not key_parts:
        dim = 1 + 94 * offsets
        return AggregatedRows(
            X=np.empty((0, dim), dtype=np.int64),
            y=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int64),
            context_keys=np.empty(0, dtype=np.int64),
        )
    joined_keys = np.concatenate(key_parts, axis=0)
    joined_counts = np.concatenate(count_parts)
    unique, inverse = np.unique(joined_keys, axis=0, return_inverse=True)
    aggregated_counts = np.zeros(unique.shape[0], dtype=np.int64)
    np.add.at(aggregated_counts, inverse, joined_counts)

    powers = np.power(27, np.arange(offsets, dtype=np.int64), dtype=np.int64)
    profile_ids = (unique[:, 0, None] // powers[None, :]) % 27
    X = np.zeros((unique.shape[0], 1 + 94 * offsets), dtype=np.int64)
    X[:, 0] = 1
    for offset_index in range(offsets):
        ids = profile_ids[:, offset_index]
        present = ids < 26
        X[present, 1 + 94 * offset_index:1 + 94 * (offset_index + 1)] = (
            profile_table[ids[present]]
        )
    return AggregatedRows(
        X=X,
        y=unique[:, 1].astype(np.int64, copy=False),
        counts=aggregated_counts,
        context_keys=unique[:, 0].astype(np.int64, copy=False),
    )


def prepare_data(
    mode: str,
    path: str | Path,
    offsets: int,
    encoding: WorkbookEncoding,
) -> PreparedData:
    """Aggregate ground-truth positions and split accepted records 80/20."""
    if mode not in {"text", "records"}:
        raise ValueError("mode must be 'text' or 'records'")
    if offsets < 1 or offsets > 13:
        raise ValueError("offsets must be in 1..13 for int64 context keys")
    path = Path(path)
    accepted, rejected = _count_records(mode, path)
    if accepted == 0:
        raise ValueError("input contains no accepted nonempty printable-ASCII records")
    train_records = (4 * accepted) // 5
    profile_ids = np.full(95, 26, dtype=np.int64)
    profile_ids[:] = encoding.profile_id
    key_parts = {"train": [], "test": []}
    count_parts = {"train": [], "test": []}
    accepted_index = 0
    started = time.monotonic()
    for text, positions in _record_texts(mode, path):
        if not _valid_text(text):
            continue
        split = "train" if accepted_index < train_records else "test"
        accepted_index += 1
        keys, counts = _aggregate_record(
            text, positions, offsets, profile_ids
        )
        if keys.size:
            key_parts[split].append(keys)
            count_parts[split].append(counts)
        if accepted_index % 50000 == 0 or accepted_index == accepted:
            elapsed = time.monotonic() - started
            rate = accepted_index / elapsed if elapsed else 0
            remaining = (accepted - accepted_index) / rate if rate else 0
            sys.stderr.write(
                f"\rzerowalk data: record {accepted_index}/{accepted}, "
                f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s"
            )
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()

    return PreparedData(
        train=_finish_rows(
            key_parts["train"], count_parts["train"], offsets,
            encoding.profile_table,
        ),
        test=_finish_rows(
            key_parts["test"], count_parts["test"], offsets,
            encoding.profile_table,
        ),
        accepted_records=accepted,
        rejected_records=rejected,
        train_records=train_records,
        test_records=accepted - train_records,
    )
