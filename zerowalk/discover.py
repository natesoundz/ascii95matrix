"""Occurrence-bound discovery and merge-tree analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

from .data import (
    AggregatedRows,
    _count_records,
    _record_texts,
    _valid_text,
)
from .readout import top1_accuracy
from .workbook import WorkbookEncoding


@dataclass(frozen=True)
class DiscoveryDataset:
    train: AggregatedRows
    test: AggregatedRows
    accepted_records: int
    rejected_records: int
    train_records: int
    test_records: int
    feature_names: tuple[str, ...]
    occurrence_chars: tuple[str, ...]


def _aggregate_discovery_record(
    text: str,
    positions,
    offsets: int,
) -> tuple[np.ndarray, np.ndarray]:
    if positions is None:
        positions = np.arange(len(text), dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return (
            np.empty((0, offsets + 1), dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    encoded_text = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    contexts = np.full((positions.size, offsets), 95, dtype=np.int64)
    for offset_index in range(offsets):
        source_positions = positions - offset_index - 1
        present = source_positions >= 0
        contexts[present, offset_index] = (
            encoded_text[source_positions[present]].astype(np.int64) - 32
        )
    targets = encoded_text[positions].astype(np.int64) - 32
    keys = np.column_stack((contexts, targets))
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    return unique, counts.astype(np.int64, copy=False)


def _finish_discovery_rows(
    key_parts: list[np.ndarray],
    count_parts: list[np.ndarray],
    offsets: int,
    encoding: WorkbookEncoding,
    occurrence_chars: tuple[str, ...],
) -> AggregatedRows:
    dimension = 1 + 94 * offsets + offsets * len(occurrence_chars)
    if not key_parts:
        return AggregatedRows(
            X=np.empty((0, dimension), dtype=np.int64),
            y=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int64),
            context_keys=np.empty((0, offsets), dtype=np.int64),
        )
    joined_keys = np.concatenate(key_parts, axis=0)
    joined_counts = np.concatenate(count_parts)
    unique, inverse = np.unique(joined_keys, axis=0, return_inverse=True)
    counts = np.zeros(unique.shape[0], dtype=np.int64)
    np.add.at(counts, inverse, joined_counts)

    context_chars = unique[:, :offsets]
    X = np.zeros((unique.shape[0], dimension), dtype=np.int64)
    X[:, 0] = 1
    for offset_index in range(offsets):
        codes = context_chars[:, offset_index]
        present = codes < 95
        profile_ids = np.empty(codes.size, dtype=np.int64)
        profile_ids[present] = encoding.profile_id[codes[present]]
        X[present, 1 + 94 * offset_index:1 + 94 * (offset_index + 1)] = (
            encoding.profile_table[profile_ids[present]]
        )
        occurrence_start = 1 + 94 * offsets + offset_index * len(occurrence_chars)
        for char_index, char in enumerate(occurrence_chars):
            X[:, occurrence_start + char_index] = codes == ord(char) - 32

    return AggregatedRows(
        X=X,
        y=unique[:, offsets].astype(np.int64, copy=False),
        counts=counts,
        context_keys=context_chars.astype(np.int64, copy=False),
    )


def prepare_discovery_data(
    mode: str,
    path: str | Path,
    offsets: int,
    encoding: WorkbookEncoding,
    progress: bool = True,
) -> DiscoveryDataset:
    if mode not in {"text", "records"}:
        raise ValueError("mode must be 'text' or 'records'")
    if offsets < 1 or offsets > 13:
        raise ValueError("offsets must be in 1..13")
    path = Path(path)
    accepted, rejected = _count_records(mode, path)
    if accepted == 0:
        raise ValueError("input contains no accepted nonempty printable-ASCII records")
    train_records = (4 * accepted) // 5
    occurrence_chars = tuple(sorted(
        {
            char
            for group in encoding.groups
            if len(group) > 1
            for char in group
        },
        key=ord,
    ))
    feature_names = (
        ("bias",)
        + tuple(
            f"offset_-{offset}:{name}"
            for offset in range(1, offsets + 1)
            for name in encoding.feature_names
        )
        + tuple(
            f"occurrence.o{offset}.{char}"
            for offset in range(1, offsets + 1)
            for char in occurrence_chars
        )
    )
    key_parts = {"train": [], "test": []}
    count_parts = {"train": [], "test": []}
    accepted_index = 0
    started = time.monotonic()
    for text, positions in _record_texts(mode, path):
        if not _valid_text(text):
            continue
        split = "train" if accepted_index < train_records else "test"
        accepted_index += 1
        keys, counts = _aggregate_discovery_record(text, positions, offsets)
        if keys.size:
            key_parts[split].append(keys)
            count_parts[split].append(counts)
        if progress and (
            accepted_index % 50000 == 0 or accepted_index == accepted
        ):
            elapsed = time.monotonic() - started
            rate = accepted_index / elapsed if elapsed else 0
            remaining = (accepted - accepted_index) / rate if rate else 0
            sys.stderr.write(
                f"\rzerowalk discover data: record {accepted_index}/{accepted}, "
                f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s"
            )
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return DiscoveryDataset(
        train=_finish_discovery_rows(
            key_parts["train"],
            count_parts["train"],
            offsets,
            encoding,
            occurrence_chars,
        ),
        test=_finish_discovery_rows(
            key_parts["test"],
            count_parts["test"],
            offsets,
            encoding,
            occurrence_chars,
        ),
        accepted_records=accepted,
        rejected_records=rejected,
        train_records=train_records,
        test_records=accepted - train_records,
        feature_names=feature_names,
        occurrence_chars=occurrence_chars,
    )


def _correct_positions(
    rows: AggregatedRows,
    scores: np.ndarray,
) -> int:
    return int(np.dot(
        rows.counts,
        np.argmax(scores, axis=1) == rows.y,
    ))


def _occurrence_columns(
    occurrence_chars: tuple[str, ...],
    offsets: int,
) -> np.ndarray:
    base = 1 + 94 * offsets
    columns = np.full((offsets, 95), -1, dtype=np.int64)
    for offset_index in range(offsets):
        start = base + offset_index * len(occurrence_chars)
        for char_index, char in enumerate(occurrence_chars):
            columns[offset_index, ord(char) - 32] = start + char_index
    return columns


def _move_delta(
    rows: AggregatedRows,
    effective_weights: np.ndarray,
    occurrence_columns: np.ndarray,
    moved_chars: tuple[str, ...],
    kept_char: str,
) -> tuple[np.ndarray, np.ndarray]:
    contexts = rows.context_keys
    moved_codes = np.asarray([ord(char) - 32 for char in moved_chars])
    affected = np.isin(contexts, moved_codes).any(axis=1)
    affected_rows = np.flatnonzero(affected)
    if affected_rows.size == 0:
        return affected_rows, np.empty((0, effective_weights.shape[0]), dtype=np.int64)

    before = contexts[affected_rows]
    after = before.copy()
    move_mask = np.isin(after, moved_codes)
    after[move_mask] = ord(kept_char) - 32
    delta = np.zeros(
        (affected_rows.size, effective_weights.shape[0]),
        dtype=np.int64,
    )
    for offset_index in range(contexts.shape[1]):
        changed = before[:, offset_index] != after[:, offset_index]
        if not np.any(changed):
            continue
        old_columns = occurrence_columns[offset_index, before[changed, offset_index]]
        new_columns = occurrence_columns[offset_index, after[changed, offset_index]]
        if np.any(old_columns < 0) or np.any(new_columns < 0):
            raise ValueError("occurrence feature is missing for a moved character")
        delta[changed] += (
            effective_weights[:, new_columns].T
            - effective_weights[:, old_columns].T
        )
    return affected_rows, delta


def _candidate_cost(
    rows: AggregatedRows,
    scores: np.ndarray,
    effective_weights: np.ndarray,
    occurrence_columns: np.ndarray,
    moved_chars: tuple[str, ...],
    kept_char: str,
) -> int:
    affected_rows, delta = _move_delta(
        rows,
        effective_weights,
        occurrence_columns,
        moved_chars,
        kept_char,
    )
    if affected_rows.size == 0:
        return 0
    old_predictions = np.argmax(scores[affected_rows], axis=1)
    new_predictions = np.argmax(scores[affected_rows] + delta, axis=1)
    old_correct = int(np.dot(
        rows.counts[affected_rows],
        old_predictions == rows.y[affected_rows],
    ))
    new_correct = int(np.dot(
        rows.counts[affected_rows],
        new_predictions == rows.y[affected_rows],
    ))
    return old_correct - new_correct


def _apply_merge(
    rows: AggregatedRows,
    scores: np.ndarray,
    effective_weights: np.ndarray,
    occurrence_columns: np.ndarray,
    moved_chars: tuple[str, ...],
    kept_char: str,
) -> int:
    affected_rows, delta = _move_delta(
        rows,
        effective_weights,
        occurrence_columns,
        moved_chars,
        kept_char,
    )
    if affected_rows.size:
        scores[affected_rows] += delta
    for char in moved_chars:
        code = ord(char) - 32
        for offset_index in range(occurrence_columns.shape[0]):
            source = occurrence_columns[offset_index, code]
            target = occurrence_columns[offset_index, ord(kept_char) - 32]
            effective_weights[:, source] = effective_weights[:, target]
    return _correct_positions(rows, scores)


def build_merge_trees(
    rows: AggregatedRows,
    W: np.ndarray,
    encoding: WorkbookEncoding,
    occurrence_chars: tuple[str, ...],
) -> list[dict]:
    offsets = rows.X.shape[1]
    offsets -= 1
    offsets //= 94 + len(occurrence_chars)
    if offsets < 1 or 1 + offsets * (94 + len(occurrence_chars)) != rows.X.shape[1]:
        raise ValueError("row features do not match occurrence-bound layout")
    occurrence_columns = _occurrence_columns(occurrence_chars, offsets)
    trees = []
    for group in encoding.groups:
        if len(group) <= 1:
            continue
        members = tuple(sorted(group, key=ord))
        clusters = {ord(char) - 32: (char,) for char in members}
        effective_weights = W.copy()
        scores = rows.X @ effective_weights.T
        current_correct = _correct_positions(rows, scores)
        merges = []
        step = 0
        while len(clusters) > 1:
            representatives = sorted(clusters)
            candidates = []
            for left_index, left_rep in enumerate(representatives):
                for right_rep in representatives[left_index + 1:]:
                    left_members = clusters[left_rep]
                    right_members = clusters[right_rep]
                    left_cost = _candidate_cost(
                        rows,
                        scores,
                        effective_weights,
                        occurrence_columns,
                        right_members,
                        chr(left_rep + 32),
                    )
                    right_cost = _candidate_cost(
                        rows,
                        scores,
                        effective_weights,
                        occurrence_columns,
                        left_members,
                        chr(right_rep + 32),
                    )
                    if left_cost <= right_cost:
                        cost = left_cost
                        kept_rep = left_rep
                        moved_members = right_members
                    else:
                        cost = right_cost
                        kept_rep = right_rep
                        moved_members = left_members
                    candidates.append((
                        cost,
                        left_rep,
                        right_rep,
                        kept_rep,
                        moved_members,
                    ))
            cost, left_rep, right_rep, kept_rep, moved_members = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2]),
            )
            left_members = clusters[left_rep]
            right_members = clusters[right_rep]
            kept_char = chr(kept_rep + 32)
            step += 1
            current_correct = _apply_merge(
                rows,
                scores,
                effective_weights,
                occurrence_columns,
                moved_members,
                kept_char,
            )
            merged_members = tuple(sorted(
                left_members + right_members,
                key=ord,
            ))
            clusters[kept_rep] = merged_members
            del clusters[right_rep if kept_rep == left_rep else left_rep]
            merges.append({
                "step": step,
                "cluster_A": "".join(left_members),
                "cluster_B": "".join(right_members),
                "kept_representative": kept_char,
                "moved_cluster": "".join(moved_members),
                "cost_positions": int(cost),
                "cost_fraction": float(cost / rows.positions),
                "train_accuracy_after": float(current_correct / rows.positions),
            })
        trees.append({
            "group": "".join(members),
            "members": "".join(members),
            "initial_train_accuracy": float(
                _correct_positions(rows, rows.X @ W.T) / rows.positions
            ),
            "merges": merges,
        })
    return trees


def _partition_weights(
    W: np.ndarray,
    trees: list[dict],
    occurrence_chars: tuple[str, ...],
    offsets: int,
    tolerance: float,
    train_positions: int,
) -> tuple[np.ndarray, list[dict]]:
    threshold = tolerance * train_positions
    columns = _occurrence_columns(occurrence_chars, offsets)
    partition_weights = W.copy()
    partition_groups = []
    for tree in trees:
        clusters = [(char,) for char in tree["members"]]
        applied = 0
        stopped_at = None
        for merge in tree["merges"]:
            if merge["cost_positions"] > threshold:
                stopped_at = merge["step"]
                break
            kept = merge["kept_representative"]
            moved = merge["moved_cluster"]
            for char in moved:
                code = ord(char) - 32
                for offset_index in range(offsets):
                    source = columns[offset_index, code]
                    target = columns[offset_index, ord(kept) - 32]
                    partition_weights[:, source] = partition_weights[:, target]
            cluster_a = tuple(merge["cluster_A"])
            cluster_b = tuple(merge["cluster_B"])
            clusters.remove(cluster_a)
            clusters.remove(cluster_b)
            clusters.append(tuple(sorted(cluster_a + cluster_b, key=ord)))
            applied += 1
        partition_groups.append({
            "group": tree["group"],
            "clusters": [
                "".join(chars)
                for chars in sorted(clusters, key=lambda members: ord(members[0]))
            ],
            "merges_applied": applied,
            "stopped_at_step": stopped_at,
        })
    return partition_weights, partition_groups


def legacy_reference_matches(
    trees: list[dict],
    legacy_path: str | Path,
) -> dict:
    legacy_path = Path(legacy_path)
    with legacy_path.open(encoding="utf-8", newline="") as stream:
        header = next(stream).rstrip("\r\n").split("\t")
        rows = [
            line.rstrip("\r\n").split("\t")
            for line in stream
            if line.rstrip("\r\n")
        ]
    if len(header) != 258 or len(rows) != 95:
        raise ValueError("legacy reference must have 95 rows and 254 binary columns")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("legacy reference rows must be tab-delimited without quoting")
    row_by_ascii = {
        int(row[1]): np.asarray(row[4:], dtype=np.int8)
        for row in rows
    }
    if set(row_by_ascii) != set(range(32, 127)):
        raise ValueError("legacy reference rows must cover ASCII 32..126")
    matrix = np.vstack([row_by_ascii[code] for code in range(32, 127)])
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("legacy reference features must be binary")

    references = []
    for tree in trees:
        if tree["group"] not in {"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"}:
            continue
        split = tree["merges"][-1]
        side_a = set(split["cluster_A"])
        side_b = set(split["cluster_B"])
        members = tree["members"]
        target = np.asarray([char in side_b for char in members], dtype=bool)
        ascii_rows = np.asarray([ord(char) - 32 for char in members])
        agreements = []
        for feature_index, feature_name in enumerate(header[4:]):
            values = matrix[ascii_rows, feature_index].astype(bool)
            normal = float(np.mean(values == target))
            inverted = float(np.mean((~values) == target))
            polarity = 1 if normal >= inverted else -1
            agreements.append({
                "column": feature_name,
                "agreement": max(normal, inverted),
                "polarity": polarity,
            })
        top = sorted(
            agreements,
            key=lambda item: (-item["agreement"], item["column"]),
        )[:5]
        references.append({
            "group": tree["group"],
            "top_split": {
                "cluster_A": split["cluster_A"],
                "cluster_B": split["cluster_B"],
                "kept_representative": split["kept_representative"],
            },
            "top_legacy_columns": top,
        })
    return {
        "label": "external reference, not used in fit",
        "source_basename": legacy_path.name,
        "groups": references,
    }


def partition_metrics(
    dataset: DiscoveryDataset,
    W: np.ndarray,
    trees: list[dict],
    offsets: int,
    tolerance: float,
) -> dict:
    partition_weights, groups = _partition_weights(
        W,
        trees,
        dataset.occurrence_chars,
        offsets,
        tolerance,
        dataset.train.positions,
    )
    train_scores = dataset.train.X @ partition_weights.T
    test_scores = dataset.test.X @ partition_weights.T

    return {
        "tolerance": float(tolerance),
        "clusters_by_group": groups,
        "train_top1_accuracy": top1_accuracy(
            train_scores,
            dataset.train.y,
            dataset.train.counts,
        ),
        "test_top1_accuracy": top1_accuracy(
            test_scores,
            dataset.test.y,
            dataset.test.counts,
        ),
    }
