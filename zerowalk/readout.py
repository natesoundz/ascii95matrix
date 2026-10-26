"""Integer-coordinate Crammer--Singer walk-fit readout."""
from __future__ import annotations

import sys
import time

import numpy as np

from .data import AggregatedRows
from .walk import line_argmin


N_CLASSES = 95


def weighted_loss(scores: np.ndarray, y: np.ndarray, counts: np.ndarray) -> int:
    if y.size == 0:
        return 0
    rows = np.arange(y.size)
    off_label = scores.copy()
    off_label[rows, y] = np.iinfo(np.int64).min
    maximum = off_label.max(axis=1)
    hinges = np.maximum(0, 1 + maximum - scores[rows, y])
    return int(np.dot(counts, hinges))


def top1_accuracy(scores: np.ndarray, y: np.ndarray, counts: np.ndarray) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    correct = int(np.dot(counts, np.argmax(scores, axis=1) == y))
    return correct / total


def _off_label_top_two(
    scores: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = scores.shape[0]
    candidates = np.argpartition(scores, kth=N_CLASSES - 3, axis=1)[:, -3:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    candidates = np.take_along_axis(candidates, order, axis=1)
    candidate_scores = np.take_along_axis(candidate_scores, order, axis=1)

    first_index = np.full(rows, -1, dtype=np.int64)
    first_score = np.full(rows, np.iinfo(np.int64).min, dtype=np.int64)
    second_score = np.full(rows, np.iinfo(np.int64).min, dtype=np.int64)
    for rank in range(3):
        valid = candidates[:, rank] != y
        new_first = valid & (first_index < 0)
        first_index[new_first] = candidates[new_first, rank]
        first_score[new_first] = candidate_scores[new_first, rank]
        new_second = valid & ~new_first & (
            second_score == np.iinfo(np.int64).min
        )
        second_score[new_second] = candidate_scores[new_second, rank]
    return first_index, first_score, second_score


def fit_readout(
    rows: AggregatedRows,
    max_sweeps: int = 50,
    max_k: int = 40,
    progress: bool = True,
    stop_acc: float | None = None,
    patience: int = 3,
) -> tuple[np.ndarray, dict]:
    if rows.X.shape[0] == 0:
        raise ValueError("training split contains no ground-truth positions")
    if max_sweeps < 0:
        raise ValueError("max_sweeps must be nonnegative")
    if stop_acc is not None and not 0 <= stop_acc <= 1:
        raise ValueError("stop_acc must be in [0, 1]")
    if patience < 1:
        raise ValueError("patience must be positive")
    started = time.monotonic()
    W = np.zeros((N_CLASSES, rows.X.shape[1]), dtype=np.int64)
    scores = rows.X @ W.T
    loss = weighted_loss(scores, rows.y, rows.counts)
    loss_per_sweep: list[int] = []
    accuracy_per_sweep: list[float] = []
    initial_accuracy = top1_accuracy(scores, rows.y, rows.counts)
    best_train_accuracy = float(initial_accuracy)
    best_train_accuracy_sweep = 0
    stale_sweeps = 0
    coordinates_walked = 0
    coordinates_moved = 0
    comparisons = 0
    sweeps = 0
    stop_reason = "max_sweeps"
    total_classes = max_sweeps * N_CLASSES
    classes_done = 0

    if loss == 0:
        stop_reason = "zero_loss"
    elif stop_acc is not None and initial_accuracy >= stop_acc:
        stop_reason = "stop_acc"
    else:
        for sweep in range(max_sweeps):
            changed = False
            for class_index in range(N_CLASSES):
                off_index, off_score_one, off_score_two = _off_label_top_two(
                    scores, rows.y
                )
                other_off_score = np.where(
                    off_index == class_index, off_score_two, off_score_one
                )
                target_rows = rows.y == class_index
                target_off_score = off_score_one
                target = rows.y
                target_score = scores[np.arange(target.size), target]
                class_score = scores[:, class_index]

                for feature_index in range(rows.X.shape[1]):
                    active = np.flatnonzero(rows.X[:, feature_index] != 0)
                    coordinates_walked += 1
                    if active.size == 0:
                        continue
                    x = rows.X[active, feature_index]
                    labels = target[active]
                    count = rows.counts[active]
                    score_c = class_score[active]
                    score_y = target_score[active]
                    is_target = target_rows[active]
                    other_score = other_off_score[active]
                    target_other = target_off_score[active]

                    def line_loss(t: int) -> int:
                        updated = score_c + int(t) * x
                        hinges = np.empty(active.size, dtype=np.int64)
                        not_target = ~is_target
                        if np.any(not_target):
                            competitor = np.maximum(
                                updated[not_target], other_score[not_target]
                            )
                            hinges[not_target] = np.maximum(
                                0,
                                1 + competitor - score_y[not_target],
                            )
                        if np.any(is_target):
                            hinges[is_target] = np.maximum(
                                0,
                                1 + target_other[is_target] - updated[is_target],
                            )
                        return int(np.dot(count, hinges))

                    baseline = line_loss(0)
                    t_star, used = line_argmin(
                        lambda t: line_loss(t) - baseline, max_k=max_k
                    )
                    comparisons += used
                    if t_star:
                        W[class_index, feature_index] += t_star
                        scores[:, class_index] += t_star * rows.X[:, feature_index]
                        class_score = scores[:, class_index]
                        coordinates_moved += 1
                        changed = True

                classes_done += 1
                if progress and (
                    classes_done == total_classes or class_index % 5 == 4
                ):
                    elapsed = time.monotonic() - started
                    rate = classes_done / elapsed if elapsed else 0
                    remaining = (
                        (total_classes - classes_done) / rate if rate else 0
                    )
                    sys.stderr.write(
                        f"\rzerowalk fit: sweep {sweep + 1}/{max_sweeps}, "
                        f"class {class_index + 1}/95, elapsed {elapsed:.1f}s, "
                        f"ETA {remaining:.1f}s"
                    )
                    sys.stderr.flush()

            sweeps = sweep + 1
            loss = weighted_loss(scores, rows.y, rows.counts)
            accuracy = float(top1_accuracy(scores, rows.y, rows.counts))
            loss_per_sweep.append(loss)
            accuracy_per_sweep.append(accuracy)
            if accuracy > best_train_accuracy:
                best_train_accuracy = accuracy
                best_train_accuracy_sweep = sweeps
                stale_sweeps = 0
            else:
                stale_sweeps += 1
            if loss == 0:
                stop_reason = "zero_loss"
                break
            if stop_acc is not None and accuracy >= stop_acc:
                stop_reason = "stop_acc"
                break
            if not changed:
                stop_reason = "coordinate_wise_optimum"
                break
            if stale_sweeps >= patience:
                stop_reason = "patience"
                break
        if progress and total_classes:
            sys.stderr.write("\n")
            sys.stderr.flush()

    stats = {
        "sweeps": sweeps,
        "coordinates_walked": coordinates_walked,
        "coordinates_moved": coordinates_moved,
        "comparisons": comparisons,
        "wall_time_seconds": time.monotonic() - started,
        "loss_per_sweep": loss_per_sweep,
        "train_top1_accuracy_per_sweep": accuracy_per_sweep,
        "initial_train_accuracy": initial_accuracy,
        "final_train_accuracy": top1_accuracy(scores, rows.y, rows.counts),
        "best_train_accuracy": best_train_accuracy,
        "best_train_accuracy_sweep": best_train_accuracy_sweep,
        "initial_loss": weighted_loss(rows.X @ np.zeros_like(W).T, rows.y, rows.counts),
        "final_loss": loss,
        "stop_reason": stop_reason,
        "stop_acc": stop_acc,
        "patience": patience,
    }
    return W, stats


def scores_for(rows: AggregatedRows, W: np.ndarray) -> np.ndarray:
    return rows.X @ W.T
