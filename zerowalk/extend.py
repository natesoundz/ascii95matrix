"""Single-class coordinate extension with optional no-flip protection."""
from __future__ import annotations

import time

import numpy as np

from .data import AggregatedRows
from .readout import (
    N_CLASSES,
    _class_coordinate_proposals,
    softmax_loss,
    top1_accuracy,
)


def add_class(
    rows: AggregatedRows,
    W_base: np.ndarray,
    class_index: int,
    protect: bool,
    max_sweeps: int = 50,
    patience: int = 3,
    unit_bits: int = 4,
    max_k: int = 40,
) -> tuple[np.ndarray, dict]:
    started = time.monotonic()
    if rows.X.shape[0] == 0:
        raise ValueError("training split contains no ground-truth positions")
    if not 0 <= class_index < N_CLASSES:
        raise ValueError("class_index must be in 0..94")
    if W_base.shape != (N_CLASSES, rows.X.shape[1]):
        raise ValueError("W_base has incompatible shape")
    if not np.issubdtype(W_base.dtype, np.integer):
        raise ValueError("W_base must contain integer weights")
    if max_sweeps < 0:
        raise ValueError("max_sweeps must be nonnegative")
    if patience < 1:
        raise ValueError("patience must be positive")
    if unit_bits < 0 or unit_bits > 1074:
        raise ValueError("unit_bits must be in 0..1074")

    W_new = np.asarray(W_base, dtype=np.int64).copy()
    W_new[class_index] = 0
    base_scores = rows.X @ W_new.T
    base_active = np.ones(N_CLASSES, dtype=bool)
    base_active[class_index] = False
    base_scores[:, ~base_active] = np.iinfo(np.int64).min
    target_scores = base_scores[np.arange(rows.y.size), rows.y]
    other_scores = base_scores.copy()
    other_scores[np.arange(rows.y.size), rows.y] = np.iinfo(np.int64).min
    strict_correct = target_scores > other_scores.max(axis=1)
    would_be_protected = strict_correct & (rows.y != class_index)
    if np.any(would_be_protected):
        initial_bias = min(
            0,
            min(
                int(score) - 1
                for score in target_scores[would_be_protected]
            ),
        )
    else:
        initial_bias = 0
    protected_rows = would_be_protected if protect else None

    X_aug = np.concatenate(
        (
            rows.X,
            np.ones((rows.X.shape[0], 1), dtype=rows.X.dtype),
        ),
        axis=1,
    )
    W_aug = np.zeros(
        (N_CLASSES, rows.X.shape[1] + 1),
        dtype=np.int64,
    )
    W_aug[:, :-1] = W_new
    W_aug[class_index, -1] = initial_bias
    scores = X_aug @ W_aug.T

    objective = softmax_loss(scores, rows.y, rows.counts, unit_bits)
    initial_loss = objective
    initial_accuracy = top1_accuracy(scores, rows.y, rows.counts)
    best_accuracy = float(initial_accuracy)
    stale_sweeps = 0
    loss_per_sweep: list[float] = []
    accuracy_per_sweep: list[float] = []
    comparisons = 0
    coordinates_walked = 0
    coordinates_moved = 0
    sweeps = 0
    stop_reason = "max_sweeps"

    for sweep in range(max_sweeps):
        _, walked, moved, used, changed = _class_coordinate_proposals(
            X_aug,
            rows.y,
            rows.counts,
            scores,
            class_index,
            "softmax",
            unit_bits,
            max_k,
            W_aug,
            protected=protected_rows,
        )
        coordinates_walked += walked
        coordinates_moved += moved
        comparisons += used
        sweeps = sweep + 1
        objective = softmax_loss(scores, rows.y, rows.counts, unit_bits)
        accuracy = float(top1_accuracy(scores, rows.y, rows.counts))
        loss_per_sweep.append(objective)
        accuracy_per_sweep.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            stale_sweeps = 0
        else:
            stale_sweeps += 1
        if not changed:
            stop_reason = "coordinate_wise_optimum"
            break
        if stale_sweeps >= patience:
            stop_reason = "patience"
            break

    stats = {
        "sweeps": sweeps,
        "comparisons": comparisons,
        "wall_time_seconds": time.monotonic() - started,
        "loss_per_sweep": loss_per_sweep,
        "train_top1_accuracy_per_sweep": accuracy_per_sweep,
        "initial_loss": initial_loss,
        "final_loss": objective,
        "initial_train_accuracy": initial_accuracy,
        "final_train_accuracy": top1_accuracy(
            scores,
            rows.y,
            rows.counts,
        ),
        "coordinates_walked": coordinates_walked,
        "coordinates_moved": coordinates_moved,
        "stop_reason": stop_reason,
        "patience": patience,
        "unit_bits": unit_bits,
        "protected_rows": (
            int(np.count_nonzero(would_be_protected))
            if protect
            else 0
        ),
        "new_class_bias": {
            "initial": int(initial_bias),
            "final": int(W_aug[class_index, -1]),
        },
    }
    return W_aug[:, :-1].copy(), stats
