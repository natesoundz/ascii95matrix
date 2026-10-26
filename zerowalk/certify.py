"""Strict-margin prediction-invariance intervals for fitted weights."""
from __future__ import annotations

import sys
import time

import numpy as np

from .data import AggregatedRows
from .walk import walk_edge


def certify(
    rows: AggregatedRows,
    W: np.ndarray,
    max_k: int = 40,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scores = rows.X @ W.T
    predictions = np.argmax(scores, axis=1)
    row_indices = np.arange(rows.y.size)
    predicted_score = scores[row_indices, predictions]
    other_scores = scores.copy()
    other_scores[row_indices, predictions] = np.iinfo(np.int64).min
    margins = predicted_score - other_scores.max(axis=1)
    strict = margins > 0
    tied_count = int(np.count_nonzero(~strict))
    tied_positions = int(rows.counts[~strict].sum())
    X = rows.X[strict]
    scores = scores[strict]
    predictions = predictions[strict]
    predicted_score = predicted_score[strict]
    counts = rows.counts[strict]
    del counts
    n_rows, dimension = X.shape
    n_classes = W.shape[0]
    slack_up = np.zeros((n_classes, dimension), dtype=np.int64)
    slack_down = np.zeros((n_classes, dimension), dtype=np.int64)
    started = time.monotonic()

    for class_index in range(n_classes):
        other = scores.copy()
        other[:, class_index] = np.iinfo(np.int64).min
        other_max = other.max(axis=1)
        class_score = scores[:, class_index]
        is_predicted = predictions == class_index

        for direction, result_slack in ((1, slack_up), (-1, slack_down)):
            def ahead(indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
                still_same = np.ones(indices.size, dtype=bool)
                for row_start in range(0, n_rows, 4096):
                    row_end = min(n_rows, row_start + 4096)
                    x = X[row_start:row_end, indices]
                    moved_score = (
                        class_score[row_start:row_end, None]
                        + direction * x * offsets[None, :]
                    )
                    predicted_here = is_predicted[row_start:row_end, None]
                    remains = np.where(
                        predicted_here,
                        moved_score > other_max[row_start:row_end, None],
                        predicted_score[row_start:row_end, None] > moved_score,
                    )
                    still_same &= np.all(remains, axis=0)
                return still_same

            edge, _, unbounded = walk_edge(
                ahead, dimension, max_k=max_k
            )
            result_slack[class_index] = np.where(unbounded, -1, edge)
        if progress and (class_index % 5 == 4 or class_index + 1 == n_classes):
            elapsed = time.monotonic() - started
            sys.stderr.write(
                f"\rzerowalk certify: class {class_index + 1}/{n_classes}, "
                f"elapsed {elapsed:.1f}s"
            )
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    unbounded_any = (slack_up < 0) | (slack_down < 0)
    finite_slack = np.concatenate((slack_up[slack_up >= 0], slack_down[slack_down >= 0]))
    interval_contains_zero = (
        ((slack_down < 0) | (W <= slack_down))
        & ((slack_up < 0) | (-W <= slack_up))
    )
    stats = {
        "strict_margin_rows": int(np.count_nonzero(strict)),
        "base_tied_rows_excluded": tied_count,
        "base_tied_positions_excluded": tied_positions,
        "unbounded_side_fraction": float(np.count_nonzero(unbounded_any) / W.size),
        "weights_with_unbounded_side": int(np.count_nonzero(unbounded_any)),
        "weight_count": int(W.size),
        "median_finite_slack": (
            float(np.median(finite_slack)) if finite_slack.size else None
        ),
        "finite_slack_side_count": int(finite_slack.size),
        "weights_whose_interval_contains_zero_individually": int(
            np.count_nonzero(interval_contains_zero)
        ),
        "interval_zero_note": "individually, not jointly",
        "max_k": int(max_k),
    }
    return slack_up, slack_down, stats
