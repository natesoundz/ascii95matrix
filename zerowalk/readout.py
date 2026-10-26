"""Integer-coordinate Crammer--Singer walk-fit readout."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import sys
import time

import numpy as np

from .data import AggregatedRows
from .walk import line_argmin, line_argmin_many


N_CLASSES = 95


def _validate_class_mask(
    class_mask: np.ndarray | None,
    n_classes: int,
) -> np.ndarray | None:
    if class_mask is None:
        return None
    mask = np.asarray(class_mask)
    if mask.dtype != np.bool_ or mask.shape != (n_classes,):
        raise ValueError("class_mask must be a bool array with one entry per class")
    if not np.any(mask):
        raise ValueError("class_mask must include at least one active class")
    return mask


def weighted_loss(scores: np.ndarray, y: np.ndarray, counts: np.ndarray) -> int:
    if y.size == 0:
        return 0
    rows = np.arange(y.size)
    off_label = scores.copy()
    off_label[rows, y] = np.iinfo(np.int64).min
    maximum = off_label.max(axis=1)
    hinges = np.maximum(0, 1 + maximum - scores[rows, y])
    return int(np.dot(counts, hinges))


def softmax_loss(
    scores: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    unit_bits: int = 4,
    class_mask: np.ndarray | None = None,
) -> float:
    if y.size == 0:
        return 0.0
    class_mask = _validate_class_mask(class_mask, scores.shape[1])
    if class_mask is not None and (
        np.any(y < 0)
        or np.any(y >= class_mask.size)
        or np.any(~class_mask[y])
    ):
        raise ValueError("targets must belong to active classes")
    unit = 2.0 ** -unit_bits
    logits = scores.astype(np.float64) * unit
    if class_mask is not None:
        logits[:, ~class_mask] = -np.inf
    maximum = logits.max(axis=1)
    logsumexp = maximum + np.log(
        np.exp(logits - maximum[:, None]).sum(axis=1)
    )
    target_logits = logits[np.arange(y.size), y]
    return float(np.dot(counts.astype(np.float64), logsumexp - target_logits))


def _other_logsumexp(
    scores: np.ndarray,
    class_index: int,
    unit: float,
    class_mask: np.ndarray | None = None,
) -> np.ndarray:
    logits = scores.astype(np.float64) * unit
    class_mask = _validate_class_mask(class_mask, scores.shape[1])
    if class_mask is None:
        other_logits = np.concatenate(
            (logits[:, :class_index], logits[:, class_index + 1:]),
            axis=1,
        )
    else:
        other_classes = class_mask.copy()
        other_classes[class_index] = False
        if not np.any(other_classes):
            return np.full(scores.shape[0], -np.inf, dtype=np.float64)
        other_logits = logits[:, other_classes]
    maximum = other_logits.max(axis=1)
    return maximum + np.log(
        np.exp(other_logits - maximum[:, None]).sum(axis=1)
    )


def top1_accuracy(
    scores: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    class_mask: np.ndarray | None = None,
) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    class_mask = _validate_class_mask(class_mask, scores.shape[1])
    if class_mask is None:
        predictions = np.argmax(scores, axis=1)
    else:
        if np.any(y < 0) or np.any(y >= class_mask.size):
            raise ValueError("targets must be valid class indices")
        active_classes = np.flatnonzero(class_mask)
        predictions = active_classes[
            np.argmax(scores[:, active_classes], axis=1)
        ]
    correct = int(np.dot(counts, predictions == y))
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


def _color_features(X: np.ndarray) -> list[np.ndarray]:
    binary = (X != 0).astype(np.float64)
    conflicts = binary.T @ binary > 0
    np.fill_diagonal(conflicts, False)
    assigned = np.full(X.shape[1], -1, dtype=np.int64)
    groups: list[list[int]] = []
    for feature_index in range(X.shape[1]):
        previous = assigned[:feature_index][
            conflicts[feature_index, :feature_index]
        ]
        forbidden = set(int(color) for color in previous)
        color = 0
        while color in forbidden:
            color += 1
        assigned[feature_index] = color
        while len(groups) <= color:
            groups.append([])
        groups[color].append(feature_index)
    return [
        np.asarray(group, dtype=np.int64)
        for group in groups
    ]


def _class_coordinate_proposals(
    X: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    scores: np.ndarray,
    class_index: int,
    loss: str,
    unit_bits: int,
    max_k: int,
    W: np.ndarray | None = None,
    protected: np.ndarray | None = None,
    class_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int, int, bool]:
    if protected is not None:
        protected = np.asarray(protected)
        if protected.dtype != np.bool_ or protected.shape != y.shape:
            raise ValueError("protected must be a bool mask with one entry per row")
    target = y
    target_score = scores[np.arange(target.size), target]
    class_score = scores[:, class_index]
    if loss == "hinge":
        off_index, off_score_one, off_score_two = _off_label_top_two(
            scores, y
        )
        other_off_score = np.where(
            off_index == class_index, off_score_two, off_score_one
        )
        target_rows = y == class_index
        target_off_score = off_score_one
    else:
        unit = 2.0 ** -unit_bits
        lse_other = _other_logsumexp(
            scores,
            class_index,
            unit,
            class_mask,
        )

    proposals = np.zeros(X.shape[1], dtype=np.int64)
    coordinates_walked = 0
    coordinates_moved = 0
    comparisons = 0
    changed = False

    for feature_index in range(X.shape[1]):
        active = np.flatnonzero(X[:, feature_index] != 0)
        coordinates_walked += 1
        if active.size == 0:
            continue
        x = X[active, feature_index]
        count = counts[active]
        score_c = class_score[active]
        score_y = target_score[active]
        if loss == "hinge":
            is_target = target[active] == class_index
            other_score = other_off_score[active]
            target_other = target_off_score[active]

            def line_value(t: int) -> int:
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
        else:
            other_lse = lse_other[active]
            target_logit = score_y.astype(np.float64) * unit
            counts_float = count.astype(np.float64)
            is_target = target[active] == class_index

            def line_value(t: int) -> float:
                displacement = int(t) * x
                updated = (score_c + displacement).astype(np.float64) * unit
                terms = np.logaddexp(other_lse, updated) - target_logit
                if np.any(is_target):
                    terms = terms - np.where(
                        is_target,
                        displacement.astype(np.float64) * unit,
                        0.0,
                    )
                return float(np.dot(counts_float, terms))

        baseline = line_value(0)
        line_cache = {0: baseline}

        def line_delta(t: int):
            t = int(t)
            if t not in line_cache:
                line_cache[t] = line_value(t)
            return line_cache[t] - baseline

        t_star, used = line_argmin(
            line_delta,
            max_k=max_k,
        )
        comparisons += used
        if protected is not None:
            protected_active = protected[active]
            if np.any(protected_active):
                lower = int(np.iinfo(np.int64).min)
                upper = int(np.iinfo(np.int64).max)
                protected_positions = np.flatnonzero(protected_active)
                for position in protected_positions:
                    difference = (
                        int(score_y[position]) - int(score_c[position])
                    )
                    if difference <= 0:
                        raise AssertionError(
                            "zero offset violates a protected row constraint"
                        )
                    value = int(x[position])
                    if value > 0:
                        upper = min(upper, (difference - 1) // value)
                    else:
                        magnitude = -value
                        quotient, remainder = divmod(difference, magnitude)
                        ceiling = quotient + int(remainder != 0)
                        lower = max(lower, 1 - ceiling)
                if not lower <= 0 <= upper:
                    raise AssertionError(
                        "zero offset must be feasible for protected rows"
                    )
                t_star = min(max(int(t_star), lower), upper)
        proposals[feature_index] = t_star
        if t_star and W is not None:
            W[class_index, feature_index] += t_star
            scores[:, class_index] += t_star * X[:, feature_index]
            class_score = scores[:, class_index]
            coordinates_moved += 1
            changed = True

    return proposals, coordinates_walked, coordinates_moved, comparisons, changed


def _class_block_proposals(
    X: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    scores: np.ndarray,
    class_index: int,
    loss: str,
    unit_bits: int,
    max_k: int,
    color_groups: list[np.ndarray],
    W: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int, int, bool]:
    target_score = scores[np.arange(y.size), y]
    if loss == "hinge":
        off_index, off_score_one, off_score_two = _off_label_top_two(
            scores, y
        )
        other_off_score = np.where(
            off_index == class_index, off_score_two, off_score_one
        )
        target_other = off_score_one
        is_target_all = y == class_index
    else:
        unit = 2.0 ** -unit_bits
        lse_other = _other_logsumexp(scores, class_index, unit)
        target_logit_all = target_score.astype(np.float64) * unit
        counts_float_all = counts.astype(np.float64)
        is_target_all = y == class_index

    proposals = np.zeros(X.shape[1], dtype=np.int64)
    coordinates_walked = 0
    coordinates_moved = 0
    comparisons = 0
    changed = False

    for group in color_groups:
        coordinates_walked += group.size
        group_values = X[:, group]
        active_rows, group_slots = np.nonzero(group_values)
        if not active_rows.size:
            continue
        active_slots = np.unique(group_slots)
        segments = group_slots.astype(np.int64, copy=False)
        x = group_values[active_rows, group_slots]
        count = counts[active_rows]
        score_c = scores[active_rows, class_index]
        score_y = target_score[active_rows]
        is_target = is_target_all[active_rows]

        if loss == "hinge":
            other_score = other_off_score[active_rows]
            target_off_score = target_other[active_rows]

            def hinge_terms(updated: np.ndarray) -> np.ndarray:
                hinges = np.empty(active_rows.size, dtype=np.int64)
                not_target = ~is_target
                if np.any(not_target):
                    competitor = np.maximum(
                        updated[not_target],
                        other_score[not_target],
                    )
                    hinges[not_target] = np.maximum(
                        0,
                        1 + competitor - score_y[not_target],
                    )
                if np.any(is_target):
                    hinges[is_target] = np.maximum(
                        0,
                        1 + target_off_score[is_target] - updated[is_target],
                    )
                return hinges

            baseline_terms = hinge_terms(score_c)
            baseline_by_slot = np.zeros(group.size, dtype=np.int64)
            np.add.at(
                baseline_by_slot,
                segments,
                count * baseline_terms,
            )
        else:
            other_lse = lse_other[active_rows]
            target_logit = target_logit_all[active_rows]
            counts_float = counts_float_all[active_rows]
            baseline_logits = score_c.astype(np.float64) * unit
            baseline_terms = (
                np.logaddexp(other_lse, baseline_logits) - target_logit
            )
            baseline_by_slot = np.bincount(
                segments,
                weights=counts_float * baseline_terms,
                minlength=group.size,
            )

        def L_many(lines: np.ndarray, steps: np.ndarray) -> np.ndarray:
            requested_slots = active_slots[lines]
            requested = np.zeros(group.size, dtype=bool)
            requested[requested_slots] = True
            selected = requested[segments]
            selected_slots = segments[selected]
            step_by_slot = np.zeros(group.size, dtype=np.int64)
            step_by_slot[requested_slots] = steps
            displacement = step_by_slot[selected_slots] * x[selected]
            updated = score_c[selected] + displacement

            if loss == "hinge":
                terms = hinge_terms_for_selected(
                    updated,
                    selected,
                    is_target,
                    other_score,
                    target_off_score,
                    score_y,
                )
                totals = np.zeros(group.size, dtype=np.int64)
                np.add.at(
                    totals,
                    selected_slots,
                    count[selected] * terms,
                )
            else:
                updated_logits = updated.astype(np.float64) * unit
                terms = (
                    np.logaddexp(other_lse[selected], updated_logits)
                    - target_logit[selected]
                )
                terms = terms - np.where(
                    is_target[selected],
                    displacement.astype(np.float64) * unit,
                    0.0,
                )
                totals = np.bincount(
                    selected_slots,
                    weights=counts_float[selected] * terms,
                    minlength=group.size,
                )
            return totals[requested_slots] - baseline_by_slot[requested_slots]

        if loss == "hinge":
            def hinge_terms_for_selected(
                updated: np.ndarray,
                selected: np.ndarray,
                selected_is_target: np.ndarray,
                selected_other_score: np.ndarray,
                selected_target_other: np.ndarray,
                selected_score_y: np.ndarray,
            ) -> np.ndarray:
                is_target_selected = selected_is_target[selected]
                other_score_selected = selected_other_score[selected]
                target_other_selected = selected_target_other[selected]
                score_y_selected = selected_score_y[selected]
                hinges = np.empty(updated.size, dtype=np.int64)
                not_target = ~is_target_selected
                if np.any(not_target):
                    competitor = np.maximum(
                        updated[not_target],
                        other_score_selected[not_target],
                    )
                    hinges[not_target] = np.maximum(
                        0,
                        1 + competitor - score_y_selected[not_target],
                    )
                if np.any(is_target_selected):
                    hinges[is_target_selected] = np.maximum(
                        0,
                        1
                        + target_other_selected[is_target_selected]
                        - updated[is_target_selected],
                    )
                return hinges

        active_offsets, used = line_argmin_many(
            L_many,
            active_slots.size,
            max_k=max_k,
        )
        block_offsets = np.zeros(group.size, dtype=np.int64)
        block_offsets[active_slots] = active_offsets
        proposals[group] = block_offsets
        comparisons += used
        moved = int(np.count_nonzero(block_offsets))
        if W is not None and moved:
            W[class_index, group] += block_offsets
            scores[:, class_index] += group_values @ block_offsets
            coordinates_moved += moved
            changed = True

    return proposals, coordinates_walked, coordinates_moved, comparisons, changed


_WORKER_X = None
_WORKER_Y = None
_WORKER_COUNTS = None
_WORKER_SCORES = None
_WORKER_LOSS = None
_WORKER_UNIT_BITS = None
_WORKER_MAX_K = None


def _initialize_proposal_worker(
    X: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    scores: np.ndarray,
    loss: str,
    unit_bits: int,
    max_k: int,
) -> None:
    global _WORKER_X, _WORKER_Y, _WORKER_COUNTS, _WORKER_SCORES
    global _WORKER_LOSS, _WORKER_UNIT_BITS, _WORKER_MAX_K
    _WORKER_X = X
    _WORKER_Y = y
    _WORKER_COUNTS = counts
    _WORKER_SCORES = scores
    _WORKER_LOSS = loss
    _WORKER_UNIT_BITS = unit_bits
    _WORKER_MAX_K = max_k


def _propose_class_worker(class_index: int):
    proposals, walked, _, comparisons, _ = _class_coordinate_proposals(
        _WORKER_X,
        _WORKER_Y,
        _WORKER_COUNTS,
        _WORKER_SCORES,
        class_index,
        _WORKER_LOSS,
        _WORKER_UNIT_BITS,
        _WORKER_MAX_K,
    )
    return class_index, proposals, walked, comparisons


def _objective(
    scores: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    loss: str,
    unit_bits: int,
    class_mask: np.ndarray | None = None,
) -> int | float:
    if loss == "hinge":
        return weighted_loss(scores, y, counts)
    if class_mask is None:
        return softmax_loss(scores, y, counts, unit_bits)
    return softmax_loss(scores, y, counts, unit_bits, class_mask)


def _joint_score_delta(X: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    int64_min = np.iinfo(np.int64).min
    if np.any(X == int64_min) or np.any(proposals == int64_min):
        return X @ proposals.T
    abs_X = np.abs(X)
    abs_proposals = np.abs(proposals)
    if (
        np.max(abs_X, initial=0) < 2**53
        and np.max(abs_proposals, initial=0) < 2**53
    ):
        absolute_bound = (
            abs_X.astype(np.float64)
            @ abs_proposals.astype(np.float64).T
        )
        if np.max(absolute_bound, initial=0.0) < 2**53:
            return (
                X.astype(np.float64)
                @ proposals.astype(np.float64).T
            ).astype(np.int64)
    return X @ proposals.T


def fit_readout(
    rows: AggregatedRows,
    max_sweeps: int = 50,
    max_k: int = 40,
    progress: bool = True,
    stop_acc: float | None = None,
    patience: int = 3,
    loss: str = "softmax",
    unit_bits: int = 4,
    strategy: str = "coordinate",
    workers: int = 1,
    active_classes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if rows.X.shape[0] == 0:
        raise ValueError("training split contains no ground-truth positions")
    if max_sweeps < 0:
        raise ValueError("max_sweeps must be nonnegative")
    if loss not in {"softmax", "hinge"}:
        raise ValueError("loss must be 'softmax' or 'hinge'")
    if unit_bits < 0 or unit_bits > 1074:
        raise ValueError("unit_bits must be in 0..1074")
    if stop_acc is not None and not 0 <= stop_acc <= 1:
        raise ValueError("stop_acc must be in [0, 1]")
    if patience < 1:
        raise ValueError("patience must be positive")
    if strategy not in {"coordinate", "joint", "blocks"}:
        raise ValueError("strategy must be 'coordinate', 'joint', or 'blocks'")
    if workers < 1:
        raise ValueError("workers must be positive")
    if strategy == "blocks" and workers > 1:
        raise ValueError("blocks strategy uses vectorized parallelism; workers must be 1")
    class_mask = _validate_class_mask(active_classes, N_CLASSES)
    if class_mask is not None:
        if strategy != "coordinate":
            raise ValueError("active_classes is supported only for coordinate strategy")
        if loss != "softmax":
            raise ValueError("active_classes is supported only for softmax loss")
        if (
            np.any(rows.y < 0)
            or np.any(rows.y >= N_CLASSES)
            or np.any(~class_mask[rows.y])
        ):
            raise ValueError("training targets must belong to active classes")
    X = rows.X
    y = rows.y
    counts = rows.counts
    started = time.monotonic()
    color_groups = _color_features(X) if strategy == "blocks" else None
    W = np.zeros((N_CLASSES, X.shape[1]), dtype=np.int64)
    scores = X @ W.T
    objective = _objective(
        scores,
        y,
        counts,
        loss,
        unit_bits,
        class_mask,
    )
    initial_objective = objective
    loss_per_sweep: list[int | float] = []
    accuracy_per_sweep: list[float] = []
    initial_accuracy = top1_accuracy(scores, y, counts, class_mask)
    best_train_accuracy = float(initial_accuracy)
    best_train_accuracy_sweep = 0
    stale_sweeps = 0
    coordinates_walked = 0
    coordinates_moved = 0
    comparisons = 0
    sweeps = 0
    stop_reason = "max_sweeps"
    class_count = (
        N_CLASSES if class_mask is None else int(np.count_nonzero(class_mask))
    )
    total_classes = max_sweeps * class_count
    classes_done = 0
    joint_steps = []

    if loss == "hinge" and objective == 0:
        stop_reason = "zero_loss"
    elif stop_acc is not None and initial_accuracy >= stop_acc:
        stop_reason = "stop_acc"
    else:
        for sweep in range(max_sweeps):
            changed = False
            if strategy == "coordinate":
                for class_index in range(N_CLASSES):
                    if class_mask is not None and not class_mask[class_index]:
                        continue
                    _, walked, moved, used, class_changed = (
                        _class_coordinate_proposals(
                            X,
                            y,
                            counts,
                            scores,
                            class_index,
                            loss,
                            unit_bits,
                            max_k,
                            W,
                            class_mask=class_mask,
                        )
                    )
                    coordinates_walked += walked
                    coordinates_moved += moved
                    comparisons += used
                    changed |= class_changed
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
            elif strategy == "blocks":
                for class_index in range(N_CLASSES):
                    _, walked, moved, used, class_changed = (
                        _class_block_proposals(
                            X,
                            y,
                            counts,
                            scores,
                            class_index,
                            loss,
                            unit_bits,
                            max_k,
                            color_groups,
                            W,
                        )
                    )
                    coordinates_walked += walked
                    coordinates_moved += moved
                    comparisons += used
                    changed |= class_changed
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
            else:
                frozen_scores = scores.copy()
                proposals = np.zeros_like(W)
                if workers > 1:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_initialize_proposal_worker,
                        initargs=(
                            X,
                            y,
                            counts,
                            frozen_scores,
                            loss,
                            unit_bits,
                            max_k,
                        ),
                    ) as executor:
                        class_results = executor.map(
                            _propose_class_worker,
                            range(N_CLASSES),
                        )
                        for class_index, proposal, walked, used in class_results:
                            proposals[class_index] = proposal
                            coordinates_walked += walked
                            comparisons += used
                            classes_done += 1
                            if progress and (
                                classes_done == total_classes
                                or class_index % 5 == 4
                            ):
                                elapsed = time.monotonic() - started
                                rate = classes_done / elapsed if elapsed else 0
                                remaining = (
                                    (total_classes - classes_done) / rate
                                    if rate
                                    else 0
                                )
                                sys.stderr.write(
                                    f"\rzerowalk fit: sweep {sweep + 1}/"
                                    f"{max_sweeps}, class {class_index + 1}/95, "
                                    f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s"
                                )
                                sys.stderr.flush()
                else:
                    for class_index in range(N_CLASSES):
                        proposal, walked, _, used, _ = (
                            _class_coordinate_proposals(
                                X,
                                y,
                                counts,
                                frozen_scores,
                                class_index,
                                loss,
                                unit_bits,
                                max_k,
                            )
                        )
                        proposals[class_index] = proposal
                        coordinates_walked += walked
                        comparisons += used
                        classes_done += 1
                        if progress and (
                            classes_done == total_classes
                            or class_index % 5 == 4
                        ):
                            elapsed = time.monotonic() - started
                            rate = classes_done / elapsed if elapsed else 0
                            remaining = (
                                (total_classes - classes_done) / rate
                                if rate
                                else 0
                            )
                            sys.stderr.write(
                                f"\rzerowalk fit: sweep {sweep + 1}/"
                                f"{max_sweeps}, class {class_index + 1}/95, "
                                f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s"
                            )
                            sys.stderr.flush()

                nonzero_proposals = int(np.count_nonzero(proposals))
                baseline = _objective(scores, y, counts, loss, unit_bits)
                accepted = False
                h = 0
                while True:
                    scaled_proposals = (
                        np.sign(proposals).astype(np.int64)
                        * (np.abs(proposals) >> h)
                    )
                    if not np.any(scaled_proposals):
                        break
                    delta_scores = _joint_score_delta(X, scaled_proposals)
                    objective_one = _objective(
                        scores + delta_scores,
                        y,
                        counts,
                        loss,
                        unit_bits,
                    )
                    comparisons += 1
                    if objective_one < baseline:
                        line_cache = {0: baseline, 1: objective_one}

                        def line_value(t: int):
                            t = int(t)
                            if t not in line_cache:
                                line_cache[t] = _objective(
                                    scores + t * delta_scores,
                                    y,
                                    counts,
                                    loss,
                                    unit_bits,
                                )
                            return line_cache[t] - baseline

                        t_star, used = line_argmin(line_value, max_k=max_k)
                        comparisons += used
                        W += t_star * scaled_proposals
                        scores += t_star * delta_scores
                        moved = int(np.count_nonzero(scaled_proposals))
                        coordinates_moved += moved
                        changed = bool(t_star and moved)
                        joint_steps.append({
                            "h": h,
                            "t": int(t_star),
                            "fallback": False,
                            "nonzero_proposals": nonzero_proposals,
                        })
                        accepted = True
                        break
                    h += 1

                if not accepted:
                    for class_index in range(N_CLASSES):
                        _, walked, moved, used, class_changed = (
                            _class_coordinate_proposals(
                                X,
                                y,
                                counts,
                                scores,
                                class_index,
                                loss,
                                unit_bits,
                                max_k,
                                W,
                            )
                        )
                        coordinates_walked += walked
                        coordinates_moved += moved
                        comparisons += used
                        changed |= class_changed
                    joint_steps.append({
                        "h": None,
                        "t": None,
                        "fallback": True,
                        "nonzero_proposals": nonzero_proposals,
                    })

            sweeps = sweep + 1
            objective = _objective(
                scores,
                y,
                counts,
                loss,
                unit_bits,
                class_mask,
            )
            accuracy = float(top1_accuracy(scores, y, counts, class_mask))
            loss_per_sweep.append(objective)
            accuracy_per_sweep.append(accuracy)
            if accuracy > best_train_accuracy:
                best_train_accuracy = accuracy
                best_train_accuracy_sweep = sweeps
                stale_sweeps = 0
            else:
                stale_sweeps += 1
            if loss == "hinge" and objective == 0:
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
        "final_train_accuracy": top1_accuracy(
            scores,
            y,
            counts,
            class_mask,
        ),
        "best_train_accuracy": best_train_accuracy,
        "best_train_accuracy_sweep": best_train_accuracy_sweep,
        "initial_loss": initial_objective,
        "final_loss": objective,
        "loss": loss,
        "unit_bits": unit_bits,
        "stop_reason": stop_reason,
        "stop_acc": stop_acc,
        "patience": patience,
        "strategy": strategy,
        "workers": workers,
        "joint_steps": joint_steps,
    }
    if color_groups is not None:
        stats["colors"] = len(color_groups)
        stats["max_color_size"] = max(
            (int(group.size) for group in color_groups),
            default=0,
        )
    return W, stats


def scores_for(rows: AggregatedRows, W: np.ndarray) -> np.ndarray:
    return rows.X @ W.T
