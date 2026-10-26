"""Integer-coordinate Crammer--Singer walk-fit readout."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
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


def softmax_loss(
    scores: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    unit_bits: int = 4,
) -> float:
    if y.size == 0:
        return 0.0
    unit = 2.0 ** -unit_bits
    logits = scores.astype(np.float64) * unit
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
) -> np.ndarray:
    logits = scores.astype(np.float64) * unit
    other_logits = np.concatenate(
        (logits[:, :class_index], logits[:, class_index + 1:]),
        axis=1,
    )
    maximum = other_logits.max(axis=1)
    return maximum + np.log(
        np.exp(other_logits - maximum[:, None]).sum(axis=1)
    )


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
) -> tuple[np.ndarray, int, int, int, bool]:
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
        lse_other = _other_logsumexp(scores, class_index, unit)

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
        proposals[feature_index] = t_star
        if t_star and W is not None:
            W[class_index, feature_index] += t_star
            scores[:, class_index] += t_star * X[:, feature_index]
            class_score = scores[:, class_index]
            coordinates_moved += 1
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
) -> int | float:
    return (
        weighted_loss(scores, y, counts)
        if loss == "hinge"
        else softmax_loss(scores, y, counts, unit_bits)
    )


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
    if strategy not in {"coordinate", "joint"}:
        raise ValueError("strategy must be 'coordinate' or 'joint'")
    if workers < 1:
        raise ValueError("workers must be positive")
    X = rows.X
    y = rows.y
    counts = rows.counts
    started = time.monotonic()
    W = np.zeros((N_CLASSES, X.shape[1]), dtype=np.int64)
    scores = X @ W.T
    objective = _objective(scores, y, counts, loss, unit_bits)
    initial_objective = objective
    loss_per_sweep: list[int | float] = []
    accuracy_per_sweep: list[float] = []
    initial_accuracy = top1_accuracy(scores, y, counts)
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
            objective = _objective(scores, y, counts, loss, unit_bits)
            accuracy = float(top1_accuracy(scores, y, counts))
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
        "final_train_accuracy": top1_accuracy(scores, y, counts),
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
    return W, stats


def scores_for(rows: AggregatedRows, W: np.ndarray) -> np.ndarray:
    return rows.X @ W.T
