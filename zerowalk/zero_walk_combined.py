#!/usr/bin/env python3
"""
zero_walk.py - the counting walk, built.

A weight is a position on a line (integer, after quantization). The delta
between an old and a new position is never computed by subtracting weights.
A walker starts on the old position and counts toward the new one:

  * the only decision at each move is DIRECTION, read from which side of the
    goal the walker is on (the zero line of the delta frame);
  * before the first crossing the step DOUBLES on every miss (gallop);
  * the move that crosses the line reverses direction and HALVES the step;
    afterwards the step halves on every move (bisection) down to one unit;
  * the walker stops when it rests exactly on the goal.

The delta is the walker's net displacement, accumulated by counting. The trace
(the ordered list of signed steps) is the reading; replaying it from the start
reproduces the goal, and two traces are compared by direct equality.

A "marble" is a tensor whose last axis holds its flats (dimensions). Its edges
are the n(n-1)/2 pairwise relations between flats, read with the same walk
(start on flat i, goal on flat j).

The same walk runs vectorized (walk_known: many weights at once), through a
comparison only (walk_edge: the goal is unknown in weight space and is read
as "which side" of a monotone predicate, e.g. every ground truth at once), and
as a line search (line_argmin: walk a weight to where one more unit step stops
lowering a convex loss summed over all ground truths).

Input : two JSON files  {"shape": [...], "data": [...]}  (old, new), same shape.
        Integers are used as-is; floats are quantized with --unit.
Output: one JSON file, checkpointed atomically after every chunk, resumable.

Usage:
  python zero_walk.py run  --old old.json --new new.json --out delta.json
  python zero_walk.py selftest
  py -3.12 zero_walk_combined.py fit --rows rows.json --out model.json

The readout fits by Gauss-Seidel sweeps of line_argmin for each
(class, feature) coordinate. It compares candidate positions by summed
softmax loss over all ground truths (hinge is optional; hinge kinks stall
coordinate walks). It stops once accuracy is good enough via stop-acc or
patience rather than requiring zero loss.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

MAX_MOVES = 10_000


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def cmp(a, b):
    return (a > b) - (a < b)


def walk(start, goal):
    """Count from start to goal. Returns (delta, trace); trace is signed steps."""
    p = start
    k = 0                      # step = 1 << k
    bracketed = False
    trace = []
    c = cmp(goal, p)           # +1: goal above walker, -1: below, 0: on it
    while c != 0:
        step = c << k if c > 0 else -(1 << k)
        p += step
        trace.append(step)
        if len(trace) > MAX_MOVES:
            raise RuntimeError("walk exceeded MAX_MOVES")
        c2 = cmp(goal, p)
        if not bracketed:
            if c2 == -c:       # crossed the line: reverse (via c2) and halve
                bracketed = True
                k = max(0, k - 1)
            else:              # miss: double
                k += 1
        else:
            k = max(0, k - 1)
        c = c2
    return sum_by_counting(trace), trace


def bracket_walk(start, goal):
    """Your rule: cross -> halve and reverse, miss -> double, and never step
    onto or past the last point that sat across the goal. Returns (delta, trace)."""
    p, k, edge, trace = start, 0, None, []
    c = cmp(goal, p)
    while c != 0:
        s = 1 << k
        if edge is not None:                      # stay strictly inside the bracket
            while s > 1 and c * (edge - (p + c * s)) <= 0:
                s >>= 1
        prev = p
        p += c * s
        trace.append(c * s)
        if len(trace) > MAX_MOVES:
            raise RuntimeError("bracket_walk exceeded MAX_MOVES")
        c2 = cmp(goal, p)
        if c2 == -c:
            edge = prev                           # point left behind is across the goal from p
            k = max(0, s.bit_length() - 2)        # halve the step actually taken
        else:
            k = s.bit_length()                    # double the step actually taken
        c = c2
    return sum_by_counting(trace), trace


def sum_by_counting(trace):
    """Net displacement accumulated one signed step at a time."""
    d = 0
    for s in trace:
        d += s
    return d


def replay(start, trace):
    p = start
    for s in trace:
        p += s
    return p


def to_signs(trace):
    """A trace as direction marks only: '+' up, '-' down. Step sizes are not kept."""
    return "".join("+" if t > 0 else "-" for t in trace)


def from_signs(start, signs):
    """Rebuild the landing position from the start and the direction marks alone.
    The step size is a pure function of the marks: it doubles while the direction
    holds, and halves from the move after a reversal (the crossing) onward."""
    p, k, bracketed = start, 0, False
    for i, ch in enumerate(signs):
        s = 1 if ch == "+" else -1
        p += s << k if s > 0 else -(1 << k)
        if i + 1 == len(signs):
            break
        nxt = 1 if signs[i + 1] == "+" else -1
        if not bracketed:
            if nxt != s:
                bracketed = True
                k = max(0, k - 1)
            else:
                k += 1
        else:
            k = max(0, k - 1)
    return p


def bracket_walk(start, goal, cap=MAX_MOVES):
    """The rule as stated in conversation: cross -> halve and reverse, miss ->
    double, and never step past a position already known to be on the far side
    of the goal. Landing on that remembered edge counts as a crossing. Returns
    the move count; raises if it fails to land."""
    p, k, n = start, 0, 0
    lo = hi = None             # known below the goal / known above the goal
    c = cmp(goal, p)
    while c != 0:
        if n >= cap:
            raise RuntimeError("bracket_walk did not land")
        if c > 0:
            lo = p if lo is None else max(lo, p)
        else:
            hi = p if hi is None else min(hi, p)
        step, edge = 1 << k, False
        if c > 0 and hi is not None and p + step >= hi:
            step, edge = hi - p, True
        if c < 0 and lo is not None and p - step <= lo:
            step, edge = p - lo, True
        p += c * step
        n += 1
        c2 = cmp(goal, p)
        k = max(0, k - 1) if (c2 == -c or edge) else k + 1
        c = c2
    return n


def literal_walk(start, goal, cap=2000):
    """The rule exactly as first stated: cross -> halve+reverse, miss -> double,
    always. Returns the trace, or None if it never lands (it can cycle)."""
    p, k, n = start, 0, 0
    c = cmp(goal, p)
    while c != 0:
        if n >= cap:
            return None
        p += (1 << k) * c
        n += 1
        c2 = cmp(goal, p)
        k = max(0, k - 1) if c2 == -c else k + 1
        c = c2
    return n


# --------------------------------------------------------------------------
# the vector walk: every weight at once, goals read through comparisons
# --------------------------------------------------------------------------

def walk_known(start, goal, return_traces=False):
    """walk() applied elementwise to integer arrays, all walkers moving in the
    same rounds. Returns (delta, moves) or (delta, moves, traces); each trace is
    identical to walk(start[i], goal[i])[1]. The delta is accumulated from the
    steps, never computed as goal - start."""
    starts = np.asarray(start, dtype=np.int64)
    goals = np.asarray(goal, dtype=np.int64)
    if starts.shape != goals.shape:
        raise ValueError("start and goal must have the same shape")

    position = starts.copy().reshape(-1)
    target = goals.reshape(-1)
    delta = np.zeros(position.size, dtype=np.int64)
    moves = np.zeros(position.size, dtype=np.int64)
    k = np.zeros(position.size, dtype=np.int64)          # step = 1 << k
    bracketed = np.zeros(position.size, dtype=bool)
    traces = [[] for _ in range(position.size)] if return_traces else None
    active = np.flatnonzero(target != position)

    while active.size:
        c = np.where(target[active] > position[active], 1, -1)
        step = c * np.left_shift(np.int64(1), k[active])
        position[active] += step
        delta[active] += step
        moves[active] += 1
        if moves[active].max() > MAX_MOVES:
            raise RuntimeError("walk_known exceeded MAX_MOVES")
        if return_traces:
            for idx, s in zip(active.tolist(), step.tolist()):
                traces[idx].append(s)
        c2 = (target[active] > position[active]).astype(np.int64) - (target[active] < position[active])
        halve = bracketed[active] | (c2 == -c)               # crossed now or before
        bracketed[active] = halve
        k[active] = np.where(halve, np.maximum(0, k[active] - 1), k[active] + 1)
        active = active[c2 != 0]

    delta = delta.reshape(starts.shape)
    moves = moves.reshape(starts.shape)
    if return_traces:
        return delta, moves, traces
    return delta, moves


def walk_edge(ahead, n_lines, max_k=40):
    """The walk when the goal is not given as a number, only as a side.

    ahead(lines, offsets) -> bool per line: True while offset is still on the
    near side of the zero line (the predicate must flip once, monotone). Gallop
    offsets 2**(k+1) - 1 (the distance covered after k+1 doubling moves), then
    halve the bracket down to one unit. Returns (edge, probes, unbounded): edge
    is the last True offset, unbounded marks lines still True at the reach."""
    if n_lines < 0:
        raise ValueError("n_lines must be nonnegative")
    if max_k < 0:
        raise ValueError("max_k must be nonnegative")
    edge = np.zeros(n_lines, dtype=np.int64)
    far = np.zeros(n_lines, dtype=np.int64)               # first False offset seen
    probes = np.zeros(n_lines, dtype=np.int64)
    has_far = np.zeros(n_lines, dtype=bool)
    unbounded = np.zeros(n_lines, dtype=bool)
    active = np.arange(n_lines, dtype=np.int64)

    for k in range(max_k + 1):                            # doubling phase
        if not active.size:
            break
        offset = np.full(active.size, (1 << (k + 1)) - 1, dtype=np.int64)
        ok = np.asarray(ahead(active, offset), dtype=bool)
        if ok.shape != active.shape:
            raise ValueError("ahead must return one boolean per probed line")
        probes[active] += 1
        edge[active[ok]] = offset[ok]
        far[active[~ok]] = offset[~ok]
        has_far[active[~ok]] = True
        active = active[ok]
    unbounded[active] = True

    active = np.flatnonzero(has_far)                      # halving phase
    while active.size:
        active = active[far[active] - edge[active] > 1]
        if not active.size:
            break
        middle = edge[active] + (far[active] - edge[active]) // 2
        ok = np.asarray(ahead(active, middle), dtype=bool)
        if ok.shape != active.shape:
            raise ValueError("ahead must return one boolean per probed line")
        probes[active] += 1
        edge[active[ok]] = middle[ok]
        far[active[~ok]] = middle[~ok]

    return edge, probes, unbounded


def line_argmin(L, max_k=40):
    """Walk one weight along its line to the minimizer of a convex loss L(t)
    (t = integer offset from the current value; L may sum every ground truth).
    The only reading is a comparison: does one more unit step still lower L?
    Returns (offset, comparisons); offset is the minimizer closest to 0."""
    cache = {}

    def at(t):
        t = int(t)
        if t not in cache:
            cache[t] = L(t)
        return cache[t]

    comparisons = 1
    if at(1) < at(0):
        direction = 1
    else:
        comparisons += 1
        if at(-1) < at(0):
            direction = -1
        else:
            return 0, comparisons

    def improving(lines, offsets):
        nonlocal comparisons
        out = np.empty(lines.shape, dtype=bool)
        for i, t in enumerate(offsets.tolist()):
            out[i] = at(direction * (t + 1)) < at(direction * t)
            comparisons += 1
        return out

    edge, _, _ = walk_edge(improving, 1, max_k=max_k)
    return direction * (int(edge[0]) + 1), comparisons


# --------------------------------------------------------------------------
# marbles
# --------------------------------------------------------------------------

def read_marble(old_q, new_q):
    """old_q, new_q: the r flats of one marble (ints).
    Returns delta per flat (+trace) and the n(n-1)/2 edges between delta flats."""
    deltas, traces = [], []
    for a, b in zip(old_q, new_q):
        d, t = walk(a, b)
        deltas.append(d)
        traces.append(to_signs(t))
    edges = []
    r = len(deltas)
    for i in range(r):
        for j in range(i + 1, r):
            e, t = walk(deltas[i], deltas[j])
            edges.append({"i": i, "j": j, "reading": e, "trace": to_signs(t)})
    return deltas, traces, edges


def read_marbles(old_block, new_block):
    """read_marble for m marbles at once. old_block, new_block: (m, r) ints.
    Every flat and every edge of every marble walks in the same vector rounds.
    Returns lists with exactly read_marble's output, one entry per marble."""
    old_a = np.asarray(old_block, dtype=np.int64)
    new_a = np.asarray(new_block, dtype=np.int64)
    m, r = old_a.shape
    d, _, tr = walk_known(old_a, new_a, return_traces=True)
    I, J = np.triu_indices(r, 1)
    e, _, etr = walk_known(d[:, I], d[:, J], return_traces=True)
    npair = len(I)
    out = []
    for mi in range(m):
        deltas = d[mi].tolist()
        traces = [to_signs(tr[mi * r + f]) for f in range(r)]
        edges = [{"i": int(I[p]), "j": int(J[p]), "reading": int(e[mi, p]),
                  "trace": to_signs(etr[mi * npair + p])} for p in range(npair)]
        out.append((deltas, traces, edges))
    return out


# --------------------------------------------------------------------------
# the readout: integer weights searched against every ground truth at once
# --------------------------------------------------------------------------

def weighted_loss(scores, y, counts):
    if y.size == 0:
        return 0
    rows = np.arange(y.size)
    off_label = scores.copy()
    off_label[rows, y] = np.iinfo(np.int64).min
    maximum = off_label.max(axis=1)
    hinges = np.maximum(0, 1 + maximum - scores[rows, y])
    return int(np.dot(counts, hinges))


def softmax_loss(scores, y, counts, unit_bits=4):
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


def _other_logsumexp(scores, class_index, unit):
    logits = scores.astype(np.float64) * unit
    other_logits = np.concatenate(
        (logits[:, :class_index], logits[:, class_index + 1:]),
        axis=1,
    )
    maximum = other_logits.max(axis=1)
    return maximum + np.log(
        np.exp(other_logits - maximum[:, None]).sum(axis=1)
    )


def top1_accuracy(scores, y, counts):
    total = int(counts.sum())
    if total == 0:
        return None
    correct = int(np.dot(counts, np.argmax(scores, axis=1) == y))
    return correct / total


def _off_label_top_two(scores, y, n_classes):
    if n_classes < 3:
        raise ValueError("n_classes must be at least 3")
    rows = scores.shape[0]
    candidates = np.argpartition(
        scores,
        kth=n_classes - 3,
        axis=1,
    )[:, -3:]
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
    X,
    y,
    counts,
    n_classes,
    loss="softmax",
    unit_bits=4,
    max_sweeps=50,
    patience=3,
    stop_acc=None,
    max_k=40,
    progress=True,
):
    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if X.shape[0] == 0:
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
    if n_classes < 3:
        raise ValueError("n_classes must be at least 3")
    unit = 2.0 ** -unit_bits
    started = time.monotonic()
    W = np.zeros((n_classes, X.shape[1]), dtype=np.int64)
    scores = X @ W.T
    objective = (
        weighted_loss(scores, y, counts)
        if loss == "hinge"
        else softmax_loss(scores, y, counts, unit_bits)
    )
    initial_objective = objective
    loss_per_sweep = []
    accuracy_per_sweep = []
    initial_accuracy = top1_accuracy(scores, y, counts)
    best_train_accuracy = float(initial_accuracy)
    best_train_accuracy_sweep = 0
    stale_sweeps = 0
    coordinates_walked = 0
    coordinates_moved = 0
    comparisons = 0
    sweeps = 0
    stop_reason = "max_sweeps"
    total_classes = max_sweeps * n_classes
    classes_done = 0

    if loss == "hinge" and objective == 0:
        stop_reason = "zero_loss"
    elif stop_acc is not None and initial_accuracy >= stop_acc:
        stop_reason = "stop_acc"
    else:
        for sweep in range(max_sweeps):
            changed = False
            for class_index in range(n_classes):
                target = y
                target_score = scores[np.arange(target.size), target]
                class_score = scores[:, class_index]
                if loss == "hinge":
                    off_index, off_score_one, off_score_two = _off_label_top_two(
                        scores, y, n_classes
                    )
                    other_off_score = np.where(
                        off_index == class_index, off_score_two, off_score_one
                    )
                    target_rows = y == class_index
                    target_off_score = off_score_one
                else:
                    lse_other = _other_logsumexp(scores, class_index, unit)

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

                        def line_value(t):
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

                        def line_value(t):
                            displacement = int(t) * x
                            updated = (
                                score_c + displacement
                            ).astype(np.float64) * unit
                            terms = np.logaddexp(other_lse, updated) - target_logit
                            if np.any(is_target):
                                terms = terms - np.where(
                                    is_target,
                                    displacement.astype(np.float64) * unit,
                                    0.0,
                                )
                            return float(np.dot(counts_float, terms))

                    baseline = line_value(0)
                    t_star, used = line_argmin(
                        lambda t: line_value(t) - baseline,
                        max_k=max_k,
                    )
                    comparisons += used
                    if t_star:
                        W[class_index, feature_index] += t_star
                        scores[:, class_index] += t_star * X[:, feature_index]
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
                        f"class {class_index + 1}/{n_classes}, "
                        f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s"
                    )
                    sys.stderr.flush()

            sweeps = sweep + 1
            objective = (
                weighted_loss(scores, y, counts)
                if loss == "hinge"
                else softmax_loss(scores, y, counts, unit_bits)
            )
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
    }
    return W, stats


def certify(X, y, counts, W, max_k=40, progress=True):
    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    scores = X @ W.T
    predictions = np.argmax(scores, axis=1)
    row_indices = np.arange(y.size)
    predicted_score = scores[row_indices, predictions]
    other_scores = scores.copy()
    other_scores[row_indices, predictions] = np.iinfo(np.int64).min
    margins = predicted_score - other_scores.max(axis=1)
    strict = margins > 0
    tied_count = int(np.count_nonzero(~strict))
    tied_positions = int(counts[~strict].sum())
    X = X[strict]
    scores = scores[strict]
    predictions = predictions[strict]
    predicted_score = predicted_score[strict]
    counts = counts[strict]
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
            def ahead(indices, offsets):
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

            edge, _, unbounded = walk_edge(ahead, dimension, max_k=max_k)
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
    finite_slack = np.concatenate((
        slack_up[slack_up >= 0],
        slack_down[slack_down >= 0],
    ))
    interval_contains_zero = (
        ((slack_down < 0) | (W <= slack_down))
        & ((slack_up < 0) | (-W <= slack_up))
    )
    stats = {
        "strict_margin_rows": int(np.count_nonzero(strict)),
        "base_tied_rows_excluded": tied_count,
        "base_tied_positions_excluded": tied_positions,
        "unbounded_side_fraction": float(
            np.count_nonzero(unbounded_any) / W.size
        ),
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


# --------------------------------------------------------------------------
# io, checkpointing, progress
# --------------------------------------------------------------------------

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_tensor(path, unit):
    with open(path, "r", encoding="utf-8") as f:
        t = json.load(f)
    shape, data = t["shape"], t["data"]
    n = 1
    for s in shape:
        n *= s
    if n != len(data):
        raise ValueError("%s: shape %r does not match %d values" % (path, shape, len(data)))
    if all(isinstance(x, int) and not isinstance(x, bool) for x in data):
        return shape, data
    if unit is None:
        raise ValueError("%s has non-integer values; pass --unit to quantize" % path)
    return shape, [int(round(x / unit)) for x in data]


def _load_fit_rows(path, unit):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("X"), list):
        raise ValueError("rows JSON must contain a two-dimensional X array")
    raw_X = data["X"]
    if not raw_X or not all(isinstance(row, list) for row in raw_X):
        raise ValueError("X must be a nonempty rectangular array")
    dimension = len(raw_X[0])
    if dimension == 0 or any(len(row) != dimension for row in raw_X):
        raise ValueError("X must be a nonempty rectangular array")
    if unit is not None and (not math.isfinite(unit) or unit <= 0):
        raise ValueError("--unit must be finite and positive")

    values = []
    for row in raw_X:
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("X values must be numeric")
            if unit is None:
                if not isinstance(value, int):
                    raise ValueError("X must contain integers unless --unit is given")
                converted.append(value)
            else:
                if not math.isfinite(value):
                    raise ValueError("X values must be finite")
                converted.append(int(round(value / unit)))
        values.append(converted)
    X = np.asarray(values, dtype=np.int64)

    raw_y = data.get("y")
    if not isinstance(raw_y, list) or len(raw_y) != X.shape[0] or not raw_y:
        raise ValueError("y must contain one integer label per X row")
    if any(
        isinstance(label, bool) or not isinstance(label, int)
        for label in raw_y
    ):
        raise ValueError("y labels must be integers")
    y = np.asarray(raw_y, dtype=np.int64)
    raw_counts = data.get("counts", [1] * y.size)
    if (
        not isinstance(raw_counts, list)
        or len(raw_counts) != y.size
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for count in raw_counts
        )
    ):
        raise ValueError("counts must contain one positive integer per X row")
    counts = np.asarray(raw_counts, dtype=np.int64)

    n_classes = data.get("classes", int(y.max()) + 1)
    if isinstance(n_classes, bool) or not isinstance(n_classes, int):
        raise ValueError("classes must be an integer")
    if n_classes < 3:
        raise ValueError("classes must be at least 3")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("y labels must be in [0, classes)")
    return X, y, counts, n_classes


def fit_command(args):
    X, y, counts, n_classes = _load_fit_rows(args.rows, args.unit)
    W, stats = fit_readout(
        X,
        y,
        counts,
        n_classes,
        loss=args.loss,
        unit_bits=args.unit_bits,
        max_sweeps=args.max_sweeps,
        patience=args.patience,
        stop_acc=args.stop_acc,
        progress=True,
    )
    certificate = None
    if args.certify:
        _, _, certificate = certify(X, y, counts, W)
    scores = X @ W.T
    with open(args.rows, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    output = {
        "version": 1,
        "rows": {
            "basename": Path(args.rows).name,
            "sha256": digest,
        },
        "classes": n_classes,
        "dimension": int(X.shape[1]),
        "loss": args.loss,
        "unit_bits": args.unit_bits,
        "W": {
            "shape": list(W.shape),
            "data": W.tolist(),
        },
        "stats": stats,
        "train_top1_accuracy": top1_accuracy(scores, y, counts),
        "certify": certificate,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(output_path), output)
    print(
        f"fit: {args.loss}, {stats['sweeps']} sweeps, "
        f"train accuracy {output['train_top1_accuracy']}, "
        f"stop={stats['stop_reason']} -> {output_path}"
    )
    return 0


def fmt_time(s):
    s = int(max(0, s))
    return "%d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


class Progress:
    def __init__(self, total):
        self.total = total
        self.t0 = time.time()
        self.base = 0

    def show(self, done, label=""):
        el = time.time() - self.t0
        fresh = done - self.base
        left = self.total - done
        eta = (el / fresh * left) if fresh > 0 else 0
        width = 28
        fill = int(width * done / self.total) if self.total else width
        sys.stderr.write("\r[%s%s] %d/%d  left %d  elapsed %s  ETA %s  %s\x1b[K" % (
            "#" * fill, "." * (width - fill), done, self.total, left,
            fmt_time(el), fmt_time(eta), label))
        sys.stderr.flush()


class TwoStrike:
    """First SIGINT: finish the current chunk of marbles, save, exit cleanly.
    Second SIGINT: hard exit now (last checkpoint is already on disk)."""
    def __init__(self):
        self.strikes = 0
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, signum, frame):
        self.strikes += 1
        if self.strikes >= 2:
            sys.stderr.write("\nsecond interrupt: exiting now\n")
            os._exit(130)
        sys.stderr.write("\ninterrupt: saving at the next chunk boundary "
                         "(press again to exit immediately)\n")

    @property
    def requested(self):
        return self.strikes > 0


def run(args):
    shape_o, old = load_tensor(args.old, args.unit)
    shape_n, new = load_tensor(args.new, args.unit)
    if shape_o != shape_n:
        raise ValueError("shape mismatch: %r vs %r" % (shape_o, shape_n))
    r = shape_o[-1]
    marbles = len(old) // r
    config = {"old": os.path.abspath(args.old), "new": os.path.abspath(args.new),
              "shape": shape_o, "unit": args.unit, "values": len(old)}

    state = {"version": 1, "config": config, "done": 0, "complete": False,
             "delta": [], "traces": [], "edges": []}
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("config") == config:
            state = prev
            sys.stderr.write("resuming at marble %d of %d\n" % (state["done"], marbles))
        else:
            raise SystemExit("%s exists with a different configuration; "
                             "choose another --out" % args.out)

    guard = TwoStrike()
    prog = Progress(marbles)
    prog.base = state["done"]
    old_a = np.asarray(old, dtype=np.int64).reshape(marbles, r)
    new_a = np.asarray(new, dtype=np.int64).reshape(marbles, r)

    while state["done"] < marbles and not guard.requested:
        m = state["done"]
        n = min(args.every, marbles - m)              # one chunk walks in vector rounds
        for deltas, traces, edges in read_marbles(old_a[m:m + n], new_a[m:m + n]):
            state["delta"].extend(deltas)
            state["traces"].extend(traces)
            state["edges"].append(edges)
        state["done"] = m + n
        prog.show(state["done"], "marbles %d-%d" % (m, m + n - 1))
        atomic_write_json(args.out, state)

    state["complete"] = state["done"] == marbles
    atomic_write_json(args.out, state)
    sys.stderr.write("\n%s: %d/%d marbles -> %s\n" % (
        "complete" if state["complete"] else "paused", state["done"], marbles, args.out))
    return 0 if state["complete"] else 130


# --------------------------------------------------------------------------
# selftest: exhaustive, no sampling
# --------------------------------------------------------------------------

def selftest(args):
    lim = args.range
    worst, count = 0, 0
    prog = Progress((2 * lim + 1) ** 2 // 1)
    total = 0
    t_last = 0.0
    for a in range(-lim, lim + 1):
        for b in range(-lim, lim + 1):
            d, t = walk(a, b)
            assert replay(a, t) == b, ("replay mismatch", a, b)
            assert d == sum_by_counting(t)
            assert from_signs(a, to_signs(t)) == b, ("signs do not recover goal", a, b)
            assert (walk(a, b)[1] == t), "walk is not deterministic"
            worst = max(worst, len(t))
            count += 1
            total += 1
        if time.time() - t_last > 0.25:
            prog.show(count, "a=%d" % a)
            t_last = time.time()
    prog.show(count, "done")
    sys.stderr.write("\n")
    print("walk: %d start/goal pairs in [-%d, %d], every one lands exactly; "
          "longest trace %d moves" % (count, lim, lim, worst))

    # the literal rule, stated plainly
    bad = None
    for g in range(0, lim + 1):
        if literal_walk(0, g) is None:
            bad = g
            break
    if bad is None:
        print("literal rule (miss=double always): landed for every goal tested")
    else:
        print("literal rule (miss=double always): never lands for goal %d "
              "(it cycles); the gallop-then-bisect rule lands every time" % bad)

    # the bracketed form of the stated rule
    r = min(lim, 60)
    longest = max(bracket_walk(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1))
    print("bracket rule (never past a remembered far-side position): lands every "
          "pair in [-%d, %d]; longest %d moves" % (r, r, longest))
    print("start + direction marks alone recover the goal for every pair above")

    # marbles
    d, tr, ed = read_marble([5, -3, 0, 7, 2], [1, 4, 0, -7, 2])
    assert d == [-4, 7, 0, -14, 0], d
    assert len(ed) == 10
    assert all(from_signs(d[e["i"]], e["trace"]) == d[e["j"]] for e in ed)
    print("marble: 5 flats -> 10 edges, every edge trace replays onto its flat")

    # vector walk: same traces as walk() for every pair, all walked at once
    grid = np.arange(-lim, lim + 1, dtype=np.int64)
    A, B = np.meshgrid(grid, grid, indexing="ij")
    vd, vm, vt = walk_known(A, B, return_traces=True)
    flat_a, flat_b = A.ravel().tolist(), B.ravel().tolist()
    for idx, (a, b) in enumerate(zip(flat_a, flat_b)):
        d, t = walk(a, b)
        assert vt[idx] == t and vd.ravel()[idx] == d, ("walk_known mismatch", a, b)
    assert int(vm.max()) == worst
    print("walk_known: all %d pairs walked in vector rounds, every trace equals "
          "walk()" % vd.size)

    # marbles in vector rounds equal marbles one at a time
    flats = np.arange(-6, 7, dtype=np.int64)
    old_b = np.stack([np.roll(flats, s)[:5] for s in range(13)])
    new_b = np.stack([np.roll(flats[::-1], 3 * s)[:5] for s in range(13)])
    got = read_marbles(old_b, new_b)
    for mi in range(len(old_b)):
        assert got[mi] == read_marble(old_b[mi].tolist(), new_b[mi].tolist())
    print("read_marbles: %d marbles in vector rounds, identical to read_marble" % len(old_b))

    # walk_edge: goal read only as a side; must find every threshold exactly
    goals = np.arange(0, 4 * lim + 1, dtype=np.int64)
    found, probes, unb = walk_edge(lambda li, off: off <= goals[li], len(goals))
    assert np.array_equal(found, goals) and not unb.any()
    _, _, unb2 = walk_edge(lambda li, off: np.ones(len(li), bool), 3, max_k=5)
    assert unb2.all()
    print("walk_edge: %d hidden edges found exactly from side readings; "
          "most probes %d" % (len(goals), int(probes.max())))

    # line_argmin: convex losses, checked against every integer in range
    cases = 0
    for c0 in range(-lim, lim + 1, 7):
        for w in (1, 3):
            for flat in (0, 2):
                L = lambda t: w * max(0, abs(t - c0) - flat)
                ts = range(-2 * lim, 2 * lim + 1)
                best = min(L(t) for t in ts)
                closest = min((t for t in ts if L(t) == best), key=abs)
                got_t, _ = line_argmin(L)
                assert L(got_t) == best and abs(got_t) == abs(closest), (c0, w, flat, got_t)
                cases += 1
    print("line_argmin: %d convex lines, each lands on the minimizer nearest 0" % cases)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--old", required=True)
    r.add_argument("--new", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--unit", type=float, default=None,
                   help="quantization unit for float inputs (value = k * unit)")
    r.add_argument("--every", type=int, default=64,
                   help="marbles walked together per chunk; checkpoint after each chunk")
    s = sub.add_parser("selftest")
    s.add_argument("--range", type=int, default=150)
    f = sub.add_parser("fit")
    f.add_argument("--rows", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--loss", choices=("softmax", "hinge"), default="softmax")
    f.add_argument("--unit-bits", type=int, default=4)
    f.add_argument("--max-sweeps", type=int, default=50)
    f.add_argument("--patience", type=int, default=3)
    f.add_argument("--stop-acc", type=float, default=None)
    f.add_argument("--certify", action="store_true")
    f.add_argument("--unit", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "run":
        return run(args)
    if args.cmd == "selftest":
        return selftest(args)
    return fit_command(args)


if __name__ == "__main__":
    sys.exit(main())
