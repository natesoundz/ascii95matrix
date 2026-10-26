"""Integer zero-walk search primitives."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def walk_known(start, goal, return_traces: bool = False):
    """Walk elementwise from ``start`` to ``goal`` using vectorized rounds."""
    starts = np.asarray(start, dtype=np.int64)
    goals = np.asarray(goal, dtype=np.int64)
    if starts.shape != goals.shape:
        raise ValueError("start and goal must have the same shape")

    position = starts.copy()
    delta = np.zeros(starts.shape, dtype=np.int64)
    moves = np.zeros(starts.shape, dtype=np.int64)
    exponent = np.zeros(starts.shape, dtype=np.int64)
    bracketed = np.zeros(starts.shape, dtype=bool)
    traces = [[] for _ in range(starts.size)] if return_traces else None
    flat_position = position.reshape(-1)
    flat_goal = goals.reshape(-1)
    flat_delta = delta.reshape(-1)
    flat_moves = moves.reshape(-1)
    flat_exponent = exponent.reshape(-1)
    flat_bracketed = bracketed.reshape(-1)
    active = np.flatnonzero(flat_goal != flat_position)

    while active.size:
        direction = np.where(
            flat_goal[active] > flat_position[active],
            1,
            -1,
        )
        step = direction * np.left_shift(np.int64(1), flat_exponent[active])
        flat_position[active] += step
        flat_delta[active] += step
        flat_moves[active] += 1
        if return_traces:
            for idx, move in zip(active, step):
                traces[int(idx)].append(int(move))

        new_direction = np.where(
            flat_goal[active] > flat_position[active],
            1,
            np.where(flat_goal[active] < flat_position[active], -1, 0),
        )
        crossed = new_direction == -direction
        previously_bracketed = flat_bracketed[active]
        flat_bracketed[active] = previously_bracketed | crossed
        crossed_or_bracketed = previously_bracketed | crossed
        flat_exponent[active] = np.where(
            crossed_or_bracketed,
            np.maximum(0, flat_exponent[active] - 1),
            flat_exponent[active] + 1,
        )
        active = active[new_direction != 0]

    if return_traces:
        return delta, moves, traces
    return delta, moves


def walk_edge(
    ahead: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_lines: int,
    max_k: int = 40,
):
    """Find the final true integer offset for monotone line predicates.

    ``max_k`` is the greatest gallop exponent. The probes are
    ``2**(k + 1) - 1`` for ``k = 0 .. max_k``.
    """
    if n_lines < 0:
        raise ValueError("n_lines must be nonnegative")
    if max_k < 0:
        raise ValueError("max_k must be nonnegative")
    edge = np.zeros(n_lines, dtype=np.int64)
    probes = np.zeros(n_lines, dtype=np.int64)
    unbounded = np.zeros(n_lines, dtype=bool)
    active = np.arange(n_lines, dtype=np.int64)
    false_edge = np.zeros(n_lines, dtype=np.int64)
    has_false = np.zeros(n_lines, dtype=bool)

    for k in range(max_k + 1):
        if not active.size:
            break
        offset = np.full(active.size, (1 << (k + 1)) - 1, dtype=np.int64)
        result = np.asarray(ahead(active, offset), dtype=bool)
        if result.shape != active.shape:
            raise ValueError("ahead must return one boolean per probed line")
        probes[active] += 1
        passed = active[result]
        failed = active[~result]
        edge[passed] = offset[result]
        false_edge[failed] = offset[~result]
        has_false[failed] = True
        active = passed

    unbounded[active] = True

    active = np.flatnonzero(has_false)
    while active.size:
        gap = false_edge[active] - edge[active]
        needs_probe = gap > 1
        if not np.any(needs_probe):
            break
        probing = active[needs_probe]
        middle = edge[probing] + (false_edge[probing] - edge[probing]) // 2
        result = np.asarray(ahead(probing, middle), dtype=bool)
        if result.shape != probing.shape:
            raise ValueError("ahead must return one boolean per probed line")
        probes[probing] += 1
        passed = probing[result]
        failed = probing[~result]
        edge[passed] = middle[result]
        false_edge[failed] = middle[~result]
        active = probing

    return edge, probes, unbounded


def line_argmin(L: Callable[[int], int], max_k: int = 40):
    """Return the smallest-absolute-value integer minimizer of convex ``L``."""
    comparisons = 0
    at_zero = L(0)
    at_up = L(1)
    comparisons += 1
    if at_up < at_zero:
        direction = 1
    else:
        at_down = L(-1)
        comparisons += 1
        if at_down < at_zero:
            direction = -1
        else:
            return 0, comparisons

    def improving(indices, offsets):
        nonlocal comparisons
        result = np.empty(indices.shape, dtype=bool)
        for i, offset in enumerate(offsets):
            t = int(offset)
            result[i] = L(direction * (t + 1)) < L(direction * t)
            comparisons += 1
        return result

    edge, _, unbounded = walk_edge(improving, 1, max_k=max_k)
    if unbounded[0]:
        return direction * (int(edge[0]) + 1), comparisons
    return direction * (int(edge[0]) + 1), comparisons
