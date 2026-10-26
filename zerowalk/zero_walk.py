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

Input : two JSON files  {"shape": [...], "data": [...]}  (old, new), same shape.
        Integers are used as-is; floats are quantized with --unit.
Output: one JSON file, checkpointed atomically after every chunk, resumable.

Usage:
  python zero_walk.py run  --old old.json --new new.json --out delta.json
  python zero_walk.py selftest
"""

import argparse
import json
import os
import signal
import sys
import time

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
    """First SIGINT: finish the current marble, save, exit cleanly.
    Second SIGINT: hard exit now (last checkpoint is already on disk)."""
    def __init__(self):
        self.strikes = 0
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, signum, frame):
        self.strikes += 1
        if self.strikes >= 2:
            sys.stderr.write("\nsecond interrupt: exiting now\n")
            os._exit(130)
        sys.stderr.write("\ninterrupt: saving at the next marble boundary "
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
    last_save = time.time()

    while state["done"] < marbles and not guard.requested:
        m = state["done"]
        lo = m * r
        deltas, traces, edges = read_marble(old[lo:lo + r], new[lo:lo + r])
        state["delta"].extend(deltas)
        state["traces"].extend(traces)
        state["edges"].append(edges)
        state["done"] = m + 1
        prog.show(state["done"], "marble %d" % m)
        if state["done"] % args.every == 0 or time.time() - last_save > 2.0:
            atomic_write_json(args.out, state)
            last_save = time.time()

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
    r.add_argument("--every", type=int, default=64, help="checkpoint every N marbles")
    s = sub.add_parser("selftest")
    s.add_argument("--range", type=int, default=150)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else selftest(args)


if __name__ == "__main__":
    sys.exit(main())
