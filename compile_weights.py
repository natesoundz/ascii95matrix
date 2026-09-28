#!/usr/bin/env python3
"""Compile the declared legacy staged-residual architecture from local evidence.

Copyright (c) 2026 Nathan / Nate Soundz. Compiled Intelligence Lab.
Source equations: Compiled-intelligence-lab@27411dc16e3cb7e286bf3eec9708dae88f62ebc4.
The canonical 437-column workbook is a typed registry. It is not a binary input
to these historical 254-coordinate equations. See README.md for the boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

V = 95
D = 254
OFF = np.asarray([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
H = 2 * V * D
VIEW_KINDS = ("causal", "offset", "four_hole", "full_hole")
VIEW_INDEX = {x: i for i, x in enumerate(VIEW_KINDS)}
SOURCE_COMMIT = "27411dc16e3cb7e286bf3eec9708dae88f62ebc4"
RUN_START = time.monotonic()

OBJECTIVE_SPECS = {
  "LM_HOLE": {
    "visibility": "ALL_EXCEPT_TARGET",
    "constraint_scope": "ALL_POSITIONS",
    "attention_mask": "BIDIRECTIONAL_EXCEPT_TARGET",
    "comparison": "TARGET_VS_ALL_94"
  },
  "SFT_RESPONSE": {
    "visibility": "PREFIX_ONLY",
    "constraint_scope": "ALL_POSITIONS",
    "attention_mask": "CAUSAL",
    "comparison": "TARGET_VS_ALL_94"
  },
  "CHAT_ASSISTANT": {
    "visibility": "PREFIX_ONLY",
    "constraint_scope": "ALL_POSITIONS",
    "attention_mask": "CAUSAL",
    "comparison": "TARGET_VS_ALL_94"
  },
  "PROBLEM_SOLVING": {
    "visibility": "PREFIX_ONLY",
    "constraint_scope": "ALL_POSITIONS",
    "attention_mask": "CAUSAL",
    "comparison": "TARGET_VS_ALL_94"
  },
  "CODE_CAUSAL": {
    "visibility": "PREFIX_ONLY",
    "constraint_scope": "ALL_POSITIONS",
    "attention_mask": "CAUSAL",
    "comparison": "TARGET_VS_ALL_94"
  },
  "CODE_INSTRUCTION": {
    "visibility": "PREFIX_ONLY",
    "constraint_scope": "ALL_POSITIONS",
    "attention_mask": "CAUSAL",
    "comparison": "TARGET_VS_ALL_94"
  }
}
ACTIVE_OBJECTIVES = tuple(OBJECTIVE_SPECS)
OBJ_INDEX = {x: i for i, x in enumerate(ACTIVE_OBJECTIVES)}


def status(message: str) -> None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram = f"{rss / (1024 ** 2 if sys.platform == 'darwin' else 1024):.1f} MiB peak"
    except ImportError:
        ram = "unavailable"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message} | "
          f"elapsed={time.monotonic()-RUN_START:.1f}s | RAM={ram} | VRAM=0 | ETA=unknown",
          flush=True)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ground_truth_positions(record: dict) -> list[int]:
    text = record["text"]
    positions = record.get("ground_truth_positions")
    if positions is None:
        return list(range(len(text)))
    if not isinstance(positions, list) or any(type(i) is not int for i in positions):
        raise ValueError("ground_truth_positions must be a list of integer positions")
    if positions != sorted(set(positions)) or any(i < 0 or i >= len(text) for i in positions):
        raise ValueError("ground_truth_positions must be sorted, unique and inside text")
    return positions


def records(path: Path, objective: str | None = None):
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected a record object")
            if objective is None or record.get("objective") == objective:
                yield record


def validate_dataset(path: Path) -> dict:
    seen = set()
    counts = Counter()
    truths = 0
    for record in records(path):
        for key in ("id", "dataset", "objective", "text"):
            if not isinstance(record.get(key), str):
                raise ValueError(f"record must contain string {key!r}")
        key = (record["dataset"], record["id"])
        if key in seen:
            raise ValueError(f"duplicate dataset/id: {key}")
        seen.add(key)
        if record["objective"] not in OBJECTIVE_SPECS:
            raise ValueError(f"unknown objective {record['objective']!r}")
        text = record["text"]
        if not text or any(not 32 <= ord(c) <= 126 for c in text):
            raise ValueError(f"{key}: text must be nonempty printable ASCII95; no implicit normalization")
        roles = record.get("roles")
        if roles is not None and (not isinstance(roles, list) or len(roles) != len(text)):
            raise ValueError(f"{key}: roles must be a list matching text length")
        if not isinstance(record.get("provenance", {}), dict):
            raise ValueError(f"{key}: provenance must be an object")
        if "weights" in record:
            raise ValueError("record weights are not supported; each declared truth contributes once")
        truths += len(ground_truth_positions(record))
        counts[record["objective"]] += 1
    if truths == 0:
        raise ValueError("dataset contains no ground-truth positions")
    return {"records": len(seen), "ground_truth_positions": truths, "objectives": dict(counts)}


def inspect_registry(path: Path) -> dict:
    """Read the workbook's own types without interpreting eligibility as data."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as z:
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        relationships = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in relationships}
        sheet = next(s for s in workbook.find("s:sheets", ns) if s.get("name") == "ASCII95 Matrix")
        target = targets[sheet.attrib[f"{{{relns}}}id"]]
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(s.itertext()) for s in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        grid = {}
        root = ET.fromstring(z.read(sheet_path))
        for row in root.findall("s:sheetData/s:row", ns):
            for c in row:
                if c.find("s:f", ns) is not None:
                    raise ValueError("registry contains formulas; resolve them in its source workbook first")
                value = c.find("s:v", ns)
                inline = c.find("s:is", ns)
                if c.get("t") == "s" and value is not None:
                    v = shared[int(value.text)]
                elif inline is not None:
                    v = "".join(inline.itertext())
                else:
                    v = value.text if value is not None else None
                if v is not None:
                    grid[c.attrib["r"]] = str(v)
    def colnum(label):
        result = 0
        for c in label:
            result = result * 26 + ord(c) - 64
        return result
    columns = sorted((cell[:-1] for cell in grid if cell.endswith("8")
                      and cell[:-1].isalpha() and colnum(cell[:-1]) >= 6), key=colnum)
    ids = [grid[c + "8"] for c in columns]
    if len(set(ids)) != len(ids) or len(ids) != 437:
        raise ValueError("expected the 437 unique canonical attribute IDs")
    codes = [int(grid[f"C{r}"]) for r in range(10, 105)]
    if codes != list(range(32, 127)):
        raise ValueError("registry rows must be ordered ASCII 32..126")
    allowed = set("X01ERMUVKS")
    values = [[grid[c + str(r)] for c in columns] for r in range(10, 105)]
    counts = Counter(v for row in values for v in row)
    if not set(counts) <= allowed:
        raise ValueError(f"unknown registry statuses: {set(counts) - allowed}")
    math_col = next(c for c in columns if grid[c + "8"] == "character_identity.math_status")
    intrinsic_math = "".join(chr(code) for code, row in zip(codes, range(10, 105))
                             if grid[math_col + str(row)] == "1")
    if intrinsic_math != "+<=>^|~":
        raise ValueError("canonical intrinsic Unicode math invariant changed")
    return {
        "workbook_sha256": file_hash(path), "characters": 95, "attribute_columns": 437,
        "status_counts": dict(sorted(counts.items())), "intrinsic_unicode_math": intrinsic_math,
        "numeric_compiler_input": False,
        "reason": "Typed participation registry. Deferred values, relational projections, measurement transforms and coordinate bindings are not supplied by status cells.",
        "attributes": [{"id": grid[c + "8"], "name": grid[c + "7"],
                        "interaction": grid.get(c + "3"), "eligibility": grid.get(c + "4"),
                        "defer_scope": grid.get(c + "6")} for c in columns],
    }


def load_architecture(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema": "ascii95.legacy_staged_residual.v1", "vocab_size": V,
                "d_model": D, "offsets": OFF.tolist(), "ffn_hidden": H,
                "geometry_dtype": "float64", "detector_dtype": "float64",
                "attention_ffn_dtype": "float32", "substrate": "historical_binary_254"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported architecture {key}: expected {value!r}; no implicit architecture substitution")
    if set(config) != set(expected) | {"matrix_path", "matrix_sha256"}:
        raise ValueError("architecture fields are missing or unknown")
    matrix = path.parent / config["matrix_path"]
    if file_hash(matrix) != config["matrix_sha256"]:
        raise ValueError("historical matrix checksum mismatch")
    lines = matrix.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line]
    if header[:4] != ["index", "ASCII", "Glyph", "Name"]:
        raise ValueError("unexpected matrix columns")
    if [int(row[1]) for row in rows] != list(range(32, 127)):
        raise ValueError("historical matrix ASCII order mismatch")
    A = np.asarray([[int(x) for x in row[4:]] for row in rows], dtype=np.float64)
    if A.shape != (V, D) or not np.all((A == 0) | (A == 1)) or len(set(header[4:])) != D:
        raise ValueError("historical matrix must have 95 rows and 254 named binary columns")
    return config, A, header[4:]


def compile_literal_evidence(sources, A):
    """Compile every declared view exactly, using vectorized sufficient statistics."""
    C_kind = np.zeros((len(VIEW_KINDS), len(OFF), V, V), dtype=np.float64)
    C_obj = np.zeros((len(ACTIVE_OBJECTIVES), len(OFF), V, V), dtype=np.float64)
    C_full = np.zeros((V, V), dtype=np.float64)  # full-view remainder beyond +/-7
    target_weight = np.zeros(V, dtype=np.float64)
    objective_summary = {}
    annotations = {k: Counter() for k in ACTIVE_OBJECTIVES}
    view_counts = {"causal": 0, "offset": 0, "four_hole": 0, "full_hole": 0}

    for obj in ACTIVE_OBJECTIVES:
        chars = 0
        ground_truth_count = 0
        weight_sum = 0.0
        record_count = 0

        for record in sources[obj]():
            record_count += 1
            text = record["text"]
            ids = np.fromiter((ord(ch) - 32 for ch in text), dtype=np.int16)
            n = len(ids)
            chars += n

            for k, v in record.get("provenance", {}).items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        annotations[obj][f"{k}:{kk}"] += int(vv) if isinstance(vv, (int, float)) else 1
                elif isinstance(v, (str, int, float, bool)):
                    annotations[obj][f"{k}:{v}"] += 1

            pos = np.asarray(ground_truth_positions(record), dtype=np.int64)
            if not len(pos):
                continue
            w = np.ones(len(pos), dtype=np.float64)

            y_all = ids[pos].astype(np.int64)
            ground_truth_count += len(pos)
            weight_sum += float(w.sum())
            np.add.at(target_weight, y_all, w)

            view_counts["causal"] += 7 * len(pos)
            view_counts["offset"] += 7 * len(pos)
            view_counts["full_hole"] += len(pos)
            for hole in range(4):
                starts = pos - hole
                view_counts["four_hole"] += int(np.count_nonzero((starts >= 0) & (starts + 4 <= n)))

            # Every declared near-context view. Nested causal/radius views are
            # represented by their exact multiplicity rather than materialized.
            for oi, d in enumerate(OFF.tolist()):
                j = pos + d
                m = (j >= 0) & (j < n)
                if obj != "LM_HOLE" and d > 0:
                    m &= False
                if not np.any(m):
                    continue

                q = pos[m]
                jj = j[m]
                ww = w[m]
                tgt = ids[q].astype(np.int64)
                ctx = ids[jj].astype(np.int64)
                ad = abs(d)

                total_mult = np.zeros(len(q), dtype=np.float64)

                if d < 0:
                    mult = float(8 - ad)
                    np.add.at(C_kind[VIEW_INDEX["causal"], oi], (ctx, tgt), ww * mult)
                    total_mult += mult

                # Radius views obey visibility; positive d only reaches here for LM_HOLE.
                mult = float(8 - ad)
                np.add.at(C_kind[VIEW_INDEX["offset"], oi], (ctx, tgt), ww * mult)
                total_mult += mult

                # Number of legal width-4 windows containing both q and j.
                lo = np.maximum(np.maximum(q - 3, jj - 3), 0)
                hi = np.minimum(np.minimum(q, jj), n - 4)
                four_mult = np.maximum(0, hi - lo + 1).astype(np.float64)
                good4 = four_mult > 0
                if np.any(good4):
                    np.add.at(
                        C_kind[VIEW_INDEX["four_hole"], oi],
                        (ctx[good4], tgt[good4]),
                        ww[good4] * four_mult[good4],
                    )
                    total_mult += four_mult

                # The near portion of the full permitted context.
                np.add.at(C_kind[VIEW_INDEX["full_hole"], oi], (ctx, tgt), ww)
                total_mult += 1.0

                np.add.at(C_obj[OBJ_INDEX[obj], oi], (ctx, tgt), ww * total_mult)

            # Long-range part of the full permitted context.
            # Sparse-target causal records (especially SFT output records) carry
            # long repeated context but only a short canonical truth suffix. For
            # those records scan once instead of allocating an n x 95 prefix matrix.
            if obj != "LM_HOLE" and len(pos) * 4 < max(1, n):
                cuts = np.maximum(0, pos - 7).astype(np.int64)
                start = int(cuts.min())
                end = int(cuts.max())
                base_counts = np.bincount(ids[:start].astype(np.int64), minlength=V).astype(np.int32)
                local_ids = ids[start:end].astype(np.int64)
                local_prefix = np.zeros((len(local_ids) + 1, V), dtype=np.int32)
                if len(local_ids):
                    local_onehot = np.zeros((len(local_ids), V), dtype=np.int8)
                    local_onehot[np.arange(len(local_ids)), local_ids] = 1
                    np.cumsum(local_onehot, axis=0, out=local_prefix[1:])
                far = base_counts[None, :] + local_prefix[cuts - start]
            else:
                onehot = np.zeros((n, V), dtype=np.int32)
                onehot[np.arange(n), ids.astype(np.int64)] = 1
                prefix = np.empty((n + 1, V), dtype=np.int32)
                prefix[0] = 0
                np.cumsum(onehot, axis=0, out=prefix[1:])

                if obj == "LM_HOLE":
                    lo = np.maximum(0, pos - 7)
                    hi = np.minimum(n, pos + 8)
                    near = prefix[hi] - prefix[lo]
                    far = prefix[n][None, :] - near
                else:
                    cut = np.maximum(0, pos - 7)
                    far = prefix[cut]

            for y in np.unique(y_all):
                sel = y_all == y
                C_full[:, int(y)] += (far[sel].astype(np.float64) * w[sel, None]).sum(axis=0)

        objective_summary[obj] = {
            "records": record_count,
            "characters": chars,
            "ground_truth_positions": ground_truth_count,
            "constraint_weight_sum": weight_sum,
            "attention_mask": OBJECTIVE_SPECS[obj]["attention_mask"],
        }
        status(f"EVIDENCE {obj}: chars={chars} ground_truth={ground_truth_count} weight={weight_sum:.1f}")

    C = C_kind.sum(axis=0)
    return C, C_kind, C_obj, C_full, target_weight, objective_summary, annotations, view_counts


def conditional_attribute_profiles(A, C):
    totals = C.sum(axis=2)
    P01 = np.zeros((C.shape[0], V, D), dtype=np.float64)
    for o in range(C.shape[0]):
        nz = np.flatnonzero(totals[o] > 0)
        if len(nz):
            P01[o, nz] = (C[o, nz] / totals[o, nz, None]) @ A
    return P01, totals


def full_profile(A, C_full):
    totals = C_full.sum(axis=1)
    P01 = np.zeros((V, D), dtype=np.float64)
    nz = np.flatnonzero(totals > 0)
    if len(nz):
        P01[nz] = (C_full[nz] / totals[nz, None]) @ A
    return P01, totals


def compat_all_candidates(A, profile):
    # Agreement uses all 254 named dimensions: presence and absence.
    # profile in [0,1], A in {0,1}; score is 1 - mean absolute disagreement.
    return 1.0 - np.mean(np.abs(A - profile[None, :]), axis=1)


def compile_rank_field(A, C, P01, C_full, P_full):
    rank_tensor = np.zeros((V, V, V), dtype=np.float64)  # truth, candidate, rank
    pair_win = np.zeros((V, V), dtype=np.float64)
    pair_total = np.zeros((V, V), dtype=np.float64)
    target_rank_hist = np.zeros((V, V), dtype=np.float64)
    ids = np.arange(V)

    def apply_context(counts_y, profile):
        if counts_y.sum() == 0:
            return
        score = compat_all_candidates(A, profile)
        order = np.lexsort((ids, -score))
        ranks = np.empty(V, dtype=np.int16)
        ranks[order] = np.arange(V, dtype=np.int16)
        for y in np.flatnonzero(counts_y > 0):
            w = float(counts_y[y])
            target_rank_hist[y, ranks[y]] += w
            rank_tensor[y, ids, ranks] += w
            diff = score[y] - score
            pair_win[y] += w * (diff > 0) + 0.5 * w * (diff == 0)
            pair_total[y] += w

    for o in range(len(OFF)):
        for c in range(V):
            apply_context(C[o, c], P01[o, c])
    for c in range(V):
        apply_context(C_full[c], P_full[c])

    pair_rate = np.divide(pair_win, pair_total, out=np.full_like(pair_win, 0.5), where=pair_total > 0)
    np.fill_diagonal(pair_rate, 1.0)

    affordance = np.ones((V, D), dtype=np.float64)
    for y in range(V):
        for k in range(D):
            comps = np.flatnonzero(A[:, k] != A[y, k])
            comps = comps[comps != y]
            if len(comps):
                affordance[y, k] = float(pair_rate[y, comps].mean())

    full_rank_summary = {}
    for y in range(V):
        totals = rank_tensor[y].sum(axis=1)
        mean_rank = np.divide(
            (rank_tensor[y] * np.arange(1, V + 1)[None, :]).sum(axis=1),
            totals,
            out=np.full(V, np.inf),
            where=totals > 0,
        )
        others = [z for z in range(V) if z != y]
        others.sort(key=lambda z: (mean_rank[z], z))
        full_rank_summary[chr(y + 32)] = {
            "target_rank_histogram": target_rank_hist[y].tolist(),
            "closest10_by_contextual_mean_rank": [chr(z + 32) for z in others[:10]],
            "middle74_by_contextual_mean_rank": [chr(z + 32) for z in others[10:84]],
            "furthest10_by_contextual_mean_rank": [chr(z + 32) for z in others[84:94]],
        }
    return rank_tensor, target_rank_hist, pair_rate, affordance, full_rank_summary


def compile_continuous_geometry(A, C, P01, C_full, P_full, affordance):
    support_sum = np.zeros((V, D), dtype=np.float64)
    support_n = np.zeros(V, dtype=np.float64)

    for o in range(len(OFF)):
        for c in range(V):
            w = C[o, c]  # one weight per actual target
            if not np.any(w):
                continue
            support_sum += w[:, None] * P01[o, c][None, :]
            support_n += w

    for c in range(V):
        w = C_full[c]
        if not np.any(w):
            continue
        support_sum += w[:, None] * P_full[c][None, :]
        support_n += w

    support = np.divide(
        support_sum,
        support_n[:, None],
        out=np.full((V, D), 0.5, dtype=np.float64),
        where=support_n[:, None] > 0,
    )

    # One continuous representation. Historical membership only fixes coordinate
    # orientation; magnitude is compiled from empirical support x affordance.
    sign = np.where(A > 0.5, 1.0, -1.0)
    side_support = np.where(A > 0.5, support, 1.0 - support)
    eps = 1e-6
    magnitude = np.maximum(eps, side_support * affordance)
    E = sign * magnitude

    # Every character must have a unique, full-rank continuous row.
    rank = int(np.linalg.matrix_rank(E, tol=1e-10))
    if rank < V:
        raise RuntimeError(f"compiled E rank {rank} < {V}; refusing hidden remap/fallback")
    return E, support, side_support


def compile_detector_from_E(E):
    # Direct least-norm left inverse. Runtime detection consumes only E.
    W = np.linalg.pinv(E, rcond=1e-12)
    got = E @ W
    err = float(np.max(np.abs(got - np.eye(V))))
    if err > 1e-8:
        raise RuntimeError(f"continuous token detector error {err}")
    return W, np.zeros(V, dtype=np.float64), err


def compile_relation_specificity(C):
    """A13-style forward/reverse/mutual relation measurements over the full evidence field."""
    context_totals = C.sum(axis=2)
    target_totals = C.sum(axis=1)
    forward = np.divide(
        C, context_totals[:, :, None],
        out=np.zeros_like(C), where=context_totals[:, :, None] > 0,
    )
    reverse = np.divide(
        C, target_totals[:, None, :],
        out=np.zeros_like(C), where=target_totals[:, None, :] > 0,
    )
    mutual = np.sqrt(forward * reverse)
    return forward, reverse, mutual


def profile_to_state(A_row, profile, affordance_row):
    """Map a measured 254-D support profile into the declared signed coordinate system."""
    profile = np.clip(np.asarray(profile, dtype=np.float64), 0.0, 1.0)
    sign = np.where(A_row > 0.5, 1.0, -1.0)
    side = np.where(A_row > 0.5, profile, 1.0 - profile)
    magnitude = np.maximum(1e-6, np.clip(side * affordance_row, 0.0, 1.0))
    return sign * magnitude


def compile_attention_from_embedding_residual(E, A, C, P01, totals, affordance, mutual):
    """Compile Q/K/V only after E is frozen.

    Q/K encode A13-style mutual specificity. V stores an ADDITIVE residual:
    the measured contextual state for a relation minus the already-frozen
    embedding state. Attention therefore writes only what E did not.
    """
    WQ = np.eye(V, dtype=np.float32)
    WK = np.zeros((len(OFF), V, V), dtype=np.float32)
    WV = np.zeros((len(OFF), V, D), dtype=np.float32)
    valid = np.zeros((len(OFF), V, V), dtype=np.uint8)
    relevance = mutual.astype(np.float64, copy=True)

    for o in range(len(OFF)):
        for c in range(V):
            if totals[o, c] <= 0:
                continue

            good = relevance[o, c] > 0
            valid[o, good, c] = 1
            WK[o, c, good] = np.log(relevance[o, c, good]).astype(np.float32)

            weights = C[o, c] * relevance[o, c]
            ys = np.flatnonzero(weights > 0)
            if not len(ys):
                continue
            residuals = []
            ww = []
            for y in ys:
                required = profile_to_state(A[y], P01[o, c], affordance[y])
                residuals.append(required - E[y])
                ww.append(float(weights[y]))
            WV[o, c] = np.average(
                np.asarray(residuals, dtype=np.float64),
                axis=0,
                weights=np.asarray(ww, dtype=np.float64),
            ).astype(np.float32)

    return WQ, WK, WV, valid, relevance


def ffn_mats(corr):
    """Exact signed residual correction with paired ReLU units.

    For d = base[k] - hA[k]:
      relu(d) - relu(-d) = d
    so paired units implement corr[y,k] * d for both positive and negative
    continuous embedding coordinates.
    """
    gate = 8.0
    W1 = np.zeros((2 * D + V, H), dtype=np.float32)
    b1 = np.full(H, -gate, dtype=np.float32)
    W2 = np.zeros((H, D), dtype=np.float32)
    b2 = np.zeros(D, dtype=np.float32)
    ks = np.arange(D)
    block = V * D

    for y in range(V):
        pos_cols = y * D + ks
        neg_cols = block + y * D + ks

        # Positive part relu(base - hA).
        W1[ks, pos_cols] = 1.0
        W1[D + ks, pos_cols] = -1.0
        W1[2 * D + y, pos_cols] = gate
        W2[pos_cols, ks] = corr[y]

        # Negative part relu(hA - base), subtracted at output.
        W1[ks, neg_cols] = -1.0
        W1[D + ks, neg_cols] = 1.0
        W1[2 * D + y, neg_cols] = gate
        W2[neg_cols, ks] = -corr[y]

    return W1, b1, W2, b2


def compile_ffn_from_post_attention_residual(sources, A, E, P01, totals, affordance, mutual, WV):
    """Freeze attention, replay all ground truth, then compile the remaining FFN write.

    The replay is vectorized per source record but is mathematically identical
    to processing the occurrences one by one. For each token/dimension the
    paired-ReLU FFN has the declared form

        delta_ffn = c[y,k] * (E[y,k] - hA[k])

    and c is prescribed directly by the normal equation of the measured
    post-attention residual field. There is no iterative update, optimizer,
    gradient, epoch, or loss-training loop.
    """
    num = np.zeros((V, D), dtype=np.float64)
    den = np.zeros((V, D), dtype=np.float64)
    residual2 = np.zeros((V, D), dtype=np.float64)
    embedding2 = np.zeros((V, D), dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    total_occurrences = 0
    attention_improved = 0

    for obj in ACTIVE_OBJECTIVES:
        obj_occ = 0
        for record in sources[obj]():
            ids = np.fromiter((ord(ch) - 32 for ch in record["text"]), dtype=np.int16)
            pos = np.asarray(ground_truth_positions(record), dtype=np.int64)
            if not len(pos):
                continue

            y = ids[pos].astype(np.int64)
            npos = len(pos)
            profile_num = np.zeros((npos, D), dtype=np.float64)
            profile_den = np.zeros(npos, dtype=np.float64)
            attention_num = np.zeros((npos, D), dtype=np.float64)
            attention_den = np.zeros(npos, dtype=np.float64)

            for oi, d in enumerate(OFF.tolist()):
                j = pos + d
                visible = (j >= 0) & (j < len(ids))
                if obj != "LM_HOLE" and d > 0:
                    visible &= False
                if not np.any(visible):
                    continue

                rows = np.flatnonzero(visible)
                c = ids[j[visible]].astype(np.int64)
                yy = y[visible]
                rel = mutual[oi, c, yy].astype(np.float64)
                good = rel > 0.0
                if not np.any(good):
                    continue

                rows = rows[good]
                c = c[good]
                rel = rel[good]
                profile_num[rows] += rel[:, None] * P01[oi, c]
                profile_den[rows] += rel
                attention_num[rows] += rel[:, None] * WV[oi, c].astype(np.float64)
                attention_den[rows] += rel

            base = E[y]
            profile = np.divide(
                profile_num,
                profile_den[:, None],
                out=np.zeros_like(profile_num),
                where=profile_den[:, None] > 0,
            )

            sign = np.where(A[y] > 0.5, 1.0, -1.0)
            side = np.where(A[y] > 0.5, profile, 1.0 - profile)
            required = sign * np.maximum(
                1e-6,
                np.clip(side * affordance[y], 0.0, 1.0),
            )
            no_requirement = profile_den <= 0
            if np.any(no_requirement):
                required[no_requirement] = base[no_requirement]

            delta_a = np.divide(
                attention_num,
                attention_den[:, None],
                out=np.zeros_like(attention_num),
                where=attention_den[:, None] > 0,
            )
            hA = base + delta_a

            embedding_residual = required - base
            remaining = required - hA
            direction = base - hA

            np.add.at(embedding2, y, embedding_residual * embedding_residual)
            np.add.at(residual2, y, remaining * remaining)
            np.add.at(num, y, direction * remaining)
            np.add.at(den, y, direction * direction)
            np.add.at(counts, y, 1)

            before_l2 = np.einsum("ij,ij->i", embedding_residual, embedding_residual)
            after_l2 = np.einsum("ij,ij->i", remaining, remaining)
            attention_improved += int(np.count_nonzero(after_l2 <= before_l2 + 1e-15))
            total_occurrences += npos
            obj_occ += npos

        status(f"RESIDUAL REPLAY {obj}: occurrences={obj_occ}")

    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-18)
    post2 = np.maximum(0.0, residual2 - 2.0 * corr * num + corr * corr * den)
    coord_n = max(1, total_occurrences * D)

    audit = {
        "occurrences": int(total_occurrences),
        "embedding_residual_rms": float(np.sqrt(embedding2.sum() / coord_n)),
        "post_attention_residual_rms": float(np.sqrt(residual2.sum() / coord_n)),
        "post_ffn_residual_rms": float(np.sqrt(post2.sum() / coord_n)),
        "attention_nonworse_occurrences": int(attention_improved),
        "attention_nonworse_fraction": float(attention_improved / max(1, total_occurrences)),
        "ffn_projection_coefficient_min": float(corr.min()),
        "ffn_projection_coefficient_max": float(corr.max()),
        "ffn_projection_nonzero": int(np.count_nonzero(np.abs(corr) > 0)),
        "stage_order": ["EMBEDDING_FROZEN", "ATTENTION_FROZEN", "FFN_COMPILED"],
        "replay_mode": "VECTORIZED_PER_RECORD_EXACT_SAME_EQUATIONS",
    }
    return corr.astype(np.float32), ffn_mats(corr.astype(np.float32)), audit



def verify_arrays(arrays: dict) -> dict:
    E = arrays["token_geometry"]
    Wd = arrays["token_detector_W"]
    WQ, WK = arrays["attention_WQ"], arrays["attention_WK"]
    W1, b1 = arrays["ffn_W1"], arrays["ffn_b1"]
    W2, b2 = arrays["ffn_W2"], arrays["ffn_b2"]
    corr = arrays["character_ffn_correction"]
    relevance = arrays["mutual_relation"]
    detector_error = float(np.abs(E @ Wd + arrays["token_detector_b"] - np.eye(V)).max())
    qk_error = 0.0
    for o in range(len(OFF)):
        scores = WQ.astype(np.float64) @ WK[o].astype(np.float64).T
        mask = relevance[o].T > 0
        if mask.any():
            qk_error = max(qk_error, float(np.abs(scores[mask] - np.log(relevance[o].T[mask])).max()))
    ffn_error = 0.0
    # Exercise positive and negative writes for all characters and coordinates.
    for factor in (0.37, 1.63):
        hA = E * factor
        x = np.concatenate([E, hA, np.eye(V)], axis=1)
        hidden = np.maximum(x @ W1 + b1, 0.0)
        got = hidden @ W2 + b2
        want = corr * (E - hA)
        ffn_error = max(ffn_error, float(np.abs(got - want).max()))
    finite = all(bool(np.isfinite(v).all()) for v in arrays.values()
                 if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number))
    rank = int(np.linalg.matrix_rank(E))
    return {"detector_max_abs_error": detector_error, "qk_max_abs_error": qk_error,
            "ffn_max_abs_error": ffn_error, "finite": finite, "embedding_rank": rank,
            "passed": bool(detector_error < 1e-8 and qk_error < 2e-6 and
                           ffn_error < 2e-6 and finite and rank == V),
            "scope": "numerical realization of declared historical equations; not target-free prediction"}


def compile_dataset(dataset: Path, architecture: Path, output: Path) -> dict:
    config, A, names = load_architecture(architecture)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty to preserve existing artifacts")
    output.mkdir(parents=True, exist_ok=True)
    # Copy literal records once. Both compilation passes read this frozen evidence.
    snapshot = output / "evidence.jsonl"
    shutil.copyfile(dataset, snapshot)
    summary = validate_dataset(snapshot)
    source_hash = file_hash(snapshot)
    status(f"VALIDATED {summary['records']} records, {summary['ground_truth_positions']} truth positions")
    sources = {obj: (lambda obj=obj: records(snapshot, obj)) for obj in ACTIVE_OBJECTIVES}
    C, C_kind, C_obj, C_full, target_weight, dataset_summary, annotations, view_counts = compile_literal_evidence(sources, A)
    P01, totals = conditional_attribute_profiles(A, C)
    Pfull, _ = full_profile(A, C_full)
    rank_tensor, rank_hist, pair_rate, affordance, _ = compile_rank_field(A, C, P01, C_full, Pfull)
    forward, reverse, mutual = compile_relation_specificity(C)
    status("COMPILE geometry using the historical sign/support/affordance equation")
    E, support, _ = compile_continuous_geometry(A, C, P01, C_full, Pfull, affordance)
    Wd, bd, _ = compile_detector_from_E(E)
    status("FREEZE E; COMPILE additive attention residual")
    WQ, WK, WV, valid, relevance = compile_attention_from_embedding_residual(E, A, C, P01, totals, affordance, mutual)
    status("FREEZE attention; REPLAY evidence for FFN residual")
    corr, (W1, b1, W2, b2), residual = compile_ffn_from_post_attention_residual(sources, A, E, P01, totals, affordance, mutual, WV)
    if file_hash(snapshot) != source_hash:
        raise ValueError("evidence changed during compilation")
    usage = np.asarray([[obj == "LM_HOLE" or d < 0 for d in OFF] for obj in ACTIVE_OBJECTIVES], dtype=np.uint8)
    arrays = dict(token_geometry=E, attribute_names=np.asarray(names), semantic_substrate_A=A.astype(np.uint8),
        offsets=OFF, token_detector_W=Wd, token_detector_b=bd,
        attention_WQ=WQ, attention_WK=WK, attention_WV=WV, attention_valid=valid,
        attention_value_semantics=np.asarray("ADDITIVE_POST_EMBEDDING_RESIDUAL"),
        objective_names=np.asarray(ACTIVE_OBJECTIVES), objective_head_usage=usage,
        residual_attention_I=np.eye(D, dtype=np.float32), residual_ffn_I=np.eye(D, dtype=np.float32),
        ffn_W1=W1, ffn_b1=b1, ffn_W2=W2, ffn_b2=b2,
        character_ffn_correction=corr, empirical_counts=C, empirical_counts_by_view=C_kind,
        empirical_counts_by_objective=C_obj, empirical_full_context_counts=C_full,
        empirical_attribute_profiles=P01, empirical_full_attribute_profiles=Pfull,
        forward_relation=forward, reverse_relation=reverse, mutual_relation=mutual,
        target_weight=target_weight, target_rank_histogram=rank_hist,
        contextual_rank_tensor=rank_tensor, pairwise_affordance_rate=pair_rate,
        attribute_affordance=affordance, attribute_support=support)
    status("VERIFY the arrays that will be serialized")
    checks = verify_arrays(arrays)
    checks["ffn_residual_nonworse"] = bool(residual["post_ffn_residual_rms"] <= residual["post_attention_residual_rms"] + 1e-12)
    checks["passed"] &= checks["ffn_residual_nonworse"]
    report = {"status": "NUMERIC_PASS" if checks["passed"] else "FAIL", "source_commit": SOURCE_COMMIT,
        "architecture": config, "dataset_sha256": source_hash, "input_summary": summary,
        "datasets": dataset_summary, "view_counts": view_counts, "verification": checks,
        "residual_stage_audit": residual, "canonical_registry_consumed": False,
        "target_free_prediction_verified": False, "generalization_evaluated": False,
        "limitations": ["Fixed historical binary 254-dimensional substrate.",
            "Embedding uses a prescribed support/affordance formula, not a global constraint-satisfaction solve.",
            "Contextual states and attention relevance are conditioned on the supplied target.",
            "Source annotations are retained as metadata, not converted to executable semantic constraints.",
            "No canonical 437-column binding or complete conversational coding runtime."]}
    if checks["passed"]:
        temporary = output / "weights.pending.npz"
        np.savez_compressed(temporary, **arrays)
        # Verify exact serialization: every saved field must equal the checked field.
        with np.load(temporary, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError("saved weight arrays differ from verified arrays")
        temporary.replace(output / "weights.npz")
        report["weights_sha256"] = file_hash(output / "weights.npz")
    (output / "compile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise RuntimeError("numerical verification failed; no weights.npz was published")
    status("DONE: weights.npz and compile_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="inspect canonical workbook types, without inventing numeric values")
    inspect.add_argument("workbook", type=Path)
    inspect.add_argument("--report", type=Path)
    compile_parser = subs.add_parser("compile", help="compile the explicitly declared historical architecture")
    compile_parser.add_argument("--dataset", type=Path, required=True)
    compile_parser.add_argument("--architecture", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    verify = subs.add_parser("verify", help="verify serialized weight equations")
    verify.add_argument("weights", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            report = inspect_registry(args.workbook)
            if args.report:
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if k != "attributes"}, indent=2))
        elif args.command == "compile":
            result = compile_dataset(args.dataset, args.architecture, args.output)
            print(json.dumps({"status": result["status"], "verification": result["verification"]}, indent=2))
        else:
            with np.load(args.weights, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            result = verify_arrays(arrays)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
    except (ValueError, OSError, KeyError, RuntimeError, StopIteration) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
